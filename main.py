import os

# Keep CPU thread pools small *before* torch is imported: fewer threads means
# less per-thread scratch memory on small instances (e.g. Render 512 MB).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import io
import time
import base64
import threading
from typing import Dict, Any, List

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
cv2.setNumThreads(1)
Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, "best_model.pt"))
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "1IR15vcwLLRoW75MYqOI57nUbTE3fwQYg")

IMG_SIZE = 224
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# ImageFolder order (alphabetical) used during training — must not change.
CLASS_NAMES = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]

# Normalised-entropy thresholds (0 = certain, 1 = uniform over 4 classes).
ENTROPY_MODERATE = 0.35
ENTROPY_HIGH = 0.60

CLINICAL_INFO = {
    "Non Demented": {
        "cdr": "CDR 0",
        "severity": "No dementia pattern",
        "description": "The model did not find the structural pattern it associates with dementia in this slice.",
    },
    "Very mild Dementia": {
        "cdr": "CDR 0.5",
        "severity": "Very mild",
        "description": "The model associates this slice with the very-mild (CDR 0.5) group, e.g. subtle atrophy or ventricular enlargement.",
    },
    "Mild Dementia": {
        "cdr": "CDR 1",
        "severity": "Mild",
        "description": "The model associates this slice with the mild (CDR 1) group, e.g. cortical thinning and ventricular dilation.",
    },
    "Moderate Dementia": {
        "cdr": "CDR 2",
        "severity": "Moderate",
        "description": "The model associates this slice with the moderate (CDR 2) group, e.g. pronounced generalised atrophy.",
    },
}

SAMPLE_CASES = [
    {"id": "non_demented", "filename": "non_dem.png", "label": "Non Demented",
     "expected_class": "Non Demented"},
    {"id": "very_mild_dementia", "filename": "ver_mild_dem.png", "label": "Very Mild",
     "expected_class": "Very mild Dementia"},
    {"id": "mild_dementia", "filename": "mild_dem.png", "label": "Mild",
     "expected_class": "Mild Dementia"},
    {"id": "moderate_dementia", "filename": "moderate_dem.png", "label": "Moderate",
     "expected_class": "Moderate Dementia"},
]
for _s in SAMPLE_CASES:
    _s["thumbnail_url"] = f"/static/samples/{_s['filename']}"

DISCLAIMER = ("Academic screening prototype trained on OASIS-1 MRI slices. "
              "Not a medical device and not a diagnosis.")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _ensure_weights() -> None:
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    import gdown  # imported lazily: only needed when weights are missing
    print(f"Downloading baseline weights to {MODEL_PATH} ...", flush=True)
    gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        raise RuntimeError("Model weights download failed (check GDRIVE_FILE_ID / sharing permissions).")


def _load_model() -> nn.Module:
    _ensure_weights()
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))

    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except Exception:
        # Older checkpoints may contain non-tensor objects; only load trusted files.
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt)

    net.eval()
    # Grad-CAM only needs gradients w.r.t. activations, not parameters.
    # Freezing params avoids allocating ~45 MB of weight gradients per request.
    for p in net.parameters():
        p.requires_grad_(False)
    return net


device = torch.device("cpu")  # Render free/starter instances are CPU-only
print("Loading ResNet-18 baseline ...", flush=True)
model = _load_model().to(device)
print("Model ready.", flush=True)

_infer_lock = threading.Lock()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Grad-CAM (layer4[-1])
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, net: nn.Module, target_layer: nn.Module):
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out
        if out.requires_grad:
            out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def compute(self, logits: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a smooth CAM in [0, 1] of shape (IMG_SIZE, IMG_SIZE)."""
        logits[0, class_idx].backward()
        acts = self.activations[0].detach()          # [C, 7, 7]
        grads = self.gradients[0].detach()           # [C, 7, 7]
        weights = grads.mean(dim=(1, 2))             # [C]
        cam = F.relu((weights[:, None, None] * acts).sum(0)).cpu().numpy()
        self.activations = self.gradients = None

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=6)
        cam = np.clip(cam, 0, None)
        if cam.max() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam.astype(np.float32)


cam_engine = GradCAM(model, model.layer4[-1])


def _overlay_to_data_uri(rgb: Image.Image, cam: np.ndarray, max_side: int = 448) -> str:
    """Blend a JET heatmap onto the image at its original aspect ratio.

    The CAM is computed on the 224x224 model input; it is resized back to the
    image's own proportions so the overlay lines up with the original view.
    """
    w, h = rgb.size
    scale = min(1.0, max_side / max(w, h))
    out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))
    base = np.asarray(rgb.resize((out_w, out_h), Image.BILINEAR), dtype=np.float32) / 255.0
    cam = np.clip(cv2.resize(cam, (out_w, out_h), interpolation=cv2.INTER_CUBIC), 0, 1)

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = (0.55 * cam)[..., None]
    overlay = np.clip(base * (1 - alpha) + heat * alpha, 0, 1)
    buf = io.BytesIO()
    Image.fromarray(np.uint8(overlay * 255)).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
def _to_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(image, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(arr)
        image = Image.fromarray(arr.astype(np.uint8), mode="L")
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(bg, image)
    return image.convert("RGB")


def check_mri_plausibility(base_rgb: np.ndarray) -> Dict[str, Any]:
    """Cheap heuristics: is this plausibly a grayscale brain MRI slice?"""
    spread = base_rgb.max(axis=2) - base_rgb.min(axis=2)
    color_spread = float(spread.mean())                      # ~0 for grayscale
    gray = base_rgb.mean(axis=2)
    dark_fraction = float((gray < 0.08).mean())              # MRI slices have dark background
    contrast = float(gray.std())

    reasons: List[str] = []
    if color_spread > 0.04:
        reasons.append("image contains colour; MRI slices are grayscale")
    if dark_fraction < 0.15:
        reasons.append("little dark background around the head, unlike typical MRI slices")
    if contrast < 0.05:
        reasons.append("very low contrast / nearly blank image")

    is_likely_mri = not reasons
    return {
        "is_likely_mri": is_likely_mri,
        "is_grayscale": color_spread <= 0.04,
        "color_spread_score": round(color_spread, 4),
        "dark_background_fraction": round(dark_fraction, 3),
        "contrast_score": round(contrast, 3),
        "warning": None if is_likely_mri else
        "This does not look like a standard grayscale brain MRI slice ("
        + "; ".join(reasons) + "). The prediction is likely unreliable.",
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(image: Image.Image) -> Dict[str, Any]:
    # The whole pipeline runs under one lock: Grad-CAM hooks are stateful, and
    # processing one request at a time keeps peak memory flat on small instances.
    with _infer_lock:
        return _run_inference_locked(image)


def _run_inference_locked(image: Image.Image) -> Dict[str, Any]:
    start = time.perf_counter()
    rgb = _to_rgb(image)
    base_rgb = np.asarray(rgb.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    x = transform(rgb).unsqueeze(0).to(device)
    x.requires_grad_(True)  # lets gradients flow to layer4 with frozen params

    logits = model(x)                                        # single forward pass
    probs = F.softmax(logits.detach(), dim=1)[0]
    pred_idx = int(probs.argmax())
    cam = cam_engine.compute(logits, pred_idx)
    del logits, x

    p = probs.numpy().astype(np.float64)
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(CLASS_NAMES)))
    entropy = round(min(max(entropy, 0.0), 1.0), 3)
    level = "high" if entropy >= ENTROPY_HIGH else "moderate" if entropy >= ENTROPY_MODERATE else "low"

    pred_class = CLASS_NAMES[pred_idx]
    probs_dict = {c: round(float(p[i]) * 100, 2) for i, c in enumerate(CLASS_NAMES)}
    ranked = sorted(({"class_name": c, "probability": v} for c, v in probs_dict.items()),
                    key=lambda r: r["probability"], reverse=True)

    return {
        "prediction": pred_class,
        "confidence": probs_dict[pred_class],
        "uncertainty_entropy": entropy,
        "uncertainty_level": level,
        "uncertainty_warning": None if level == "low" else
        f"Prediction uncertainty is {level} (normalised entropy {entropy}); the model is not confident.",
        "mri_check": check_mri_plausibility(base_rgb),
        "gradcam_overlay": _overlay_to_data_uri(rgb, cam),
        "probabilities": probs_dict,
        "ranked_probabilities": ranked,
        "clinical_info": CLINICAL_INFO[pred_class],
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "device": str(device),
        "model": "ResNet-18 baseline (class-weighted CE)",
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NeuroScan AI - OASIS Dementia Classifier",
    description="Explainable ResNet-18 dementia staging with Grad-CAM, uncertainty and MRI checks.",
    version="1.2.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SamplePredictRequest(BaseModel):
    sample_id: str


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "architecture": "ResNet-18 (Baseline)",
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "device": str(device),
    }


@app.get("/api/samples")
def get_samples() -> List[Dict[str, Any]]:
    return SAMPLE_CASES


# Plain `def` endpoints run in FastAPI's threadpool, so CPU-bound inference
# never blocks the event loop.
@app.post("/api/predict-sample")
def predict_sample(payload: SamplePredictRequest):
    match = next((s for s in SAMPLE_CASES if s["id"] == payload.sample_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Sample not found")
    path = os.path.join(SAMPLES_DIR, match["filename"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample image file missing on server")
    with Image.open(path) as img:
        result = run_inference(img)
    result["sample_info"] = match
    return result


@app.post("/api/predict")
def predict_file(file: UploadFile = File(...)):
    if file.content_type and not (file.content_type.startswith("image/")
                                  or file.content_type == "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG/PNG)")
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image size exceeds 10MB limit")
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Please upload a JPEG or PNG.")
    result = run_inference(img)
    result["filename"] = file.filename
    return result


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "Frontend not yet mounted"})
