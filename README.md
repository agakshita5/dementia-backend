# NeuroScan AI — Dementia-Stage Classification Backend

Explainable 4-class dementia staging on OASIS-1 brain MRI slices: a ResNet-18 classifier served through FastAPI, with Grad-CAM, an uncertainty score and an MRI input check.

> Academic screening prototype. Not a medical device and not a diagnosis.

## Repository layout

| Path | Purpose |
|---|---|
| `main.py` | FastAPI app: model loading, inference, Grad-CAM, uncertainty, MRI check, API routes |
| `static/index.html` | Web UI served by the same app |
| `static/samples/` | One OASIS slice per class for one-click demos |
| `models/best_model.pt` | Deployed weights (baseline ResNet-18); auto-downloaded via `gdown` if missing |
| `src/dataset.py` | Subject-level stratified train/val/test split, transforms, DataLoaders |
| `src/train.py` | Baseline training script (run on Colab) |
| `outputs/` | Test classification reports and confusion matrices (baseline and focal) |
| `baseline_dementia.ipynb` | Colab: runs `src/train.py`, evaluates, then the extra class-weighted fine-tune experiment |
| `focal_loss_dementia.ipynb` | Colab: focal-loss training and evaluation |
| `requirements.txt`, `.python-version` | Server dependencies (CPU-only PyTorch) and Python 3.11 pin |
| `src/device_utils`, `src/patient_id` |  Two helper modules | 

Training runs on Google Colab (GPU) only.

---

## 1. Data and training (Colab)

### Data (`src/dataset.py`)
- **Dataset:** OASIS-1 MRI slices (Kaggle `ninadaithal/imagesoasis`), loaded with `torchvision.datasets.ImageFolder`.
- **Class index order:** alphabetical folder order: `0 Mild Dementia, 1 Moderate Dementia, 2 Non Demented, 3 Very mild Dementia`. The backend's `CLASS_NAMES` must keep this order.
- **Subject-level split** (`split_mode="subject"`, seed 42):
  - Every slice is mapped to its OASIS subject ID (`extract_subject_id`), and each subject must have exactly one label.
  - *Subjects*, not slices, are split 70 / 15 / 15, stratified by label: 347 subjects → 242 train / 52 val / 53 test.
  - This means no patient appears in both training and test, so accuracy isn't inflated by recognising the same person.
- **Transforms:**
  - Train: `Resize(224)`, `RandomHorizontalFlip(0.5)`, `RandomRotation(8°)`, `ColorJitter(brightness 0.1, contrast 0.1)`, `ToTensor`, ImageNet `Normalize`.
  - Eval: `Resize(224)`, `ToTensor`, ImageNet `Normalize`. The server uses exactly this eval pipeline.
- **Severe class imbalance.** The test set has 13,237 slices from 53 subjects: 76.5% Non Demented, 16.1% Very mild, 5.5% Mild and 1.8% Moderate. By subject that is 40 Non, 9 Very mild, 3 Mild and only **1 Moderate**, so minority-class test metrics rest on very few people.

### Models compared
All three use torchvision `resnet18` with ImageNet weights and the final layer replaced by `fc = Linear(512, 4)`.

1. **Baseline: deployed (`src/train.py`).**
   - 5 epochs, batch size 32, Adam at lr 1e-4.
   - Best checkpoint chosen by validation accuracy (0.7997) and saved as `best_model.pt`.
   - The loss is **class-weighted cross-entropy**.
   - Weights: `w_c = N / (K · n_c)`, then divided by their mean → `[0.247, 3.641, 0.019, 0.093]`.
2. **Extra class-weighted fine-tune.**
   - Starts from the baseline and runs 1 more epoch with Adam at lr 1e-5.
   - Uses *un-normalised* inverse-frequency weights `N / (K · n_c)`.
3. **Focal Loss.**
   - `FL = α_t · (1 − p_t)^γ · CE`, with γ = 2 and inverse-frequency α.
   - Up to 12 epochs, Adam at lr 1e-4, starting from ImageNet weights. The saved run stopped during epoch 11; the evaluated checkpoint is the best by validation accuracy (epoch 9, 0.7802).

### Test results (53 held-out subjects)

| Model | Recall Non | Recall Very mild | Recall Mild | Recall Moderate | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| **Baseline (deployed)** | 0.88 | 0.49 | 0.11 | 0.00 | **0.76** | **0.37** |
| + extra weighted fine-tune | 0.85 | 0.43 | 0.09 | 0.00 | 0.73 | 0.34 |
| Focal Loss | 0.96 | 0.07 | 0.03 | 0.01 | 0.75 | 0.26 |

Full reports and confusion matrices are in `outputs/`.

**Why the baseline is deployed:**
- It has the best macro F1 and the most balanced recall.
- Focal Loss reached 0.67 precision on Moderate, but only by rarely predicting Moderate at all (1% recall). It also collapsed Very-mild recall (0.49 → 0.07).
- Mild and Moderate stay hard for every variant. The baseline confusion matrix shows Moderate slices mostly predicted as Mild (108 / 244) or Non Demented (103 / 244).
- That is why the backend reports uncertainty instead of only a label.

---

## 2. Model serving (`main.py`)

### Startup
1. Before importing torch, set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, then call `torch.set_num_threads(1)` and `cv2.setNumThreads(1)`. Small thread pools keep memory low on a 512 MB instance.
2. If `models/best_model.pt` is missing or smaller than 1 MB, download it from Google Drive with `gdown`. `MODEL_PATH` and `GDRIVE_FILE_ID` can be overridden with environment variables.
3. Build `resnet18(weights=None)` and set `fc = Linear(512, 4)`.
4. Load the checkpoint with `torch.load(map_location="cpu", weights_only=True)`. The loader accepts three formats:
   - a raw `state_dict`
   - a dict wrapping it under `model_state_dict` or `state_dict`
   - weights saved with `DataParallel`, whose keys have a `module.` prefix (stripped)
5. Call `model.eval()` and **freeze all parameters** (`requires_grad=False`). Grad-CAM still works because gradients are taken with respect to the input and activations, not the weights. This avoids allocating about 45 MB of weight gradients per request.
6. Register the Grad-CAM hook on `model.layer4[-1]` once.

### Preprocessing (per request)
1. Undo EXIF rotation, then normalise the image mode:
   - 16-bit or float images: min-max scaled to 8-bit grayscale
   - images with transparency: composited onto black
   - everything ends up as RGB
2. Resize to 224×224, convert with `ToTensor`, and normalise with the ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. This must match the transform in the training `dataset.py`.
3. Keep a 224×224 RGB copy scaled to 0–1. It is the base image for the Grad-CAM overlay and the input to the MRI check.

### Prediction
- **One forward pass** produces the logits. Softmax turns them into probabilities, and the predicted class is the argmax.
- The same logits are reused for the Grad-CAM backward pass, so there is no second forward pass.
- The whole pipeline runs under a `threading.Lock`. The Grad-CAM hook state is shared, and handling one request at a time keeps peak memory flat.

### Grad-CAM (explainability)
- A forward hook on `layer4[-1]` stores the activation `A` (512×7×7) and attaches a tensor hook that captures its gradient `∂y_c/∂A`.
- The steps that follow:
  1. Backpropagate the predicted class's logit `y_c`.
  2. Channel weights: `w_k = mean(∂y_c/∂A_k)` (global-average-pooled gradients).
  3. `CAM = ReLU(Σ_k w_k · A_k)`.
  4. Smooth it: bicubic upsampling from 7×7 to 224×224, then a Gaussian blur (σ = 6), then min-max scaling to 0–1.
  5. Colour it with OpenCV's JET colormap and blend it in proportion to attention: `overlay = img·(1−0.55·cam) + heat·0.55·cam`. Low-attention regions stay as the original scan.
- The result is returned as a base64 PNG in `gradcam_overlay`.

### Uncertainty (normalised entropy)
`H = −Σ p_i log p_i / log 4`, which ranges from 0 (fully confident) to 1 (uniform guess).

| Value | Level | Warning returned? |
|---|---|---|
| below 0.35 | `low` | no |
| 0.35 to below 0.60 | `moderate` | yes |
| 0.60 or more | `high` | yes |

### MRI plausibility check (input guardrail)
These cheap heuristics run on the 224×224 image. The input is flagged when any of them fails.

| Check | Measure | Fails when |
|---|---|---|
| Grayscale | mean per-pixel spread between R, G and B | > 0.04 |
| Dark background | fraction of pixels with intensity < 0.08 | < 0.15 |
| Contrast | standard deviation of intensity | < 0.05 |

The check exists because a CNN gives confident answers on anything. A random colour image was classified "Non Demented" with *low* entropy, so entropy alone cannot catch inputs that aren't brain scans. The MRI check catches them.

---

## 3. FastAPI layer

| Method | Route | Description |
|---|---|---|
| GET | `/api/health` | Status, architecture, class list, device |
| GET | `/api/samples` | The bundled sample cases (id, label, expected class, thumbnail URL) |
| POST | `/api/predict-sample` | Body `{"sample_id": "mild_dementia"}`; runs inference on a bundled slice (404 if unknown) |
| POST | `/api/predict` | `multipart/form-data` with a `file` field (PNG/JPEG, ≤ 10 MB) |
| GET | `/` | Serves `static/index.html`; `/static/*` is mounted via `StaticFiles` |

- **Plain `def` endpoints.** FastAPI runs them in its threadpool, so CPU-heavy inference doesn't block the event loop.
- **Upload validation.** The content type must be `image/*` (or `application/octet-stream`, which some browsers send). At most 10 MB + 1 byte is read, and anything larger is rejected with **413**. The file is fully decoded with PIL; failure returns **400**. `Image.MAX_IMAGE_PIXELS` guards against decompression bombs.
- **Request body validation:** a Pydantic model (`SamplePredictRequest`).
- **CORS** is open, so the API can also be called from a separately hosted frontend.

### Response (`/api/predict`, `/api/predict-sample`) — illustrative example

```json
{
  "prediction": "Mild Dementia",
  "confidence": 89.09,
  "uncertainty_entropy": 0.286,
  "uncertainty_level": "low",
  "uncertainty_warning": null,
  "mri_check": {
    "is_likely_mri": true, "is_grayscale": true, "color_spread_score": 0.0,
    "dark_background_fraction": 0.542, "contrast_score": 0.31, "warning": null
  },
  "gradcam_overlay": "data:image/png;base64,...",
  "probabilities": {"Mild Dementia": 89.09, "Moderate Dementia": 0.4, "Non Demented": 2.1, "Very mild Dementia": 8.41},
  "ranked_probabilities": [{"class_name": "Mild Dementia", "probability": 89.09}, "..."],
  "clinical_info": {"cdr": "CDR 1", "severity": "Mild", "description": "..."},
  "latency_ms": 51.6,
  "device": "cpu",
  "model": "ResNet-18 baseline (class-weighted CE)",
  "disclaimer": "..."
}
```

`/api/predict` also returns `filename`; `/api/predict-sample` also returns `sample_info`.

---

## 4. Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload        # http://127.0.0.1:8000  (API docs: /docs)
```

## 5. Deploy on Render (single web service)

| Setting | Value |
|---|---|
| Root Directory | blank (or the folder containing `main.py`) |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` (no `--workers`) |
| Env vars | `PYTHON_VERSION=3.11.9`, `MALLOC_ARENA_MAX=2` |
| Health check | `/api/health` |

**Avoiding out-of-memory errors:**
- `requirements.txt` adds the PyTorch **CPU-only** package index. Without it, pip installs the multi-GB CUDA build of torch.
- Single worker, single-thread torch, frozen weights and serialised requests keep memory to about 350–400 MB under load (measured on macOS). That fits the 512 MB plan.
