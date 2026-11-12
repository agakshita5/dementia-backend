import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Oasis Dementia Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define class names matching training order
CLASS_NAMES = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']

# Setup Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

# Load best_model.pt
model_path = "best_model.pt"
state_dict = torch.load(model_path, map_location=device)
if "model_state_dict" in state_dict:
    model.load_state_dict(state_dict["model_state_dict"])
else:
    model.load_state_dict(state_dict)

model.to(device)
model.eval()

# Image Preprocessing Pipeline (matching training validation transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"status": "Oasis Dementia Backend is running successfully!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Preprocess image
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)
        
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = conf.item()
    
    return {
        "prediction": predicted_class,
        "confidence": round(confidence_score * 100, 2),
        "probabilities": {CLASS_NAMES[i]: round(probabilities[0][i].item() * 100, 2) for i in range(4)}
    }
