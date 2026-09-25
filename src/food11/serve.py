"""Serve the registered Food-11 model with FastAPI."""

from __future__ import annotations

import io
import os

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision.models import ResNet18_Weights


# Same class order used by torchvision.datasets.ImageFolder during training.
CLASS_NAMES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]


# Read the MLflow server address from an environment variable.
# Locally this defaults to 127.0.0.1:5000.
# Inside Docker we will override it with host.docker.internal:5000.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Load the registered model once when the API starts.
model = mlflow.pyfunc.load_model("models:/food11@champion")


# Use exactly the same preprocessing as training.
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()


app = FastAPI(
    title="Food-11 Classification API",
    description="Classifies food images using the champion Food-11 model.",
    version="1.0",
)


@app.get("/health")
def health():
    """Check whether the API is running."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the Food-11 category of an uploaded image."""

    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Same transform used in train.py.
        image_tensor = transform(image)

        # Model expects a batch: [N, C, H, W].
        batch = image_tensor.unsqueeze(0).numpy()

        # MLflow pyfunc returns the model's raw logits.
        predictions = model.predict(batch)

        logits = torch.tensor(np.asarray(predictions))

        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_index = torch.max(probabilities, dim=1)

        class_index = predicted_index.item()

        return {
            "category": CLASS_NAMES[class_index],
            "confidence": float(confidence.item()),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {exc}",
        ) from exc