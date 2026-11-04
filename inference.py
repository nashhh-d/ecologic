"""Inference utilities for ECOLOGIC's trained MobileNetV3-Small checkpoint."""
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torch import nn


@lru_cache(maxsize=2)
def _load_model_cached(checkpoint_path: str, device_name: str):
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_file}. Train the model first with train.py."
        )

    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    classes = checkpoint.get("classes")
    if not classes:
        raise ValueError("Checkpoint does not contain a 'classes' list.")

    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mean = checkpoint.get("normalization_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalization_std", [0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, classes, preprocess, device


def load_model(checkpoint_path="outputs/ecologic_mobilenetv3.pth", device=None):
    """Load and return (model, class_names, preprocess, device)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return _load_model_cached(str(Path(checkpoint_path).resolve()), str(device))


def predict_image(image, checkpoint_path="outputs/ecologic_mobilenetv3.pth",
                  low_confidence_threshold=0.60, top_k=5):
    """Classify a PIL image and return a frontend-friendly dictionary.

    Confidence is the model's softmax score, not a calibrated probability of correctness.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance")
    if not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be between 0 and 1")

    model, classes, preprocess, device = load_model(checkpoint_path)
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    count = min(max(1, int(top_k)), len(classes))
    top_values, top_indices = torch.topk(probabilities, k=count)
    predicted_index = int(top_indices[0].item())
    confidence = float(top_values[0].item())
    category = classes[predicted_index]

    ranked = [
        {"category": classes[int(idx.item())], "confidence": float(value.item())}
        for value, idx in zip(top_values, top_indices)
    ]
    return {
        "category": category,
        "label": category.replace("_", " ").title(),
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "low_confidence": confidence < low_confidence_threshold,
        "threshold": low_confidence_threshold,
        "top_predictions": ranked,
        "probabilities": {item["category"]: item["confidence"] for item in ranked},
        "model": "MobileNetV3-Small",
        "note": "Softmax confidence is not a guarantee of correctness; review uncertain predictions.",
    }
