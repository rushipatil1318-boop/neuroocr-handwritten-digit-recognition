from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .cnn_numpy import TinyCNN


def preprocess_uploaded_image(image: Image.Image) -> np.ndarray:
    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale, dtype=np.uint8)
    if float(pixels.mean()) > 127:
        pixels = 255 - pixels

    foreground = pixels > max(20, int(pixels.max() * 0.15))
    if foreground.any():
        rows, columns = np.where(foreground)
        pixels = pixels[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]

    side = max(pixels.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    top = (side - pixels.shape[0]) // 2
    left = (side - pixels.shape[1]) // 2
    square[top : top + pixels.shape[0], left : left + pixels.shape[1]] = pixels
    resized = Image.fromarray(square).resize((6, 6), Image.Resampling.LANCZOS)
    centered = np.zeros((8, 8), dtype=np.float32)
    centered[1:7, 1:7] = np.asarray(resized, dtype=np.float32) / 255.0
    return centered


def predict_image(
    model: TinyCNN,
    image: Image.Image,
    confidence_threshold: float = 0.85,
) -> dict[str, float | int | bool | np.ndarray]:
    processed = preprocess_uploaded_image(image)
    probabilities = model.predict_proba(processed[None, :, :])[0]
    prediction = int(probabilities.argmax())
    confidence = float(probabilities[prediction])
    return {
        "prediction": prediction,
        "confidence": confidence,
        "accepted": confidence >= confidence_threshold,
        "probabilities": probabilities,
        "processed_image": processed,
    }


def load_model(path: str | Path) -> TinyCNN:
    return TinyCNN.load(path)
