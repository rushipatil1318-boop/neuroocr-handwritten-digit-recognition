from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.inference import load_model, predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from an image.")
    parser.add_argument("image", type=Path, help="Path to a PNG or JPEG containing one digit")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    model = load_model(Path(__file__).parent / "models" / "neuroocr_cnn.npz")
    result = predict_image(model, Image.open(args.image), args.threshold)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Decision accepted: {result['accepted']}")


if __name__ == "__main__":
    main()
