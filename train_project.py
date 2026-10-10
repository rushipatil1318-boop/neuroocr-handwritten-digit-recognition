from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cnn_numpy import TinyCNN
from src.data import flatten_images, load_real_digits
from src.evaluation import (
    classification_metrics,
    save_class_distribution,
    save_confusion_matrix,
    save_dataset_examples,
    save_model_comparison,
    save_saliency_examples,
    save_sample_predictions,
    save_training_curves,
)


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SAMPLES = ROOT / "samples"
SEED = 42
CONFIDENCE_THRESHOLD = 0.85


def train_and_evaluate() -> dict[str, object]:
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    SAMPLES.mkdir(exist_ok=True)

    dataset = load_real_digits(seed=SEED)
    all_images = np.concatenate([dataset.X_train, dataset.X_val, dataset.X_test])
    all_labels = np.concatenate([dataset.y_train, dataset.y_val, dataset.y_test])
    save_dataset_examples(all_images, all_labels, RESULTS / "dataset_examples.png")
    save_class_distribution(all_labels, RESULTS / "class_distribution.png")

    ml_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    max_iter=2500,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )
    ml_start = time.perf_counter()
    ml_model.fit(flatten_images(dataset.X_train), dataset.y_train)
    ml_seconds = time.perf_counter() - ml_start
    ml_probabilities = ml_model.predict_proba(flatten_images(dataset.X_test))
    ml_predictions = ml_probabilities.argmax(axis=1)
    ml_metrics = classification_metrics(dataset.y_test, ml_predictions, ml_probabilities)
    joblib.dump(ml_model, MODELS / "logistic_regression.joblib")

    cnn_model = TinyCNN(conv_filters=16, hidden_dim=64, seed=SEED)
    cnn_start = time.perf_counter()
    history = cnn_model.fit(
        dataset.X_train,
        dataset.y_train,
        dataset.X_val,
        dataset.y_val,
        epochs=70,
        batch_size=64,
        learning_rate=0.0025,
        l2=1e-4,
        patience=12,
        verbose=True,
    )
    cnn_seconds = time.perf_counter() - cnn_start
    cnn_probabilities = cnn_model.predict_proba(dataset.X_test)
    cnn_predictions = cnn_probabilities.argmax(axis=1)
    cnn_metrics = classification_metrics(dataset.y_test, cnn_predictions, cnn_probabilities)
    cnn_model.save(MODELS / "neuroocr_cnn.npz")

    confidence = cnn_probabilities.max(axis=1)
    accepted = confidence >= CONFIDENCE_THRESHOLD
    selective_accuracy = (
        float((cnn_predictions[accepted] == dataset.y_test[accepted]).mean())
        if accepted.any()
        else 0.0
    )
    uncertainty = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "coverage": float(accepted.mean()),
        "accepted_accuracy": selective_accuracy,
        "rejected_images": int((~accepted).sum()),
    }

    comparison = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "family": "Classical ML",
                **ml_metrics,
                "training_seconds": ml_seconds,
            },
            {
                "model": "NumPy CNN",
                "family": "Deep Learning",
                **cnn_metrics,
                "training_seconds": cnn_seconds,
            },
        ]
    )
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    prediction_table = pd.DataFrame(
        {
            "true_label": dataset.y_test,
            "ml_prediction": ml_predictions,
            "cnn_prediction": cnn_predictions,
            "cnn_confidence": confidence,
            "accepted_at_0_85": accepted,
        }
    )
    prediction_table.to_csv(RESULTS / "test_predictions.csv", index=False)

    metrics_payload = {
        "dataset": {
            "name": "Optical Recognition of Handwritten Digits",
            "source": dataset.source,
            "samples": int(len(all_labels)),
            "image_shape": [8, 8],
            "classes": 10,
            "split_sizes": dataset.split_sizes,
            "seed": SEED,
        },
        "logistic_regression": {**ml_metrics, "training_seconds": ml_seconds},
        "numpy_cnn": {
            **cnn_metrics,
            "training_seconds": cnn_seconds,
            "epochs_completed": len(history["train_loss"]),
        },
        "uncertainty_policy": uncertainty,
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    save_confusion_matrix(
        dataset.y_test,
        ml_predictions,
        "Logistic Regression confusion matrix",
        RESULTS / "confusion_matrix_ml.png",
    )
    save_confusion_matrix(
        dataset.y_test,
        cnn_predictions,
        "NumPy CNN confusion matrix",
        RESULTS / "confusion_matrix_cnn.png",
    )
    save_training_curves(history, RESULTS / "training_curves.png")
    save_model_comparison(comparison, RESULTS / "model_comparison.png")
    save_sample_predictions(
        dataset.X_test,
        dataset.y_test,
        cnn_predictions,
        cnn_probabilities,
        RESULTS / "sample_predictions.png",
    )
    save_saliency_examples(
        cnn_model,
        dataset.X_test,
        dataset.y_test,
        RESULTS / "saliency_examples.png",
    )

    for digit in range(10):
        index = int(np.flatnonzero(dataset.y_test == digit)[0])
        image = (dataset.X_test[index] * 255).astype(np.uint8)
        Image.fromarray(image).resize((256, 256), Image.Resampling.NEAREST).save(
            SAMPLES / f"digit_{digit}.png"
        )

    run_info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "status": "completed",
    }
    (RESULTS / "run_environment.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    print("\nFinal test results")
    print(comparison.to_string(index=False))
    print("\nConfidence policy")
    print(json.dumps(uncertainty, indent=2))
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
