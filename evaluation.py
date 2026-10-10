from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, average="macro")),
        "macro_recall": float(recall_score(labels, predictions, average="macro")),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(10))),
    }


def save_dataset_examples(images: np.ndarray, labels: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for digit, axis in enumerate(axes.flat):
        index = int(np.flatnonzero(labels == digit)[0])
        axis.imshow(images[index], cmap="gray_r", vmin=0, vmax=1)
        axis.set_title(f"Label: {digit}")
        axis.axis("off")
    fig.suptitle("Real handwritten samples from the UCI digits dataset", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_class_distribution(labels: np.ndarray, path: Path) -> None:
    counts = np.bincount(labels, minlength=10)
    fig, axis = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x=np.arange(10), y=counts, ax=axis, color="#2d6cdf")
    axis.set(title="Class distribution", xlabel="Digit", ylabel="Number of images")
    axis.bar_label(axis.containers[0])
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    title: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(10))
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set(title=title, xlabel="Predicted digit", ylabel="True digit")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history: dict[str, list[float]], path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="CNN loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="CNN accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0.6, 1.01))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison(table: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    chart = table.set_index("model")[["accuracy", "macro_f1"]].plot(
        kind="bar", ylim=(0.8, 1.0), rot=0, color=["#2d6cdf", "#16a085"], ax=axis
    )
    chart.set(title="Classical ML vs neural network", xlabel="", ylabel="Score")
    chart.legend(["Accuracy", "Macro F1"], loc="lower right")
    for container in chart.containers:
        chart.bar_label(container, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sample_predictions(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    confidence = probabilities.max(axis=1)
    chosen = np.argsort(confidence)[:4].tolist() + np.argsort(confidence)[-4:].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
    for index, axis in zip(chosen, axes.flat):
        is_correct = labels[index] == predictions[index]
        axis.imshow(images[index], cmap="gray_r", vmin=0, vmax=1)
        axis.set_title(
            f"True {labels[index]} | Pred {predictions[index]}\n"
            f"Confidence {confidence[index]:.1%}",
            color="#137333" if is_correct else "#c5221f",
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle("CNN predictions: difficult and confident examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_saliency_examples(model, images: np.ndarray, labels: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(9, 6.7))
    for column, index in enumerate(range(4)):
        prediction = int(model.predict(images[index : index + 1])[0])
        saliency = model.saliency(images[index], prediction)
        axes[0, column].imshow(images[index], cmap="gray_r", vmin=0, vmax=1)
        axes[0, column].set_title(f"True {labels[index]} | Pred {prediction}")
        axes[1, column].imshow(saliency, cmap="magma", vmin=0, vmax=1)
        axes[1, column].set_title("Saliency")
        axes[2, column].imshow(images[index], cmap="gray_r", vmin=0, vmax=1)
        axes[2, column].imshow(saliency, cmap="magma", alpha=0.55, vmin=0, vmax=1)
        axes[2, column].set_title("Overlay")
        for row in range(3):
            axes[row, column].axis("off")
    fig.suptitle("Which pixels influenced the CNN decision?", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
