from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    source: str

    @property
    def split_sizes(self) -> dict[str, int]:
        return {
            "train": len(self.y_train),
            "validation": len(self.y_val),
            "test": len(self.y_test),
        }


def load_real_digits(seed: int = 42) -> DatasetBundle:
    """Load and stratify the real UCI optical handwritten-digits dataset."""
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images,
        labels,
        test_size=0.15,
        random_state=seed,
        stratify=labels,
    )
    validation_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train_val,
    )

    return DatasetBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        source="UCI Optical Recognition of Handwritten Digits via scikit-learn",
    )


def flatten_images(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1)
