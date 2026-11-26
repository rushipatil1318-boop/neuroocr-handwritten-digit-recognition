from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class TinyCNN:
    """A small convolutional neural network implemented only with NumPy."""

    def __init__(
        self,
        conv_filters: int = 16,
        hidden_dim: int = 64,
        n_classes: int = 10,
        seed: int = 42,
    ) -> None:
        self.conv_filters = conv_filters
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.params = {
            "Wc": (
                self.rng.standard_normal((conv_filters, 1, 3, 3))
                * np.sqrt(2.0 / 9.0)
            ).astype(np.float32),
            "bc": np.zeros(conv_filters, dtype=np.float32),
            "W1": (
                self.rng.standard_normal((conv_filters * 3 * 3, hidden_dim))
                * np.sqrt(2.0 / (conv_filters * 3 * 3))
            ).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "W2": (
                self.rng.standard_normal((hidden_dim, n_classes))
                * np.sqrt(2.0 / hidden_dim)
            ).astype(np.float32),
            "b2": np.zeros(n_classes, dtype=np.float32),
        }
        self.history_: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / exp_values.sum(axis=1, keepdims=True)

    @staticmethod
    def _pool_forward(values: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        n, channels, height, width = values.shape
        if height % 2 or width % 2:
            raise ValueError("Pooling expects even spatial dimensions.")
        blocks = values.reshape(n, channels, height // 2, 2, width // 2, 2)
        pooled = blocks.max(axis=(3, 5))
        return pooled, (blocks, pooled)

    @staticmethod
    def _pool_backward(
        gradient: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        blocks, pooled = cache
        mask = blocks == pooled[:, :, :, None, :, None]
        ties = mask.sum(axis=(3, 5), keepdims=True)
        distributed = gradient[:, :, :, None, :, None] * mask / ties
        n, channels, out_h, _, out_w, _ = blocks.shape
        return distributed.reshape(n, channels, out_h * 2, out_w * 2)

    def _forward(
        self,
        images: np.ndarray,
        keep_cache: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]] | None]:
        x = images[:, None, :, :].astype(np.float32, copy=False)
        patches = sliding_window_view(x, (3, 3), axis=(2, 3))
        conv = np.einsum("nchwkl,ockl->nohw", patches, self.params["Wc"])
        conv += self.params["bc"][None, :, None, None]
        activated = np.maximum(conv, 0.0)
        pooled, pool_cache = self._pool_forward(activated)
        flat = pooled.reshape(len(images), -1)
        hidden_pre = flat @ self.params["W1"] + self.params["b1"]
        hidden = np.maximum(hidden_pre, 0.0)
        logits = hidden @ self.params["W2"] + self.params["b2"]
        probabilities = self._softmax(logits)

        if not keep_cache:
            return probabilities, None
        cache = {
            "x": x,
            "patches": patches,
            "conv": conv,
            "activated": activated,
            "pool_cache": pool_cache,
            "flat": flat,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
            "probabilities": probabilities,
        }
        return probabilities, cache

    def _loss(self, probabilities: np.ndarray, labels: np.ndarray, l2: float) -> float:
        selected = probabilities[np.arange(len(labels)), labels]
        data_loss = -np.log(selected + 1e-12).mean()
        weight_loss = 0.5 * l2 * sum(
            float(np.sum(self.params[name] ** 2)) for name in ("Wc", "W1", "W2")
        )
        return float(data_loss + weight_loss)

    def _backward(
        self,
        labels: np.ndarray,
        cache: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]],
        l2: float,
    ) -> dict[str, np.ndarray]:
        probabilities = np.asarray(cache["probabilities"])
        hidden = np.asarray(cache["hidden"])
        hidden_pre = np.asarray(cache["hidden_pre"])
        flat = np.asarray(cache["flat"])
        conv = np.asarray(cache["conv"])
        patches = np.asarray(cache["patches"])
        pool_cache = cache["pool_cache"]

        d_logits = probabilities.copy()
        d_logits[np.arange(len(labels)), labels] -= 1.0
        d_logits /= len(labels)

        dW2 = hidden.T @ d_logits + l2 * self.params["W2"]
        db2 = d_logits.sum(axis=0)
        d_hidden = d_logits @ self.params["W2"].T
        d_hidden_pre = d_hidden * (hidden_pre > 0)

        dW1 = flat.T @ d_hidden_pre + l2 * self.params["W1"]
        db1 = d_hidden_pre.sum(axis=0)
        d_flat = d_hidden_pre @ self.params["W1"].T
        d_pooled = d_flat.reshape(len(labels), self.conv_filters, 3, 3)
        d_activated = self._pool_backward(d_pooled, pool_cache)
        d_conv = d_activated * (conv > 0)

        dWc = np.einsum("nchwkl,nohw->ockl", patches, d_conv)
        dWc += l2 * self.params["Wc"]
        dbc = d_conv.sum(axis=(0, 2, 3))

        return {
            "Wc": dWc.astype(np.float32),
            "bc": dbc.astype(np.float32),
            "W1": dW1.astype(np.float32),
            "b1": db1.astype(np.float32),
            "W2": dW2.astype(np.float32),
            "b2": db2.astype(np.float32),
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 60,
        batch_size: int = 64,
        learning_rate: float = 0.0025,
        l2: float = 1e-4,
        patience: int = 10,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        adam_m = {name: np.zeros_like(value) for name, value in self.params.items()}
        adam_v = {name: np.zeros_like(value) for name, value in self.params.items()}
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        step = 0
        best_loss = float("inf")
        best_params = {name: value.copy() for name, value in self.params.items()}
        stale_epochs = 0

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(len(y_train))
            for start in range(0, len(order), batch_size):
                batch_indices = order[start : start + batch_size]
                _, cache = self._forward(X_train[batch_indices], keep_cache=True)
                if cache is None:
                    raise RuntimeError("Training cache was not created.")
                gradients = self._backward(y_train[batch_indices], cache, l2)
                step += 1
                for name, gradient in gradients.items():
                    adam_m[name] = beta1 * adam_m[name] + (1.0 - beta1) * gradient
                    adam_v[name] = beta2 * adam_v[name] + (1.0 - beta2) * (gradient**2)
                    m_hat = adam_m[name] / (1.0 - beta1**step)
                    v_hat = adam_v[name] / (1.0 - beta2**step)
                    self.params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            train_probabilities = self.predict_proba(X_train)
            val_probabilities = self.predict_proba(X_val)
            train_loss = self._loss(train_probabilities, y_train, l2)
            val_loss = self._loss(val_probabilities, y_val, l2)
            train_accuracy = float((train_probabilities.argmax(axis=1) == y_train).mean())
            val_accuracy = float((val_probabilities.argmax(axis=1) == y_val).mean())

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["train_accuracy"].append(train_accuracy)
            self.history_["val_accuracy"].append(val_accuracy)

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                    f"accuracy {train_accuracy:.4f} | val_accuracy {val_accuracy:.4f}"
                )

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_params = {name: value.copy() for name, value in self.params.items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    if verbose:
                        print(f"Early stopping after epoch {epoch}; restoring best weights.")
                    break

        self.params = best_params
        return self.history_

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        probabilities, _ = self._forward(images, keep_cache=False)
        return probabilities

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.predict_proba(images).argmax(axis=1)

    def saliency(self, image: np.ndarray, target_class: int | None = None) -> np.ndarray:
        batch = np.asarray(image, dtype=np.float32).reshape(1, 8, 8)
        probabilities, cache = self._forward(batch, keep_cache=True)
        if cache is None:
            raise RuntimeError("Saliency cache was not created.")
        target = int(probabilities.argmax(axis=1)[0] if target_class is None else target_class)

        hidden_pre = np.asarray(cache["hidden_pre"])
        conv = np.asarray(cache["conv"])
        pool_cache = cache["pool_cache"]
        d_logits = np.zeros((1, self.n_classes), dtype=np.float32)
        d_logits[0, target] = 1.0
        d_hidden = d_logits @ self.params["W2"].T
        d_hidden_pre = d_hidden * (hidden_pre > 0)
        d_flat = d_hidden_pre @ self.params["W1"].T
        d_pooled = d_flat.reshape(1, self.conv_filters, 3, 3)
        d_activated = self._pool_backward(d_pooled, pool_cache)
        d_conv = d_activated * (conv > 0)

        d_input = np.zeros((1, 1, 8, 8), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                contribution = np.einsum(
                    "nohw,oc->nchw", d_conv, self.params["Wc"][:, :, row, col]
                )
                d_input[:, :, row : row + 6, col : col + 6] += contribution
        saliency = np.abs(d_input[0, 0])
        maximum = float(saliency.max())
        return saliency / maximum if maximum > 0 else saliency

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            **self.params,
            conv_filters=np.array(self.conv_filters),
            hidden_dim=np.array(self.hidden_dim),
            n_classes=np.array(self.n_classes),
            seed=np.array(self.seed),
            history_json=np.array(json.dumps(self.history_)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinyCNN":
        stored = np.load(path, allow_pickle=False)
        model = cls(
            conv_filters=int(stored["conv_filters"]),
            hidden_dim=int(stored["hidden_dim"]),
            n_classes=int(stored["n_classes"]),
            seed=int(stored["seed"]),
        )
        for name in model.params:
            model.params[name] = stored[name].astype(np.float32)
        model.history_ = json.loads(str(stored["history_json"].item()))
        return model
