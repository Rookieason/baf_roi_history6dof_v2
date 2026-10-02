from __future__ import annotations

from pathlib import Path

import numpy as np


class StreamingStandardizer:
    def __init__(self) -> None:
        self.n = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def partial_fit(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if self.mean is None:
            self.mean = np.zeros(x.shape[1], dtype=np.float64)
            self.m2 = np.zeros(x.shape[1], dtype=np.float64)
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (row - self.mean)

    @property
    def scale(self) -> np.ndarray:
        if self.mean is None or self.m2 is None or self.n < 2:
            return np.ones(1, dtype=np.float64)
        return np.sqrt(np.maximum(self.m2 / (self.n - 1), 1e-8))

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return x.astype(np.float32)
        return ((x - self.mean) / self.scale).astype(np.float32)


class BinarySigmoidClassifier:
    def __init__(self, input_dim: int, output_dim: int, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.01, size=(input_dim, output_dim)).astype(np.float32)
        self.bias = np.zeros(output_dim, dtype=np.float32)

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x)
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(np.float32)

    def predict_mask(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= float(threshold)).astype(np.float32)

    def train_batch(self, x: np.ndarray, y: np.ndarray, lr: float, l2: float) -> float:
        logits = self.predict_logits(x)
        y = np.asarray(y, dtype=np.float32)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        n = x.shape[0]
        loss = np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
        error = probs - y
        grad_w = x.T @ error / n + l2 * self.weights
        grad_b = error.mean(axis=0)
        self.weights -= lr * grad_w.astype(np.float32)
        self.bias -= lr * grad_b.astype(np.float32)
        return float(loss.mean())

    def save(self, path: str | Path, standardizer: StreamingStandardizer) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=self.bias,
            mean=standardizer.mean,
            scale=standardizer.scale,
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["BinarySigmoidClassifier", StreamingStandardizer]:
        data = np.load(path)
        model = cls(int(data["weights"].shape[0]), int(data["weights"].shape[1]))
        model.weights = data["weights"].astype(np.float32)
        model.bias = data["bias"].astype(np.float32)
        standardizer = StreamingStandardizer()
        standardizer.mean = data["mean"].astype(np.float64)
        standardizer.m2 = (data["scale"].astype(np.float64) ** 2)
        standardizer.n = 2
        return model, standardizer
