"""
utils.py
========
Small, dependency-free numpy helpers shared across the codebase.
"""
import time
import numpy as np


def set_seed(seed: int):
    np.random.seed(seed)


def softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a = a - np.max(a, axis=axis, keepdims=True)
    e = np.exp(a)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return out


def cross_entropy(probs: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    n = y.shape[0]
    p_true = probs[np.arange(n), y.astype(int)]
    return float(-np.mean(np.log(p_true + eps)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == y.astype(int)))


class Batcher:
    """Simple shuffling minibatch iterator over (X, y) numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.n = X.shape[0]

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        idx = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, self.n, self.batch_size):
            sel = idx[start:start + self.batch_size]
            yield self.X[sel], self.y[sel]


class Timer:
    def __init__(self):
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.t0


def ensure_dir(path: str):
    import os
    os.makedirs(path, exist_ok=True)
    return path
