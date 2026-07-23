"""
data.py
=======
Loads MNIST when possible, and degrades gracefully to a small built-in
fallback dataset when it isn't (e.g. this sandbox has no internet access
and no torch/torchvision installed -- see README "Running this for real").

Loading strategy, in order:
  1. `sklearn.datasets.fetch_openml('mnist_784')`            (needs internet, no torch needed)
  2. raw IDX files (train/test images+labels) under data_dir  (works fully offline once downloaded once)
  3. `torchvision.datasets.MNIST` if torch/torchvision exist   (needs internet the first time)
  4. sklearn's bundled `load_digits` (8x8, 1797 samples)       (always available, NOT real MNIST --
     used only so the full pipeline can be smoke-tested in environments with no network and no MNIST cache)

Everything downstream (models, metrics, plotting) is written against the
same (X, y) numpy-array contract regardless of which branch fired, so
swapping in real MNIST later requires zero code changes elsewhere.
"""
import os
import gzip
import struct
import warnings
import numpy as np

from utils import ensure_dir


def _try_fetch_openml_mnist():
    from sklearn.datasets import fetch_openml
    d = fetch_openml("mnist_784", version=1, as_frame=False)
    X = d.data.astype(np.float64)
    y = d.target.astype(int)
    return X, y


def _try_idx_files(data_dir):
    paths = {
        "train_images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
        "test_images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        "test_labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        raise FileNotFoundError("MNIST idx files not found under data_dir")

    def read_images(path):
        with gzip.open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            buf = f.read(n * rows * cols)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
            return arr.astype(np.float64)

    def read_labels(path):
        with gzip.open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            buf = f.read(n)
            return np.frombuffer(buf, dtype=np.uint8).astype(int)

    Xtr = read_images(paths["train_images"])
    ytr = read_labels(paths["train_labels"])
    Xte = read_images(paths["test_images"])
    yte = read_labels(paths["test_labels"])
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)
    return X, y


def _try_torchvision(data_dir):
    import torch
    import torchvision

    train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    Xtr = train.data.numpy().reshape(len(train), -1).astype(np.float64)
    ytr = train.targets.numpy().astype(int)
    Xte = test.data.numpy().reshape(len(test), -1).astype(np.float64)
    yte = test.targets.numpy().astype(int)
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)
    return X, y


def _fallback_digits():
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data.astype(np.float64)  # 0..16, 8x8 = 64 dims
    y = d.target.astype(int)
    # rescale to 0..255 so downstream normalization behaves the same way as for real MNIST
    X = X / X.max() * 255.0
    return X, y


def load_raw(cfg) -> "tuple[np.ndarray, np.ndarray, bool]":
    """Returns (X, y, is_real_mnist)."""
    if cfg.dataset == "digits":
        warnings.warn(
            "DataConfig.dataset == 'digits': using sklearn's bundled 8x8 digits dataset "
            "as a SMOKE-TEST stand-in for MNIST. This is NOT a substitute for a real "
            "MNIST run -- see README for how to point this at real MNIST."
        )
        X, y = _fallback_digits()
        return X, y, False

    ensure_dir(cfg.data_dir)
    errors = []
    for name, fn in [
        ("fetch_openml", lambda: _try_fetch_openml_mnist()),
        ("idx_files", lambda: _try_idx_files(cfg.data_dir)),
        ("torchvision", lambda: _try_torchvision(cfg.data_dir)),
    ]:
        try:
            X, y = fn()
            print(f"[data] loaded real MNIST via {name}: X={X.shape}, y={y.shape}")
            return X, y, True
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {e}")

    if not cfg.allow_fallback:
        raise RuntimeError("Could not load MNIST and allow_fallback=False:\n" + "\n".join(errors))

    warnings.warn(
        "Could not load real MNIST through any strategy (no internet / no torch / no local "
        "cache?). Falling back to sklearn's bundled 8x8 'digits' dataset purely to smoke-test "
        "the pipeline. Failures were:\n" + "\n".join(errors)
    )
    X, y = _fallback_digits()
    return X, y, False


def normalize(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zero_one":
        return X / 255.0
    elif mode == "standardize":
        mu, sd = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-8
        return (X - mu) / sd
    raise ValueError(mode)


def load_splits(cfg, seed: int = 0):
    """
    Returns a dict with 'train', 'val', 'test' each holding (X, y), plus
    'is_real_mnist' and 'input_dim' / 'n_classes' metadata.
    """
    X, y, is_real = load_raw(cfg)
    X = normalize(X, cfg.normalize)

    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    # Use the standard MNIST 60000/10000 split when we actually have 70000 real
    # MNIST samples; otherwise split 80/20 (the digits fallback only has 1797).
    if is_real and n >= 69000:
        n_test = 10000
    else:
        n_test = int(0.2 * n)

    X_test, y_test = X[:n_test], y[:n_test]
    X_rest, y_rest = X[n_test:], y[n_test:]

    n_val = int(cfg.val_fraction * X_rest.shape[0])
    X_val, y_val = X_rest[:n_val], y_rest[:n_val]
    X_train, y_train = X_rest[n_val:], y_rest[n_val:]

    n_classes = int(np.max(y) + 1)
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "is_real_mnist": is_real,
        "input_dim": X.shape[1],
        "n_classes": n_classes,
    }
