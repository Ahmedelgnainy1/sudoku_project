"""MNIST loader: digits 1-9 only, seeded train/validation split, untouched test set."""
import numpy as np
from torchvision.datasets import MNIST

def _prep(ds):
    x = ds.data.numpy().astype(np.float32) / 255.0   # scale to 0-1
    y = ds.targets.numpy().astype(np.int64)
    keep = y != 0                                    # drop digit 0
    return x[keep], y[keep]                          # labels stay 1..9

def load_mnist(seed=42, root="data/raw", download=True):
    x, y = _prep(MNIST(root, train=True, download=download))
    x_test, y_test = _prep(MNIST(root, train=False, download=download))
    idx = np.random.default_rng(seed).permutation(len(x))
    n_val = round(len(x) * 10000 / 60000)            # same proportion as 50k/10k
    val, trn = idx[:n_val], idx[n_val:]
    return dict(x_train=x[trn], y_train=y[trn],
                x_val=x[val], y_val=y[val],
                x_test=x_test, y_test=y_test)