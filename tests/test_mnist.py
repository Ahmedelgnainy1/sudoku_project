import numpy as np
from data.mnist import load_mnist

D = load_mnist(seed=42)

# ---- normal ----
def test_shapes_and_range():
    assert D["x_train"].shape[1:] == (28, 28)
    assert D["x_train"].min() >= 0.0 and D["x_train"].max() <= 1.0

def test_labels_are_1_to_9_only():
    for k in ("train", "val", "test"):
        assert set(D[f"y_{k}"].tolist()) == set(range(1, 10))

def test_lengths_match():
    for k in ("train", "val", "test"):
        assert len(D[f"x_{k}"]) == len(D[f"y_{k}"])

def test_val_proportion():
    n = len(D["x_train"]) + len(D["x_val"])
    assert abs(len(D["x_val"]) / n - 1 / 6) < 0.001

def test_same_seed_same_split():
    d2 = load_mnist(seed=42)
    assert np.array_equal(D["y_val"], d2["y_val"])

# ---- edge ----
def test_different_seed_different_split():
    d2 = load_mnist(seed=7)
    assert not np.array_equal(D["y_val"], d2["y_val"])

def test_test_set_independent_of_seed():
    d2 = load_mnist(seed=7)
    assert np.array_equal(D["x_test"], d2["x_test"])

def test_no_overlap_train_val_by_count():
    assert len(D["x_train"]) + len(D["x_val"]) == 60000 - 5923   # 5923 zeros in train