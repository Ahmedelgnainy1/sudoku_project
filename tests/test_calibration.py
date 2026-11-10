import numpy as np
from nn.calibration import softmax, ece, confusion

def test_softmax_sums_to_one():
    p = softmax(np.random.default_rng(0).normal(size=(5, 9)))
    assert np.allclose(p.sum(1), 1)

def test_ece_zero_for_perfect_confident():
    p = np.eye(9)[[0, 1, 2]]
    assert ece(p, np.array([0, 1, 2])) < 1e-9

def test_ece_high_when_confident_and_wrong():
    p = np.eye(9)[[0, 0, 0]]
    assert ece(p, np.array([1, 2, 3])) > 0.9

def test_confusion_counts():
    m = confusion(np.array([0, 1, 1]), np.array([0, 1, 0]), n=2)
    assert m.tolist() == [[1, 1], [0, 1]] and m.sum() == 3