import numpy as np

def softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def ece(probs, labels0, bins=10):
    """labels0 are class indices 0..8."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels0).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total

def nll(probs, labels0):
    return float(-np.log(probs[np.arange(len(labels0)), labels0] + 1e-12).mean())

def confusion(pred0, labels0, n=9):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(labels0, pred0):
        m[t, p] += 1
    return m