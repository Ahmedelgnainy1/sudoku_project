import numpy as np, torch
from nn.cnn import CNN, set_seed, get_logits
from nn.baseline import train_logreg, logreg_proba

def test_cnn_output_shape():
    assert CNN()(torch.zeros(4, 1, 28, 28)).shape == (4, 9)

def test_get_logits_shape():
    assert get_logits(CNN(), np.zeros((1500, 28, 28), np.float32)).shape == (1500, 9)

def test_same_seed_same_weights():
    set_seed(1); a = CNN()
    set_seed(1); b = CNN()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))

def test_logreg_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.random((300, 28, 28)).astype(np.float32)
    y = np.arange(300) % 9 + 1
    p = logreg_proba(train_logreg(x, y, max_iter=20), x[:10])
    assert p.shape == (10, 9) and np.allclose(p.sum(1), 1)