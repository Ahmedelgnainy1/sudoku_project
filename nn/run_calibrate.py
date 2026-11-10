"""Usage: python -m nn.run_calibrate"""
import numpy as np, torch
from data.mnist import load_mnist
from nn.cnn import CNN, get_logits
from nn.baseline import train_logreg, logreg_proba
from nn.calibration import softmax, ece, nll, confusion

d = load_mnist(seed=42)
yt = d["y_test"] - 1
cnn = CNN(); cnn.load_state_dict(torch.load("models/cnn.pt")); cnn.eval()
p_cnn = softmax(get_logits(cnn, d["x_test"]))
p_lr = logreg_proba(train_logreg(d["x_train"], d["y_train"]), d["x_test"])
for name, p in [("logreg", p_lr), ("cnn", p_cnn)]:
    print(f"{name:7s} acc {(p.argmax(1) == yt).mean():.4f}  "
          f"ECE {ece(p, yt):.4f}  NLL {nll(p, yt):.4f}")
cm = confusion(p_cnn.argmax(1), yt)
np.savetxt("models/confusion_cnn.csv", cm, fmt="%d", delimiter=",")
print("CNN confusion matrix (rows true 1..9, cols predicted 1..9):")
print(cm)