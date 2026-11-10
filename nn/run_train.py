"""Train logistic baseline and CNN. Usage: python -m nn.run_train"""
import os, torch
from data.mnist import load_mnist
from nn.baseline import train_logreg, logreg_proba
from nn.cnn import train_cnn, get_logits

d = load_mnist(seed=42)
lr = train_logreg(d["x_train"], d["y_train"])
for k in ("val", "test"):
    acc = (logreg_proba(lr, d[f"x_{k}"]).argmax(1) == d[f"y_{k}"] - 1).mean()
    print(f"logreg {k} acc {acc:.4f}")
cnn = train_cnn(d["x_train"], d["y_train"], d["x_val"], d["y_val"])
for k in ("val", "test"):
    acc = (get_logits(cnn, d[f"x_{k}"]).argmax(1) == d[f"y_{k}"] - 1).mean()
    print(f"cnn {k} acc {acc:.4f}")
os.makedirs("models", exist_ok=True)
torch.save(cnn.state_dict(), "models/cnn.pt")
print("saved models/cnn.pt")