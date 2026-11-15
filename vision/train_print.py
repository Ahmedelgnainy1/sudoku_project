"""Usage: python -m vision.train_print"""
import numpy as np, torch
from data.mnist import load_mnist
from nn.cnn import train_cnn, get_logits
from vision.printed import make_printed, available_fonts

fonts = available_fonts()
print("fonts used:", [f.split("\\")[-1] for f in fonts])
d = load_mnist(seed=42)
xp, yp = make_printed(600, 1, fonts)      # printed train: 5,400 tiles
xv, yv = make_printed(100, 2, fonts)      # printed validation (same fonts, so optimistic)
x_tr = np.concatenate([d["x_train"], xp]); y_tr = np.concatenate([d["y_train"], yp])
x_va = np.concatenate([d["x_val"], xv]);   y_va = np.concatenate([d["y_val"], yv])
cnn = train_cnn(x_tr, y_tr, x_va, y_va, seed=42, epochs=5)
for name, x, y in [("mnist test", d["x_test"], d["y_test"]), ("printed val", xv, yv)]:
    print(name, "acc", round(float((get_logits(cnn, x).argmax(1) == y - 1).mean()), 4))
torch.save(cnn.state_dict(), "models/cnn_print.pt")
print("saved models/cnn_print.pt")