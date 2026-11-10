import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))   # logits

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

@torch.no_grad()
def get_logits(m, x):
    m.eval()
    out = [m(torch.from_numpy(x[i:i + 1000]).unsqueeze(1))
           for i in range(0, len(x), 1000)]
    return torch.cat(out).numpy()

def _shift(xb, k=2):
    dy, dx = np.random.randint(-k, k + 1, 2)
    return torch.roll(xb, shifts=(int(dy), int(dx)), dims=(2, 3))

def train_cnn(x_tr, y_tr, x_val, y_val, seed=42, epochs=5,
              augment=True, lr=1e-3, bs=128):
    set_seed(seed)
    m = CNN()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    X = torch.from_numpy(x_tr).unsqueeze(1)
    Y = torch.from_numpy(y_tr - 1)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            xb = _shift(X[b]) if augment else X[b]
            loss = F.cross_entropy(m(xb), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        acc = (get_logits(m, x_val).argmax(1) == y_val - 1).mean()
        print(f"epoch {ep + 1} val_acc {acc:.4f}")
    return m