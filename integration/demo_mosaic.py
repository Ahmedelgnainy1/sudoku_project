"""Usage: python -m integration.demo_mosaic"""
import csv, torch
from csp.grid import parse
from data.mnist import load_mnist
from nn.cnn import CNN, get_logits
from nn.calibration import softmax
from integration.mosaic import build_pools, make_mosaic
from integration.reader import read_mosaic

d = load_mnist(seed=42)
cnn = CNN(); cnn.load_state_dict(torch.load("models/cnn.pt")); cnn.eval()
pools = build_pools(d["y_test"], softmax(get_logits(cnn, d["x_test"])))
print("digit  easy  hard")
for k in range(1, 10):
    print(f"{k:5d} {len(pools['easy'][k]):5d} {len(pools['hard'][k]):5d}")
predict = lambda t: softmax(get_logits(cnn, t))
row = next(r for r in csv.DictReader(open("data/generated/puzzles.csv"))
           if r["n_givens"] == "35")
puz = parse(row["puzzle"])
for m in range(5):
    mos, src, inj = make_mosaic(puz, d["x_test"], pools, m, seed=1)
    grid, _ = read_mosaic(mos, predict)
    wrong = [i for i in range(81) if grid[i] != puz[i]]
    print(f"m={m} injected={inj} misread={wrong} match={wrong == inj}")