"""Usage: python -m integration.demo_pipeline"""
import csv, torch
from csp.grid import parse
from data.mnist import load_mnist
from nn.cnn import CNN, get_logits
from nn.calibration import softmax
from integration.mosaic import build_pools, make_mosaic
from integration.reader import read_mosaic
from integration.pipeline import solve_reading

d = load_mnist(seed=42)
cnn = CNN(); cnn.load_state_dict(torch.load("models/cnn.pt")); cnn.eval()
predict = lambda t: softmax(get_logits(cnn, t))
pools = build_pools(d["y_test"], predict(d["x_test"]))
row = next(r for r in csv.DictReader(open("data/generated/puzzles.csv"))
           if r["n_givens"] == "35")
puz, sol = parse(row["puzzle"]), parse(row["solution"])
for m in range(5):
    mos, src, inj = make_mosaic(puz, d["x_test"], pools, m, seed=1)
    grid, probs = read_mosaic(mos, predict)
    for meth in ("ucs", "astar", "greedy"):
        r = solve_reading(grid, probs, method=meth)
        fixed = [c["cell"] for c in r.corrections]
        print(f"m={m} {meth:6s} {r.status:10s} exact={r.solution == sol} "
              f"injected={inj} fixed={fixed} expanded={r.search_stats.expanded} "
              f"goal_calls={r.goal_calls}")