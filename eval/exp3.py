"""Experiment 3: NN-guided (A*) vs no correction vs random correction.
Usage: python -m eval.exp3 [--seeds 5] [--per-seed 20] [--cap 5000]"""
import argparse, csv, os, random, statistics as st, time, torch
from csp.generator import make_puzzle
from csp.grid import conflicts
from csp.solver import Solver
from data.mnist import load_mnist
from nn.cnn import CNN, get_logits
from nn.calibration import softmax
from integration.mosaic import build_pools, make_mosaic
from integration.reader import read_mosaic
from integration.pipeline import solve_reading, SOLVER_KW

METHODS = ("nn_guided", "none", "random")
LEVELS = (0, 1, 2, 3, 4)

def solve_or_none(g):
    if conflicts(g):
        return None
    return Solver(**SOLVER_KW).solve(g)

def run_none(grid):
    sol = solve_or_none(grid)
    return sol, 1

def run_random(grid, cap, rng):
    cells = [i for i in range(81) if grid[i]]
    sol = solve_or_none(grid)
    calls, seen = 1, {frozenset()}
    if sol is not None:
        return sol, calls
    while calls < cap:
        k = rng.randint(1, 3)
        state = frozenset((c, rng.choice([d for d in range(1, 10) if d != grid[c]]))
                          for c in rng.sample(cells, k))
        if state in seen:
            continue
        seen.add(state)
        calls += 1
        g = list(grid)
        for c, d in state:
            g[c] = d
        sol = solve_or_none(g)
        if sol is not None:
            return sol, calls
    return None, calls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--per-seed", type=int, default=20)
    ap.add_argument("--cap", type=int, default=5000)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    d = load_mnist(seed=42)
    cnn = CNN(); cnn.load_state_dict(torch.load("models/cnn.pt")); cnn.eval()
    predict = lambda t: softmax(get_logits(cnn, t))
    pools = build_pools(d["y_test"], predict(d["x_test"]))
    rows = []
    for seed in range(1, a.seeds + 1):
        for k in range(a.per_seed):
            puz, truth = make_puzzle(seed * 1000 + k, 35)
            for m in LEVELS:
                mos, _, _ = make_mosaic(puz, d["x_test"], pools, m, seed=seed * 1000 + k * 10 + m)
                grid, probs = read_mosaic(mos, predict)
                for meth in METHODS:
                    t = time.perf_counter()
                    if meth == "nn_guided":
                        r = solve_reading(grid, probs, method="astar", max_goal_calls=a.cap)
                        sol, calls = r.solution, r.goal_calls
                    elif meth == "none":
                        sol, calls = run_none(grid)
                    else:
                        sol, calls = run_random(grid, a.cap, random.Random(seed * 100000 + k * 10 + m))
                    ms = (time.perf_counter() - t) * 1000
                    rows.append(dict(seed=seed, k=k, m=m, method=meth, goal_calls=calls,
                                     ms=round(ms, 2), solved=int(sol is not None),
                                     exact=int(sol == truth)))
        print("seed", seed, "done", flush=True)
    with open("results/exp3_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def agg(m, meth, key):
        per = {}
        for r in rows:
            if r["m"] == m and r["method"] == meth:
                per.setdefault(r["seed"], []).append(r[key])
        v = [st.mean(x) for x in per.values()]
        return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

    summary = []
    print(f"\n{'m':>2} {'method':10s} {'solved':>14} {'exact':>14} {'goal_calls':>18} {'ms':>18}")
    for m in LEVELS:
        for meth in METHODS:
            S, X, G, T = (agg(m, meth, k) for k in ("solved", "exact", "goal_calls", "ms"))
            print(f"{m:2d} {meth:10s} {S[0]:6.2f} +/- {S[1]:4.2f} {X[0]:6.2f} +/- {X[1]:4.2f} "
                  f"{G[0]:9.1f} +/- {G[1]:6.1f} {T[0]:9.1f} +/- {T[1]:6.1f}")
            summary.append((m, meth, S, X, G, T))
    with open("results/exp3_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "method", "solved_mean", "solved_std", "exact_mean", "exact_std",
                    "calls_mean", "calls_std", "ms_mean", "ms_std"])
        for m, meth, S, X, G, T in summary:
            w.writerow([m, meth, *(round(v, 3) for p in (S, X, G, T) for v in p)])
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    for meth in METHODS:
        pts = [s for s in summary if s[1] == meth]
        ax.errorbar(LEVELS, [p[3][0] for p in pts], [p[3][1] for p in pts],
                    label=meth, marker="o", capsize=3)
    ax.set_xlabel("injected misreads"); ax.set_ylabel("exact recovery rate")
    ax.set_xticks(LEVELS); ax.legend(); fig.tight_layout()
    fig.savefig("results/exp3.png", dpi=150)
    print("wrote results/exp3.png")

if __name__ == "__main__":
    main()