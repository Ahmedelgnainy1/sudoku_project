"""Experiment 1: UCS vs Greedy vs A*, 0-4 injected misreads, 35-given puzzles.
Usage: python -m eval.exp1 [--seeds 5] [--per-seed 20] [--cap 5000]"""
import argparse, csv, os, statistics as st, time, torch
from csp.generator import make_puzzle
from data.mnist import load_mnist
from nn.cnn import CNN, get_logits
from nn.calibration import softmax
from integration.mosaic import build_pools, make_mosaic
from integration.reader import read_mosaic
from integration.pipeline import solve_reading

METHODS = ("ucs", "greedy", "astar")
LEVELS = (0, 1, 2, 3, 4)

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
            puz, sol = make_puzzle(seed * 1000 + k, 35)
            for m in LEVELS:
                mos, _, inj = make_mosaic(puz, d["x_test"], pools, m, seed=seed * 1000 + k * 10 + m)
                grid, probs = read_mosaic(mos, predict)
                for meth in METHODS:
                    t = time.perf_counter()
                    r = solve_reading(grid, probs, method=meth, max_goal_calls=a.cap)
                    ms = (time.perf_counter() - t) * 1000
                    rows.append(dict(seed=seed, k=k, m=m, method=meth,
                                     expanded=r.search_stats.expanded if r.search_stats else r.goal_calls,
                                     goal_calls=r.goal_calls, ms=round(ms, 2),
                                     solved=int(r.status == "solved"),
                                     exact=int(r.solution == sol), capped=int(r.capped),
                                     cost=r.cost if r.cost is not None else ""))
        print("seed", seed, "done", flush=True)
    with open("results/exp1_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def agg(m, meth, key):
        per = {}
        for r in rows:
            if r["m"] == m and r["method"] == meth:
                per.setdefault(r["seed"], []).append(r[key])
        v = [st.mean(x) for x in per.values()]
        return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

    summary = []
    print(f"\n{'m':>2} {'method':7s} {'expanded':>18} {'ms':>16} {'solved':>14} {'exact':>14} capped")
    for m in LEVELS:
        for meth in METHODS:
            E, T, S, X = (agg(m, meth, k) for k in ("expanded", "ms", "solved", "exact"))
            cap = sum(r["capped"] for r in rows if r["m"] == m and r["method"] == meth)
            print(f"{m:2d} {meth:7s} {E[0]:9.1f} +/- {E[1]:6.1f} {T[0]:8.1f} +/- {T[1]:5.1f} "
                  f"{S[0]:6.2f} +/- {S[1]:4.2f} {X[0]:6.2f} +/- {X[1]:4.2f} {cap}")
            summary.append((m, meth, E, T, S, X))
    with open("results/exp1_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "method", "expanded_mean", "expanded_std", "ms_mean", "ms_std",
                    "solved_mean", "solved_std", "exact_mean", "exact_std"])
        for m, meth, E, T, S, X in summary:
            w.writerow([m, meth, *(round(v, 3) for p in (E, T, S, X) for v in p)])
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for meth in METHODS:
        pts = [s for s in summary if s[1] == meth]
        axs[0].errorbar(LEVELS, [p[2][0] for p in pts], [p[2][1] for p in pts], label=meth, marker="o", capsize=3)
        axs[1].errorbar(LEVELS, [p[5][0] for p in pts], [p[5][1] for p in pts], label=meth, marker="o", capsize=3)
    axs[0].set_yscale("log"); axs[0].set_ylabel("nodes expanded"); axs[1].set_ylabel("exact recovery rate")
    for ax in axs: ax.set_xlabel("injected misreads"); ax.set_xticks(LEVELS)
    axs[0].legend(); fig.tight_layout(); fig.savefig("results/exp1.png", dpi=150)
    print("wrote results/exp1.png")

if __name__ == "__main__":
    main()