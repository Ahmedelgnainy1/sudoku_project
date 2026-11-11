"""Experiment 2: BT vs BT+FC vs BT+FC+AC-3 at 25/30/35 givens.
Usage: python -m eval.exp2 [--seeds 5] [--per-seed 10] [--cap 200000]"""
import argparse, csv, os, statistics as st, time
from csp.generator import make_puzzle
from csp.solver import Solver, BudgetExceeded

CONFIGS = [("BT", {}), ("BT+FC", dict(use_fc=True)),
           ("BT+FC+AC3", dict(use_fc=True, use_ac3=True))]
LEVELS = (25, 30, 35)

def run_one(puz, kw, cap):
    s = Solver(max_assignments=cap, **kw)
    t = time.perf_counter()
    try:
        s.solve(puz); timeout = False
    except BudgetExceeded:
        timeout = True
    return s.stats, timeout, (time.perf_counter() - t) * 1000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--per-seed", type=int, default=10)
    ap.add_argument("--cap", type=int, default=200000)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    rows = []
    for seed in range(1, a.seeds + 1):
        for n in LEVELS:
            for k in range(a.per_seed):
                puz, _ = make_puzzle(seed * 1000 + k, n)
                for name, kw in CONFIGS:
                    s, to, ms = run_one(puz, kw, a.cap)
                    rows.append(dict(seed=seed, givens=n, k=k, config=name,
                                     assignments=s.assignments, backtracks=s.backtracks,
                                     revisions=s.revisions, ms=round(ms, 2), timeout=int(to)))
        print("seed", seed, "done", flush=True)
    with open("results/exp2_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary = []
    print(f"\n{'givens':>6} {'config':10s} {'assignments':>24} {'backtracks':>24} "
          f"{'revisions':>10} {'ms':>20} timeouts")
    for n in LEVELS:
        for name, _ in CONFIGS:
            per = {}
            for r in rows:
                if r["givens"] == n and r["config"] == name:
                    per.setdefault(r["seed"], []).append(r)
            def ms_(key):
                v = [st.mean(x[key] for x in g) for g in per.values()]
                return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            A, B, R, T = ms_("assignments"), ms_("backtracks"), ms_("revisions"), ms_("ms")
            to = sum(x["timeout"] for g in per.values() for x in g)
            tot = sum(len(g) for g in per.values())
            print(f"{n:6d} {name:10s} {A[0]:12.1f} ± {A[1]:9.1f} {B[0]:12.1f} ± {B[1]:9.1f} "
                  f"{R[0]:10.1f} {T[0]:9.1f} ± {T[1]:7.1f} {to}/{tot}")
            summary.append((n, name, A, T))
    with open("results/exp2_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["givens", "config", "assign_mean", "assign_std", "ms_mean", "ms_std"])
        for n, name, A, T in summary:
            w.writerow([n, name, round(A[0], 1), round(A[1], 1), round(T[0], 2), round(T[1], 2)])
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, idx, label in ((axs[0], 2, "assignments (mean ± std over seeds)"),
                               (axs[1], 3, "time ms (mean ± std over seeds)")):
            for j, (name, _) in enumerate(CONFIGS):
                pts = [s for s in summary if s[1] == name]
                ax.bar([i + 0.27 * j for i in range(len(LEVELS))], [p[idx][0] for p in pts],
                       0.27, yerr=[p[idx][1] for p in pts], label=name, capsize=3)
            ax.set_xticks([i + 0.27 for i in range(len(LEVELS))]); ax.set_xticklabels(LEVELS)
            ax.set_xlabel("givens"); ax.set_ylabel(label); ax.set_yscale("log")
        axs[0].legend()
        fig.tight_layout(); fig.savefig("results/exp2.png", dpi=150)
        print("wrote results/exp2.png")
    except ImportError:
        print("matplotlib not installed: no plot")

if __name__ == "__main__":
    main()