"""Demo. Usage: python run_demo.py [--seed 1] [--misreads 2] [--method astar]"""
import argparse, os, random, time, torch
from csp.generator import make_puzzle
from csp.solver import Solver, BudgetExceeded
from data.mnist import load_mnist
from nn.cnn import CNN, get_logits
from nn.calibration import softmax
from integration.mosaic import build_pools, make_mosaic
from integration.reader import read_mosaic
from integration.pipeline import solve_reading
from eval.exp3 import solve_or_none, run_random

def show(grid, mark=()):
    for r in range(9):
        print(" ".join(f"{grid[9*r+c] or '.'}{'*' if 9*r+c in mark else ' '}"
                       for c in range(9)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--method", default="astar", choices=["ucs", "greedy", "astar"])
    a = ap.parse_args()
    d = load_mnist(seed=42)
    cnn = CNN(); cnn.load_state_dict(torch.load("models/cnn.pt")); cnn.eval()
    predict = lambda t: softmax(get_logits(cnn, t))
    pools = build_pools(d["y_test"], predict(d["x_test"]))
    puz, truth = make_puzzle(a.seed * 1000, 35)
    mos, _, inj = make_mosaic(puz, d["x_test"], pools, a.misreads, seed=a.seed * 1000 + a.misreads)
    grid, probs = read_mosaic(mos, predict)
    wrong = [i for i in range(81) if grid[i] != puz[i]]

    print("=== 1. INPUT: grid read by the CNN (* = misread) ===")
    show(grid, wrong)
    print("injected cells:", inj, " misread cells:", wrong)

    print(f"\n=== 2. SEARCH ({a.method}) ===")
    trace = []
    r = solve_reading(grid, probs, method=a.method, max_goal_calls=5000, trace=trace)
    print("expanded states (cost, CSP accepted):")
    for k, (state, g, ok) in enumerate(trace[:15]):
        print(f"  {k:3d} {sorted(state)} cost={g:.3f} ok={ok}")
    if len(trace) > 15:
        print(f"  ... {len(trace) - 15} more")
    if r.search_stats:
        print("expanded", r.search_stats.expanded, "generated", r.search_stats.generated)
    print("CSP checks:", r.goal_calls, "->", r.message)

    if r.status == "solved":
        print("\ncorrection log (cell, read, new, cost):")
        for c in r.corrections:
            print(f"  cell {c['cell']}: {c['read']} -> {c['new']}  cost {c['cost']:.3f}")
        print("\n=== 3. SOLUTION ===")
        show(r.solution)
        print("exact match with true solution:", r.solution == truth)
    else:
        print("least confident cells (cell, top-1 prob):", r.least_confident)

    print("\n=== 4. CSP pruning on the true puzzle ===")
    print(f"{'config':10s} {'assign':>8} {'backtr':>8} {'revis':>8} {'ms':>8}")
    counts = []
    for name, kw in [("BT", {}), ("BT+FC", dict(use_fc=True)),
                     ("BT+FC+AC3", dict(use_fc=True, use_ac3=True))]:
        s = Solver(max_assignments=200000, **kw)
        t = time.perf_counter()
        try:
            s.solve(puz)
        except BudgetExceeded:
            pass
        ms = (time.perf_counter() - t) * 1000
        counts.append((name, s.stats.assignments))
        print(f"{name:10s} {s.stats.assignments:8d} {s.stats.backtracks:8d} "
              f"{s.stats.revisions:8d} {ms:8.1f}")

    print("\n=== 5. NN ON vs OFF ===")
    sol_none = solve_or_none(grid)
    sol_rand, calls_rand = run_random(grid, 5000, random.Random(a.seed))
    print(f"NN-guided : solved={r.status == 'solved'} exact={r.solution == truth} checks={r.goal_calls}")
    print(f"no correct: solved={sol_none is not None} exact={sol_none == truth} checks=1")
    print(f"random    : solved={sol_rand is not None} exact={sol_rand == truth} checks={calls_rand}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    os.makedirs("results", exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].imshow(mos, cmap="gray"); axs[0].set_title("mosaic (red = misread)")
    for i in wrong:
        rr, cc = divmod(i, 9)
        axs[0].add_patch(Rectangle((cc * 28 - .5, rr * 28 - .5), 28, 28, fill=False, ec="red", lw=2))
    axs[0].axis("off")
    if trace:
        axs[1].plot([t[1] for t in trace], marker=".", color="gray")
        bad = [k for k, t in enumerate(trace) if not t[2]]
        good = [k for k, t in enumerate(trace) if t[2]]
        axs[1].scatter(bad, [trace[k][1] for k in bad], c="red", label="CSP rejected", zorder=3)
        axs[1].scatter(good, [trace[k][1] for k in good], c="green", label="CSP accepted", zorder=3)
        axs[1].legend()
    axs[1].set_xlabel("expansion order"); axs[1].set_ylabel("path cost g")
    axs[1].set_title(f"search trace ({a.method})")
    axs[2].bar([c[0] for c in counts], [max(c[1], 1) for c in counts])
    axs[2].set_yscale("log"); axs[2].set_ylabel("assignments"); axs[2].set_title("CSP pruning")
    fig.tight_layout(); fig.savefig("results/demo.png", dpi=150)
    print("\nwrote results/demo.png")

if __name__ == "__main__":
    main()