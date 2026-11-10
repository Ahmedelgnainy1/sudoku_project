"""Reproducible data generation. Usage: python make_data.py [--seed 42] [--n 20]"""
import argparse, csv, os
from csp.generator import make_puzzle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=20, help="puzzles per givens level")
    a = ap.parse_args()
    os.makedirs("data/generated", exist_ok=True)
    path = "data/generated/puzzles.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["puzzle_id", "seed", "n_givens", "puzzle", "solution"])
        for n_givens in (25, 30, 35):
            for k in range(a.n):
                s = a.seed * 1000 + k
                puz, sol = make_puzzle(s, n_givens)
                w.writerow([f"g{n_givens}_{k}", s, n_givens,
                            "".join(map(str, puz)), "".join(map(str, sol))])
    print("wrote", path)

if __name__ == "__main__":
    main()