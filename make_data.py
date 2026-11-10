"""Reproducible data generation.
Usage: python make_data.py [--seed 42] [--n 20] [--part all|puzzles|mnist]"""
import argparse, csv, hashlib, os
from csp.generator import make_puzzle

def make_puzzles(seed, n):
    os.makedirs("data/generated", exist_ok=True)
    path = "data/generated/puzzles.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["puzzle_id", "seed", "n_givens", "puzzle", "solution"])
        for n_givens in (25, 30, 35):
            for k in range(n):
                s = seed * 1000 + k
                puz, sol = make_puzzle(s, n_givens)
                w.writerow([f"g{n_givens}_{k}", s, n_givens,
                            "".join(map(str, puz)), "".join(map(str, sol))])
    print("wrote", path)

def make_mnist(seed):
    from data.mnist import load_mnist
    d = load_mnist(seed=seed)                        # downloads once into data/raw
    raw = os.path.join("data", "raw", "MNIST", "raw")
    lines = []
    for name in sorted(os.listdir(raw)):
        with open(os.path.join(raw, name), "rb") as f:
            lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {name}")
    out = os.path.join("data", "raw", "checksums.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)
    for k in ("train", "val", "test"):
        print(k, d[f"x_{k}"].shape, "labels", sorted(set(d[f"y_{k}"].tolist())))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=20, help="puzzles per givens level")
    ap.add_argument("--part", choices=["all", "puzzles", "mnist"], default="all")
    a = ap.parse_args()
    if a.part in ("all", "puzzles"):
        make_puzzles(a.seed, a.n)
    if a.part in ("all", "mnist"):
        make_mnist(a.seed)

if __name__ == "__main__":
    main()