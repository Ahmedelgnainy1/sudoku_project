"""Seeded Sudoku generator with uniqueness check."""
import random
from csp.grid import PEERS

def _candidates(g, i):
    used = {g[p] for p in PEERS[i]}
    return [v for v in range(1, 10) if v not in used]

def count_solutions(grid, limit=2):
    """Count solutions, stopping at `limit`. Uses MRV for speed."""
    g = list(grid)
    count = 0
    def rec():
        nonlocal count
        best, best_c = None, None
        for i in range(81):
            if g[i] == 0:
                c = _candidates(g, i)
                if best is None or len(c) < len(best_c):
                    best, best_c = i, c
                    if not c:
                        return
        if best is None:
            count += 1
            return
        for v in best_c:
            g[best] = v
            rec()
            g[best] = 0
            if count >= limit:
                return
    rec()
    return count

def random_full_grid(rng):
    g = [0] * 81
    def rec(i):
        if i == 81:
            return True
        digits = _candidates(g, i)
        rng.shuffle(digits)
        for v in digits:
            g[i] = v
            if rec(i + 1):
                return True
        g[i] = 0
        return False
    rec(0)
    return g

def make_puzzle(seed, n_givens, max_tries=20):
    """Return (puzzle, solution) with exactly n_givens givens and a unique solution."""
    for attempt in range(max_tries):
        rng = random.Random(f"{seed}-{n_givens}-{attempt}")
        sol = random_full_grid(rng)
        puz = list(sol)
        order = list(range(81))
        rng.shuffle(order)
        givens = 81
        for i in order:
            if givens == n_givens:
                break
            keep = puz[i]
            puz[i] = 0
            if count_solutions(puz) == 1:
                givens -= 1
            else:
                puz[i] = keep
        if givens == n_givens:
            return puz, sol
    raise RuntimeError(f"could not reach {n_givens} givens for seed {seed}")