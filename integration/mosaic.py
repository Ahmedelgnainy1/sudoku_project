"""Mosaic builder with hard-digit injection (documented rule):
- injected cell: random test image of the right digit that the CNN misreads
  but whose true digit is its 2nd choice ("hard pool")
- other given cells: random test image of that digit the CNN reads correctly ("easy pool")
- empty cells: all-zero 28x28 tile"""
import numpy as np

def build_pools(y, probs):
    """y: labels 1..9; probs: (n, 9) CNN probabilities. Returns {'easy':{d:idx}, 'hard':{d:idx}}."""
    y0 = y - 1
    order = np.argsort(-probs, axis=1)
    top1, top2 = order[:, 0], order[:, 1]
    easy, hard = {}, {}
    for d in range(1, 10):
        same = y0 == d - 1
        easy[d] = np.where(same & (top1 == y0))[0]
        hard[d] = np.where(same & (top1 != y0) & (top2 == y0))[0]
    return {"easy": easy, "hard": hard}

def make_mosaic(puzzle, x, pools, m, seed):
    """Return (mosaic 252x252, src, injected).
    src[i] = index of the source image for cell i (-1 = blank); injected = sorted cell ids."""
    rng = np.random.default_rng(seed)
    eligible = [i for i in range(81) if puzzle[i] and len(pools["hard"][puzzle[i]]) > 0]
    if m > len(eligible):
        raise ValueError(f"no eligible cells: need {m}, have {len(eligible)}")
    injected = sorted(int(i) for i in rng.choice(eligible, m, replace=False)) if m else []
    mosaic = np.zeros((252, 252), dtype=np.float32)
    src = []
    for i in range(81):
        d = puzzle[i]
        if d == 0:
            src.append(-1)
            continue
        pool = pools["hard"][d] if i in injected else pools["easy"][d]
        k = int(rng.choice(pool))
        src.append(k)
        r, c = divmod(i, 9)
        mosaic[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = x[k]
    return mosaic, src, injected