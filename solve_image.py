"""Usage: python solve_image.py IMAGE_PATH [--debug]"""
import sys
import numpy as np, torch
from PIL import Image
from csp.grid import conflicts
from csp.ac3 import ac3, initial_domains
from csp.solver import Stats
from nn.cnn import CNN, get_logits
from nn.calibration import softmax
from integration.reader import read_mosaic
from integration.pipeline import solve_reading
from vision.extract import image_to_mosaic
import os

def show(g, mark=()):
    for r in range(9):
        print("  " + " ".join(f"{g[9*r+c] or '.'}{'*' if 9*r+c in mark else ' '}" for c in range(9)))

def main(path, debug=False):
    print("STEP 1  Load the picture, make ink bright, trim the empty border")
    mosaic, info = image_to_mosaic(path)
    print(f"        size {info['original']} -> grid {info['cropped']}")
    print("STEP 2  Cut 9x9 cells, remove grid lines, keep the digit blob in each cell")
    if debug:
        Image.fromarray((mosaic * 255).astype(np.uint8)).resize((756, 756), Image.NEAREST).save("debug_tiles.png")
        print("        wrote debug_tiles.png")
    wpath = "models/cnn_print.pt" if os.path.exists("models/cnn_print.pt") else "models/cnn.pt"
    cnn = CNN(); cnn.load_state_dict(torch.load(wpath)); cnn.eval()
    print("model:", wpath)
    grid, probs = read_mosaic(mosaic, lambda t: softmax(get_logits(cnn, t)))
    n = sum(1 for x in grid if x)
    print(f"STEP 3  Blank rule + CNN reads {n} digits (* = confidence below 0.90):")
    show(grid, [i for i in range(81) if grid[i] and probs[i].max() < 0.90])
    if n < 10:
        print("STOP    fewer than 17 digits read: not a valid Sudoku, or the picture is not supported")
        return
    print("STEP 4  Check the reading for conflicts")
    conf = conflicts(grid)
    d = initial_domains(grid); st = Stats()
    ok = ac3(d, st)
    print(f"        duplicate pairs: {len(conf)}; AC-3: {'consistent' if ok else 'a domain became empty'} "
          f"({st.revisions} revisions, {st.removals} values removed)")
    print("STEP 5  A* search over corrections (max 3), CSP checks each candidate")
    trace = []
    res = solve_reading(grid, probs, method="astar", max_goal_calls=5000, trace=trace)
    for k, (state, g, okk) in enumerate(trace[:8]):
        print(f"        try {k}: {sorted(state)} cost {g:.3f} -> CSP {'accepts' if okk else 'rejects'}")
    if len(trace) > 8:
        print(f"        ... {len(trace) - 8} more tries")
    print(f"        search nodes expanded: {res.search_stats.expanded if res.search_stats else '-'}, CSP checks: {res.goal_calls}")
    print("\nRESULT ", res.status, "-", res.message)
    if res.status != "solved":
        print("        least confident cells (row, col, prob):",
              [(c // 9 + 1, c % 9 + 1, round(p, 2)) for c, p in res.least_confident])
        return
    for c in res.corrections:
        print(f"STEP 6  Corrected row {c['cell']//9+1}, col {c['cell']%9+1}: {c['read']} -> {c['new']} (cost {c['cost']:.3f})")
    s = res.solver_stats
    print(f"STEP 7  CSP solved it with Forward Checking + AC-3 + MRV: "
          f"{s.assignments} assignments, {s.backtracks} backtracks, {s.revisions} AC-3 revisions")
    print("SOLUTION")
    show(res.solution)
    if n < 25:
        print("note: few givens, so several valid solutions may exist; this is one of them")

if __name__ == "__main__":
    main(sys.argv[1], "--debug" in sys.argv)