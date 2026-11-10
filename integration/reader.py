"""Reading a mosaic: blank rule (no NN), then the NN reads the non-blank tiles."""
import numpy as np

BLANK_THRESHOLD = 0.1

def split_mosaic(mosaic):
    """252x252 -> (81, 28, 28), cell i = 9*row + col."""
    return np.ascontiguousarray(
        mosaic.reshape(9, 28, 9, 28).transpose(0, 2, 1, 3).reshape(81, 28, 28))

def is_blank(tile, thr=BLANK_THRESHOLD):
    return bool(tile.max() < thr)

def read_mosaic(mosaic, predict_proba, thr=BLANK_THRESHOLD):
    """Return (grid of 81 ints, 0 = blank; probs (81, 9), zero rows for blanks)."""
    tiles = split_mosaic(mosaic)
    idx = [i for i in range(81) if not is_blank(tiles[i], thr)]
    probs = np.zeros((81, 9))
    if idx:
        probs[idx] = predict_proba(tiles[idx])
    grid = [0] * 81
    for i in idx:
        grid[i] = int(probs[i].argmax()) + 1
    return grid, probs