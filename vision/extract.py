"""Picture -> 81 MNIST-style tiles. numpy/PIL/scipy only."""
import numpy as np
from PIL import Image
from scipy import ndimage

def load_ink(path):
    """Grayscale 0..1, ink bright (inverts dark-on-light pictures)."""
    a = np.asarray(Image.open(path).convert("L"), np.float32) / 255
    if a.mean() > 0.5:
        a = 1 - a
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

def trim(a, thr=0.25):
    """Cut the empty border: keep the bounding box of everything bright."""
    ys, xs = np.where(a > thr)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def remove_lines(mask, cell):
    """Delete grid lines: any horizontal/vertical run longer than 0.8 cell."""
    L = max(3, int(0.8 * cell))
    h = ndimage.binary_opening(mask, structure=np.ones((1, L)))
    v = ndimage.binary_opening(mask, structure=np.ones((L, 1)))
    return mask & ~ndimage.binary_dilation(h | v, iterations=1)

def cell_to_tile(a_cell, m_cell):
    """Keep the largest digit-sized blob, centre it in a 28x28 tile (zeros = blank)."""
    n = a_cell.shape[0]
    k = int(0.06 * n)
    a_cell, m_cell = a_cell[k:n - k, k:n - k], m_cell[k:n - k, k:n - k]
    lab, cnt = ndimage.label(m_cell)
    best, size = None, 0
    for i in range(1, cnt + 1):
        ys, xs = np.where(lab == i)
        if len(ys) >= 20 and (ys.max() - ys.min() + 1) >= 0.25 * n and len(ys) > size:
            best, size = i, len(ys)
    if best is None:
        return np.zeros((28, 28), np.float32), False
    ys, xs = np.where(lab == best)
    crop = (a_cell * (lab == best))[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20 / max(h, w)
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    t = np.asarray(img, np.float32) / 255
    tile = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - t.shape[0]) // 2, (28 - t.shape[1]) // 2
    tile[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    return tile, True

def image_to_mosaic(path):
    """Returns (mosaic 252x252, info dict with the steps taken)."""
    full = load_ink(path)
    a = trim(full)
    mask = remove_lines(a > 0.5, a.shape[0] / 9)
    ch, cw = a.shape[0] / 9, a.shape[1] / 9
    mosaic = np.zeros((252, 252), np.float32)
    for r in range(9):
        for c in range(9):
            sl = (slice(round(r * ch), round((r + 1) * ch)),
                  slice(round(c * cw), round((c + 1) * cw)))
            tile, _ = cell_to_tile(a[sl], mask[sl])
            mosaic[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = tile
    return mosaic, dict(original=full.shape, cropped=a.shape)