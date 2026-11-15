"""Generated printed-digit tiles (Windows fonts), passed through the SAME cell_to_tile as real pictures."""
import os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vision.extract import cell_to_tile

FONTS = ["arial.ttf", "calibri.ttf", "segoeui.ttf", "verdana.ttf",
         "tahoma.ttf", "times.ttf", "cour.ttf", "consola.ttf", "arialbd.ttf"]

def available_fonts():
    d = r"C:\Windows\Fonts"
    return [os.path.join(d, f) for f in FONTS if os.path.exists(os.path.join(d, f))]

def make_printed(n_per_digit, seed, fonts):
    rng = random.Random(seed)
    X, y = [], []
    for d in range(1, 10):
        got = 0
        while got < n_per_digit:
            n = rng.randint(28, 64)                       # cell size in pixels
            f = ImageFont.truetype(rng.choice(fonts), int(n * rng.uniform(0.5, 0.75)))
            im = Image.new("L", (n, n), 0)
            ImageDraw.Draw(im).text((n // 2 + rng.randint(-2, 2), n // 2 + rng.randint(-2, 2)),
                                    str(d), fill=rng.randint(200, 255), font=f, anchor="mm")
            a = np.asarray(im, np.float32) / 255
            a = (a - a.min()) / (a.max() - a.min() + 1e-9)
            tile, ok = cell_to_tile(a, a > 0.5)
            if ok:
                X.append(tile); y.append(d); got += 1
    return np.stack(X), np.array(y, dtype=np.int64)