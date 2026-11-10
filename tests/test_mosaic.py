import numpy as np, pytest
from csp.grid import parse
from integration.mosaic import make_mosaic, build_pools
from integration.reader import split_mosaic, is_blank, read_mosaic

PUZZLE = parse("530070000600195000098000060800060003400803001"
               "700020006060000280000419005000080079")

# fake data: image 2*(d-1) is "easy" for digit d, image 2*(d-1)+1 is "hard"
X = np.random.default_rng(0).uniform(0.3, 1.0, (18, 28, 28)).astype(np.float32)
POOLS = {"easy": {d: np.array([2 * (d - 1)]) for d in range(1, 10)},
         "hard": {d: np.array([2 * (d - 1) + 1]) for d in range(1, 10)}}

# ---- normal ----
def test_mosaic_shape_and_blanks():
    mos, src, inj = make_mosaic(PUZZLE, X, POOLS, 2, seed=1)
    assert mos.shape == (252, 252)
    tiles = split_mosaic(mos)
    for i in range(81):
        assert (src[i] == -1) == (PUZZLE[i] == 0)
        assert (tiles[i] == 0).all() == (PUZZLE[i] == 0)

def test_injected_cells_use_hard_images():
    mos, src, inj = make_mosaic(PUZZLE, X, POOLS, 3, seed=2)
    assert len(inj) == 3
    for i in range(81):
        d = PUZZLE[i]
        if d == 0: continue
        assert src[i] == (2 * (d - 1) + 1 if i in inj else 2 * (d - 1))

def test_split_matches_source_images():
    mos, src, _ = make_mosaic(PUZZLE, X, POOLS, 1, seed=3)
    tiles = split_mosaic(mos)
    for i in range(81):
        if src[i] >= 0:
            assert np.array_equal(tiles[i], X[src[i]])

def test_same_seed_same_mosaic():
    a = make_mosaic(PUZZLE, X, POOLS, 2, seed=5)
    b = make_mosaic(PUZZLE, X, POOLS, 2, seed=5)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]

def test_read_mosaic_with_fake_predictor():
    # fake tiles: constant value 0.1*digit + 0.05; predictor decodes it
    tiles = np.zeros((81, 28, 28), np.float32)
    for i, d in enumerate(PUZZLE):
        if d: tiles[i] = 0.1 * d + 0.05
    mos = tiles.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)
    def predict(t):
        p = np.zeros((len(t), 9))
        for k, tile in enumerate(t):
            p[k, int(round((tile.max() - 0.05) / 0.1)) - 1] = 1.0
        return p
    grid, probs = read_mosaic(mos, predict)
    assert grid == PUZZLE and (probs[PUZZLE.index(0)] == 0).all()

# ---- edge ----
def test_zero_misreads_injects_nothing():
    _, _, inj = make_mosaic(PUZZLE, X, POOLS, 0, seed=1)
    assert inj == []

def test_blank_rule_threshold():
    assert is_blank(np.zeros((28, 28), np.float32))
    assert is_blank(np.full((28, 28), 0.05, np.float32))
    assert not is_blank(np.full((28, 28), 0.5, np.float32))

def test_build_pools_rules():
    y = np.array([1, 1, 2, 2])
    probs = np.array([[.9, .05, .05] + [0] * 6,     # digit 1, correct      -> easy[1]
                      [.3, .6, .1] + [0] * 6,       # digit 1 read as 2, 1 is 2nd -> hard[1]
                      [.1, .2, .7] + [0] * 6,       # digit 2 read as 3, 2 is 2nd -> hard[2]
                      [.1, .1, .8] + [0] * 6])      # digit 2 read as 3, 2 not top-2 -> neither
    p = build_pools(y, probs)
    assert p["easy"][1].tolist() == [0] and p["hard"][1].tolist() == [1]
    assert p["hard"][2].tolist() == [2] and p["easy"][2].tolist() == []

# ---- impossible ----
def test_too_many_misreads_raises():
    with pytest.raises(ValueError, match="no eligible cells"):
        make_mosaic(PUZZLE, X, {"easy": POOLS["easy"], "hard": {d: np.array([]) for d in range(1, 10)}}, 1, seed=1)

def test_empty_puzzle_cannot_inject():
    with pytest.raises(ValueError, match="no eligible cells"):
        make_mosaic([0] * 81, X, POOLS, 1, seed=1)