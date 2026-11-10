"""Sudoku grid helpers. A grid is a list of 81 ints; 0 means empty.
Cell index i = 9*row + col."""

def row_of(i): return i // 9
def col_of(i): return i % 9
def box_of(i): return 3 * (i // 27) + (i % 9) // 3

def _build_peers():
    peers = []
    for i in range(81):
        same = {j for j in range(81) if j != i and
                (row_of(j) == row_of(i) or col_of(j) == col_of(i)
                 or box_of(j) == box_of(i))}
        peers.append(frozenset(same))
    return tuple(peers)

PEERS = _build_peers()          # PEERS[i] = the 20 cells that must differ from i

def parse(s):
    """'530070000...' or with dots -> list of 81 ints."""
    chars = [c for c in s if c in "0123456789."]
    if len(chars) != 81:
        raise ValueError(f"expected 81 cells, got {len(chars)}")
    return [0 if c in "0." else int(c) for c in chars]

def conflicts(grid):
    """All pairs (i, j), i < j, of peers holding the same nonzero digit."""
    return [(i, j) for i in range(81) if grid[i]
            for j in PEERS[i] if j > i and grid[j] == grid[i]]