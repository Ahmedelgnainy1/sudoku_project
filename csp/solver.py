"""Plain backtracking Sudoku solver (no inference yet)."""
from dataclasses import dataclass
from csp.grid import PEERS, conflicts

@dataclass
class Stats:
    assignments: int = 0   # values placed
    backtracks: int = 0    # placed values undone

class Solver:
    def __init__(self):
        self.stats = Stats()

    def solve(self, grid):
        """Return a solved copy of grid, or None if there is no solution."""
        self.stats = Stats()
        if conflicts(grid):            # two equal givens in a unit
            return None
        work = list(grid)              # never modify the caller's grid
        return work if self._backtrack(work) else None

    def _backtrack(self, g):
        if 0 not in g:                 # complete assignment
            return True
        var = g.index(0)               # first empty cell
        for v in range(1, 10):
            if all(g[p] != v for p in PEERS[var]):   # consistent?
                g[var] = v
                self.stats.assignments += 1
                if self._backtrack(g):
                    return True
                g[var] = 0             # undo
                self.stats.backtracks += 1
        return False