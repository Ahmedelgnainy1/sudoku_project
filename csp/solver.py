"""Backtracking Sudoku solver with switchable inference."""
from dataclasses import dataclass
from csp.grid import PEERS, conflicts

@dataclass
class Stats:
    assignments: int = 0   # values placed during search
    backtracks: int = 0    # placed values undone

class Solver:
    def __init__(self, use_fc=False):
        self.use_fc = use_fc
        self.stats = Stats()

    def solve(self, grid):
        """Return a solved copy of grid, or None if there is no solution."""
        self.stats = Stats()
        if conflicts(grid):
            return None
        g = list(grid)
        domains = self._initial_domains(g)
        if domains is None:
            return None
        return g if self._backtrack(g, domains) else None

    def _initial_domains(self, g):
        domains = [set(range(1, 10)) for _ in range(81)]
        if self.use_fc:
            for i in range(81):
                if g[i]:
                    for p in PEERS[i]:
                        domains[p].discard(g[i])
            if any(g[i] == 0 and not domains[i] for i in range(81)):
                return None            # an empty cell has no legal digit
        return domains

    def _backtrack(self, g, domains):
        if 0 not in g:
            return True
        var = g.index(0)
        if self.use_fc:
            candidates = sorted(domains[var])
        else:
            candidates = [v for v in range(1, 10)
                          if all(g[p] != v for p in PEERS[var])]
        for v in candidates:
            g[var] = v
            self.stats.assignments += 1
            trail, wipeout = [], False
            if self.use_fc:
                for p in PEERS[var]:
                    if g[p] == 0 and v in domains[p]:
                        domains[p].discard(v)
                        trail.append(p)
                        if not domains[p]:
                            wipeout = True
                            break
            if not wipeout and self._backtrack(g, domains):
                return True
            for p in trail:            # undo domain changes
                domains[p].add(v)
            g[var] = 0
            self.stats.backtracks += 1
        return False