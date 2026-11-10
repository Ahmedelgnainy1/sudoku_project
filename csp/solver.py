"""Backtracking Sudoku solver with switchable inference."""
from dataclasses import dataclass
from csp.grid import PEERS, conflicts
from csp.ac3 import ac3, initial_domains

@dataclass
class Stats:
    assignments: int = 0   # values placed during search
    backtracks: int = 0    # placed values undone
    revisions: int = 0     # AC-3 revise() calls
    removals: int = 0      # values deleted by AC-3

class Solver:
    def __init__(self, use_fc=False, use_ac3=False, use_mrv=False, use_degree=False):
        self.use_fc = use_fc
        self.use_ac3 = use_ac3
        self.use_mrv = use_mrv
        self.use_degree = use_degree
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
        if self.use_ac3:
            domains = initial_domains(g)
            return domains if ac3(domains, self.stats) else None
        domains = [set(range(1, 10)) for _ in range(81)]
        if self.use_fc:
            for i in range(81):
                if g[i]:
                    for p in PEERS[i]:
                        domains[p].discard(g[i])
            if any(g[i] == 0 and not domains[i] for i in range(81)):
                return None            # an empty cell has no legal digit
        return domains

    def _candidates(self, g, domains, i):
        if self.use_fc:
            return sorted(domains[i])
        base = sorted(domains[i]) if self.use_ac3 else range(1, 10)
        return [v for v in base if all(g[p] != v for p in PEERS[i])]

    def _select(self, g, domains):
        empties = [i for i in range(81) if g[i] == 0]
        if not (self.use_mrv or self.use_degree):
            return empties[0]              # static order (old behaviour)
        def key(i):
            dom = len(self._candidates(g, domains, i)) if self.use_mrv else 0
            deg = sum(1 for p in PEERS[i] if g[p] == 0) if self.use_degree else 0
            return (dom, -deg, i)          # fewest values, then most neighbours
        return min(empties, key=key)

    def _backtrack(self, g, domains):
        if 0 not in g:
            return True
        var = self._select(g, domains)
        candidates = self._candidates(g, domains, var)
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