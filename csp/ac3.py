"""AC-3 arc consistency for Sudoku (binary constraint: cell != peer)."""
from collections import deque
from csp.grid import PEERS

def initial_domains(grid):
    """Givens become singleton domains; empty cells get 1..9."""
    return [{v} if v else set(range(1, 10)) for v in grid]

def revise(domains, i, j):
    """Remove from D(i) any value with no support in D(j). Returns number removed (0 or 1)."""
    if len(domains[j]) == 1:
        (v,) = domains[j]
        if v in domains[i]:
            domains[i].discard(v)
            return 1
    return 0

def ac3(domains, stats=None):
    """Prune domains in place. Returns False if some domain becomes empty."""
    queue = deque((i, j) for i in range(81) for j in PEERS[i])
    while queue:
        i, j = queue.popleft()
        if stats is not None:
            stats.revisions += 1
        if revise(domains, i, j):
            if stats is not None:
                stats.removals += 1
            if not domains[i]:
                return False
            for k in PEERS[i]:
                if k != j:
                    queue.append((k, i))
    return True