"""Correction search over reading mistakes. State = frozenset of (cell, new_digit)."""
import heapq, itertools, math
from dataclasses import dataclass
from csp.grid import conflicts

@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0

def candidate_moves(read_grid, probs, top_k=3):
    """{cell: [(digit, step_cost), ...]} for non-blank cells, cheapest first.
    cost = -log p_new + log p_old, clipped at 0 (p_old is the read digit)."""
    moves = {}
    for i in range(81):
        if read_grid[i] == 0:
            continue
        p_old = max(probs[i][read_grid[i] - 1], 1e-12)
        opts = []
        for d in range(1, 10):
            if d == read_grid[i]:
                continue
            p_new = max(probs[i][d - 1], 1e-12)
            opts.append((d, max(0.0, -math.log(p_new) + math.log(p_old))))
        opts.sort(key=lambda t: t[1])
        moves[i] = opts[:top_k]
    return moves

def apply(read_grid, state):
    g = list(read_grid)
    for cell, d in state:
        g[cell] = d
    return g

def h_admissible(read_grid, state, moves):
    """0 if no conflicts, else cheapest remaining single correction."""
    if not conflicts(apply(read_grid, state)):
        return 0.0
    used = {c for c, _ in state}
    costs = [o[0][1] for c, o in moves.items() if c not in used and o]
    return min(costs) if costs else float("inf")

def h_conflicts(read_grid, state, moves):
    """Greedy-only heuristic: number of conflicting pairs (not admissible)."""
    return float(len(conflicts(apply(read_grid, state))))

def search(read_grid, probs, goal, method="ucs", h=h_admissible,
           max_corr=3, top_k=3, trace=None):
    """Return (state, cost, stats) or (None, None, stats). goal(grid) -> bool.
    If trace is a list, each expanded state is appended as (state, g, goal_ok)."""
    stats = SearchStats()
    moves = candidate_moves(read_grid, probs, top_k)
    tie = itertools.count()
    start = frozenset()
    def f(g, state):
        hv = h(read_grid, state, moves)
        return {"ucs": g, "greedy": hv, "astar": g + hv}[method]
    frontier = [(f(0.0, start), next(tie), 0.0, start)]
    stats.generated = 1
    explored = set()
    while frontier:
        _, _, g, state = heapq.heappop(frontier)
        if state in explored:
            continue
        explored.add(state)
        stats.expanded += 1
        ok = goal(apply(read_grid, state))
        if trace is not None:
            trace.append((state, g, ok))
        if ok:
            return state, g, stats
        if len(state) >= max_corr:
            continue
        used = {c for c, _ in state}
        for cell, opts in moves.items():
            if cell in used:
                continue
            for d, cost in opts:
                nxt = state | {(cell, d)}
                if nxt in explored:
                    continue
                heapq.heappush(frontier, (f(g + cost, nxt), next(tie), g + cost, nxt))
                stats.generated += 1
    return None, None, stats