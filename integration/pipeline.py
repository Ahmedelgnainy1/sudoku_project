"""Pipeline: read grid + probs -> search over corrections -> CSP validates -> log."""
import math
from dataclasses import dataclass, field
from csp.grid import conflicts
from csp.solver import Solver
from search.correct import search, apply, h_admissible

class GoalBudget(Exception):
    pass

@dataclass
class Result:
    status: str                      # "solved" or "unreadable"
    solution: list = None
    corrections: list = field(default_factory=list)   # dicts: cell, read, new, cost
    cost: float = None
    search_stats: object = None
    solver_stats: object = None
    goal_calls: int = 0
    least_confident: list = field(default_factory=list)
    message: str = ""
    capped: bool = False

SOLVER_KW = dict(use_fc=True, use_ac3=True, use_mrv=True)

def least_confident_cells(read_grid, probs, n=5):
    cells = [i for i in range(81) if read_grid[i]]
    cells.sort(key=lambda i: probs[i].max())
    return [(i, float(probs[i].max())) for i in cells[:n]]

def _step_cost(probs, cell, old, new):
    p_old, p_new = max(probs[cell][old - 1], 1e-12), max(probs[cell][new - 1], 1e-12)
    return max(0.0, -math.log(p_new) + math.log(p_old))

def solve_reading(read_grid, probs, method="ucs", h=h_admissible,
                  max_corr=3, top_k=3, solver_kw=None, max_goal_calls=None):
    kw = solver_kw or SOLVER_KW
    calls = [0]
    def goal(g):
        calls[0] += 1
        if max_goal_calls and calls[0] > max_goal_calls:
            raise GoalBudget
        if conflicts(g):
            return False
        return Solver(**kw).solve(g) is not None
    try:
        state, cost, sstats = search(read_grid, probs, goal, method, h, max_corr, top_k)
    except GoalBudget:
        return Result("unreadable", goal_calls=calls[0], capped=True,
                      least_confident=least_confident_cells(read_grid, probs),
                      message="unreadable: goal-call budget exceeded")
    if state is None:
        return Result("unreadable", search_stats=sstats, goal_calls=calls[0],
                      least_confident=least_confident_cells(read_grid, probs),
                      message=f"unreadable: no solvable puzzle within {max_corr} corrections")
    final = apply(read_grid, state)
    solver = Solver(**kw)
    solution = solver.solve(final)
    log = [dict(cell=c, read=read_grid[c], new=d,
                cost=_step_cost(probs, c, read_grid[c], d)) for c, d in sorted(state)]
    return Result("solved", solution, log, cost, sstats, solver.stats, calls[0],
                  message="solved" if not log else f"solved with {len(log)} correction(s)")