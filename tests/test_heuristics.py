import pytest
from csp.grid import parse, conflicts
from csp.solver import Solver

PUZZLE = ("530070000600195000098000060800060003400803001"
          "700020006060000280000419005000080079")
SOLUTION = ("534678912672195348198342567859761423426853791"
            "713924856961537284287419635345286179")

COMBOS = [dict(use_fc=True, use_mrv=True),
          dict(use_fc=True, use_degree=True),
          dict(use_fc=True, use_mrv=True, use_degree=True),
          dict(use_fc=True, use_ac3=True, use_mrv=True, use_degree=True)]

# ---- normal ----
@pytest.mark.parametrize("kw", COMBOS)
def test_heuristics_solve_known_puzzle(kw):
    assert Solver(**kw).solve(parse(PUZZLE)) == parse(SOLUTION)

@pytest.mark.parametrize("kw", COMBOS)
def test_heuristics_counter_invariant(kw):
    g = parse(PUZZLE)
    s = Solver(**kw); s.solve(g)
    assert s.stats.assignments - s.stats.backtracks == g.count(0)

def test_defaults_unchanged():
    s = Solver(); s.solve(parse(PUZZLE))
    assert (s.stats.assignments, s.stats.backtracks) == (4208, 4157)

# ---- edge ----
def test_heuristics_empty_grid():
    sol = Solver(use_fc=True, use_mrv=True, use_degree=True).solve([0] * 81)
    assert 0 not in sol and conflicts(sol) == []

def test_heuristics_solved_grid():
    s = Solver(use_fc=True, use_mrv=True)
    assert s.solve(parse(SOLUTION)) == parse(SOLUTION)
    assert s.stats.assignments == 0

# ---- impossible ----
def test_heuristics_impossible_no_value():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    g[17] = 9
    s = Solver(use_fc=True, use_mrv=True, use_degree=True)
    assert s.solve(g) is None and s.stats.assignments == 0