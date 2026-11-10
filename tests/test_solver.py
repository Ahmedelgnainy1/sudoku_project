from csp.grid import parse, conflicts
from csp.solver import Solver
import pytest

PUZZLE = ("530070000600195000098000060800060003400803001"
          "700020006060000280000419005000080079")
SOLUTION = ("534678912672195348198342567859761423426853791"
            "713924856961537284287419635345286179")

# ---- normal cases ----
def test_solves_known_puzzle():
    assert Solver().solve(parse(PUZZLE)) == parse(SOLUTION)

def test_solution_is_valid_and_keeps_givens():
    g = parse(PUZZLE)
    sol = Solver().solve(g)
    assert 0 not in sol and conflicts(sol) == []
    assert all(sol[i] == g[i] for i in range(81) if g[i])

def test_counters_are_recorded():
    g = parse(PUZZLE)
    s = Solver()
    s.solve(g)
    assert s.stats.assignments >= g.count(0)   # every empty cell needs one

# ---- edge cases ----
def test_already_solved_grid():
    s = Solver()
    assert s.solve(parse(SOLUTION)) == parse(SOLUTION)
    assert s.stats.assignments == 0

def test_empty_grid_is_solvable():
    sol = Solver().solve([0] * 81)
    assert 0 not in sol and conflicts(sol) == []

def test_input_grid_not_modified():
    g = parse(PUZZLE)
    before = list(g)
    Solver().solve(g)
    assert g == before

# ---- impossible cases ----
def test_impossible_duplicate_givens():
    g = [0] * 81
    g[0] = g[8] = 5                       # two 5s in row 0
    s = Solver()
    assert s.solve(g) is None and s.stats.assignments == 0

def test_impossible_cell_with_no_value():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]    # row 0 needs a 9 in cell 8...
    g[17] = 9                             # ...but column 8 already has a 9
    assert conflicts(g) == []             # the givens alone look fine
    assert Solver().solve(g) is None

    # ---- Phase 3a: Forward Checking ----
def test_bt_counters_unchanged_by_refactor():
    s = Solver(use_fc=False)
    s.solve(parse(PUZZLE))
    assert (s.stats.assignments, s.stats.backtracks) == (4208, 4157)

def test_fc_solves_known_puzzle():
    assert Solver(use_fc=True).solve(parse(PUZZLE)) == parse(SOLUTION)

def test_fc_uses_fewer_assignments_than_bt():
    bt, fc = Solver(use_fc=False), Solver(use_fc=True)
    bt.solve(parse(PUZZLE)); fc.solve(parse(PUZZLE))
    assert fc.stats.assignments < bt.stats.assignments

@pytest.mark.parametrize("fc", [False, True])
def test_counter_invariant(fc):
    g = parse(PUZZLE)
    s = Solver(use_fc=fc)
    s.solve(g)
    assert s.stats.assignments - s.stats.backtracks == g.count(0)

def test_fc_empty_grid_is_solvable():
    sol = Solver(use_fc=True).solve([0] * 81)
    assert 0 not in sol and conflicts(sol) == []

def test_fc_impossible_duplicate_givens():
    g = [0] * 81
    g[0] = g[8] = 5
    assert Solver(use_fc=True).solve(g) is None

def test_fc_impossible_detected_before_search():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    g[17] = 9
    s = Solver(use_fc=True)
    assert s.solve(g) is None and s.stats.assignments == 0