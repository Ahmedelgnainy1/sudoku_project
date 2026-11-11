import pytest
from csp.grid import parse
from csp.solver import Solver, BudgetExceeded

PUZZLE = ("530070000600195000098000060800060003400803001"
          "700020006060000280000419005000080079")

def test_budget_raises_when_too_small():
    with pytest.raises(BudgetExceeded):
        Solver(max_assignments=100).solve(parse(PUZZLE))

def test_large_budget_does_not_raise():
    assert Solver(max_assignments=10**6).solve(parse(PUZZLE)) is not None

def test_default_has_no_budget():
    s = Solver(); s.solve(parse(PUZZLE))
    assert (s.stats.assignments, s.stats.backtracks) == (4208, 4157)