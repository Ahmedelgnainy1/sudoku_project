import math, numpy as np
from csp.grid import parse, conflicts
from integration.pipeline import solve_reading

PUZZLE = parse("530070000600195000098000060800060003400803001"
               "700020006060000280000419005000080079")
SOLUTION = parse("534678912672195348198342567859761423426853791"
                 "713924856961537284287419635345286179")

def make_probs(grid, overrides=None):
    """read digit 0.6, next digit 0.3, rest share 0.1; overrides {cell: (read_p, true_digit, true_p)}."""
    p = np.zeros((81, 9))
    for i, d in enumerate(grid):
        if d == 0: continue
        p[i, :] = 0.1 / 7
        p[i, d - 1] = 0.6
        p[i, d % 9] = 0.3
    for i, (rp, td, tp) in (overrides or {}).items():
        p[i, :] = 0.05 / 7
        p[i, grid[i] - 1] = rp
        p[i, td - 1] = tp
    return p

# ---- normal ----
def test_clean_reading_needs_no_correction():
    r = solve_reading(PUZZLE, make_probs(PUZZLE))
    assert r.status == "solved" and r.corrections == [] and r.cost == 0.0
    assert r.solution == SOLUTION

def test_one_misread_is_corrected_and_logged():
    g = list(PUZZLE); g[0] = 3                      # true digit 5; conflicts with cell 1
    r = solve_reading(g, make_probs(g, {0: (0.5, 5, 0.45)}))
    assert r.status == "solved" and conflicts(r.solution) == [] and 0 not in r.solution
    assert r.corrections[0]["cell"] == 0 and r.corrections[0]["read"] == 3
    assert r.corrections[0]["new"] == 5
    assert math.isclose(r.cost, math.log(0.5 / 0.45))

def test_all_methods_agree_on_solved():
    g = list(PUZZLE); g[0] = 3
    P = make_probs(g, {0: (0.5, 5, 0.45)})
    for m in ("ucs", "astar", "greedy"):
        assert solve_reading(g, P, method=m).status == "solved"

def test_stats_are_recorded():
    g = list(PUZZLE); g[0] = 3
    r = solve_reading(g, make_probs(g, {0: (0.5, 5, 0.45)}))
    assert r.search_stats.expanded >= 1 and r.goal_calls >= 1
    assert r.solver_stats.assignments >= 0

def test_correction_cost_never_negative():
    g = list(PUZZLE); g[0] = 3
    r = solve_reading(g, make_probs(g, {0: (0.5, 5, 0.45)}))
    assert all(c["cost"] >= 0 for c in r.corrections)

# ---- edge ----
def test_all_blank_grid_solved_cost_zero():
    r = solve_reading([0] * 81, np.zeros((81, 9)))
    assert r.status == "solved" and r.cost == 0.0

def test_already_solved_reading():
    r = solve_reading(SOLUTION, make_probs(SOLUTION))
    assert r.status == "solved" and r.corrections == []

def test_max_corr_zero_does_not_fix_misread():
    g = list(PUZZLE); g[0] = 3
    r = solve_reading(g, make_probs(g, {0: (0.5, 5, 0.45)}), max_corr=0)
    assert r.status == "unreadable"

# ---- impossible ----
def test_unreadable_reports_message_and_least_confident():
    g = [0] * 81; g[0] = g[8] = 5
    r = solve_reading(g, make_probs(g), max_corr=0)
    assert r.status == "unreadable" and r.solution is None
    assert "unreadable" in r.message
    assert {c for c, _ in r.least_confident} == {0, 8}

def test_unreadable_when_cell_has_no_value():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]; g[17] = 9
    r = solve_reading(g, make_probs(g), max_corr=0)
    assert r.status == "unreadable" and len(r.least_confident) <= 5