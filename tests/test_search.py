import math, numpy as np, pytest
from search.correct import candidate_moves, search, apply, h_admissible, h_conflicts

def probs_for(grid, alt):
    """probs rows: read digit 0.6, alt[cell] gets 0.3, rest share 0.1."""
    p = np.zeros((81, 9))
    for i, d in enumerate(grid):
        if d == 0: continue
        p[i, :] = 0.1 / 7
        p[i, d - 1] = 0.6
        p[i, alt.get(i, d % 9 + 1) - 1] = 0.3
    return p

# Tiny known example: row 0 reads 1 2 3 4 5 6 7 8 1 (true last digit = 9, cell 8)
# plus an extra wrong cell 40 whose fix is cheaper than fixing cell 8.
GRID = [0] * 81
GRID[0:9] = [1, 2, 3, 4, 5, 6, 7, 8, 1]
goal_no_conflict = lambda g: not __import__("csp.grid", fromlist=["x"]).conflicts(g)

def test_candidate_costs_never_negative():
    P = probs_for(GRID, {8: 9})
    for opts in candidate_moves(GRID, P).values():
        assert all(c >= 0 for _, c in opts)

def test_known_answer_cheapest_fix():
    # fixing cell 8 -> 9 costs log(.6/.3); fixing cell 0 -> 9 costs the same if alt=9.
    P = probs_for(GRID, {8: 9, 0: 2})            # cell 0 alt = 2 (no help)
    state, cost, _ = search(GRID, P, goal_no_conflict, "ucs")
    assert state == frozenset({(8, 9)})
    assert cost == pytest.approx(math.log(0.6 / 0.3))

@pytest.mark.parametrize("m", ["ucs", "astar", "greedy"])
def test_all_methods_find_a_goal(m):
    P = probs_for(GRID, {8: 9})
    state, cost, st = search(GRID, P, goal_no_conflict, m)
    assert state is not None and goal_no_conflict(apply(GRID, state))
    assert st.expanded >= 1 and st.generated >= st.expanded

def test_ucs_and_astar_same_cost():
    P = probs_for(GRID, {8: 9})
    c1 = search(GRID, P, goal_no_conflict, "ucs")[1]
    c2 = search(GRID, P, goal_no_conflict, "astar")[1]
    assert c1 == pytest.approx(c2)

def test_astar_expands_no_more_than_ucs():
    P = probs_for(GRID, {8: 9})
    ucs = search(GRID, P, goal_no_conflict, "ucs")[2]
    ast = search(GRID, P, goal_no_conflict, "astar")[2]
    assert ast.expanded <= ucs.expanded

# ---- edge ----
def test_no_correction_needed():
    good = [0] * 81; good[0:3] = [1, 2, 3]
    state, cost, st = search(good, probs_for(good, {}), goal_no_conflict, "ucs")
    assert state == frozenset() and cost == 0.0 and st.expanded == 1

def test_max_corr_zero_limits_search():
    P = probs_for(GRID, {8: 9})
    state, _, _ = search(GRID, P, goal_no_conflict, "ucs", max_corr=0)
    assert state is None

def test_heuristic_zero_when_no_conflict():
    good = [0] * 81; good[0:3] = [1, 2, 3]
    assert h_admissible(good, frozenset(), candidate_moves(good, probs_for(good, {}))) == 0.0

# ---- impossible ----
def test_impossible_goal_returns_none():
    P = probs_for(GRID, {8: 9})
    state, cost, st = search(GRID, P, lambda g: False, "ucs")
    assert state is None and cost is None and st.expanded > 0

def test_h_conflicts_counts_pairs():
    P = probs_for(GRID, {8: 9})
    assert h_conflicts(GRID, frozenset(), candidate_moves(GRID, P)) == 1.0