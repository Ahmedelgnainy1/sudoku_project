from csp.grid import parse, conflicts
from csp.ac3 import ac3, revise, initial_domains
from csp.solver import Solver, Stats

PUZZLE = ("530070000600195000098000060800060003400803001"
          "700020006060000280000419005000080079")
SOLUTION = ("534678912672195348198342567859761423426853791"
            "713924856961537284287419635345286179")

# ---- normal ----
def test_revise_removes_singleton_value():
    d = [set(range(1, 10)) for _ in range(81)]
    d[0], d[1] = {5}, {5, 7}
    assert revise(d, 1, 0) == 1 and d[1] == {7}

def test_revise_no_change_when_neighbour_not_singleton():
    d = [set(range(1, 10)) for _ in range(81)]
    d[0] = {5, 6}
    assert revise(d, 1, 0) == 0 and d[1] == set(range(1, 10))

def test_ac3_is_sound_on_known_puzzle():
    d = initial_domains(parse(PUZZLE))
    assert ac3(d) is True
    sol = parse(SOLUTION)
    assert all(sol[i] in d[i] for i in range(81))

def test_ac3_shrinks_domains():
    d = initial_domains(parse(PUZZLE))
    before = sum(len(x) for x in d)
    s = Stats(); ac3(d, s)
    assert sum(len(x) for x in d) < before and s.removals > 0


def test_solver_ac3_variants_solve():
    for fc in (False, True):
        assert Solver(use_fc=fc, use_ac3=True).solve(parse(PUZZLE)) == parse(SOLUTION)

# ---- edge ----
def test_ac3_solved_grid_no_removals():
    s = Stats(); d = initial_domains(parse(SOLUTION))
    assert ac3(d, s) and s.removals == 0

def test_ac3_empty_grid_no_removals():
    s = Stats(); d = initial_domains([0] * 81)
    assert ac3(d, s) and s.removals == 0 and s.revisions == 1620

def test_ac3_off_leaves_counters_zero():
    s = Solver(use_fc=True); s.solve(parse(PUZZLE))
    assert s.stats.revisions == 0 and s.stats.removals == 0

# ---- impossible ----
def test_ac3_detects_duplicate_givens():
    g = [0] * 81
    g[0] = g[8] = 5
    d = initial_domains(g)
    assert ac3(d) is False and any(len(x) == 0 for x in d)

def test_ac3_detects_cell_without_value():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    g[17] = 9
    assert ac3(initial_domains(g)) is False

def test_solver_fc_ac3_impossible_no_search():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    g[17] = 9
    s = Solver(use_fc=True, use_ac3=True)
    assert s.solve(g) is None and s.stats.assignments == 0