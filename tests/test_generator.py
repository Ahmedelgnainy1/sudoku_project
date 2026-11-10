import pytest
from csp.grid import parse, conflicts
from csp.generator import count_solutions, random_full_grid, make_puzzle
import random

PUZZLE = ("530070000600195000098000060800060003400803001"
          "700020006060000280000419005000080079")
SOLUTION = ("534678912672195348198342567859761423426853791"
            "713924856961537284287419635345286179")

# ---- normal ----
def test_count_known_puzzle_is_unique():
    assert count_solutions(parse(PUZZLE)) == 1

def test_full_grid_is_valid():
    g = random_full_grid(random.Random(1))
    assert 0 not in g and conflicts(g) == []

def test_same_seed_same_puzzle():
    assert make_puzzle(7, 35) == make_puzzle(7, 35)

def test_different_seed_different_puzzle():
    assert make_puzzle(1, 35)[0] != make_puzzle(2, 35)[0]

@pytest.mark.parametrize("n", [35, 30])
def test_puzzle_has_n_givens_and_unique_solution(n):
    puz, sol = make_puzzle(3, n)
    assert sum(1 for x in puz if x) == n
    assert count_solutions(puz) == 1
    assert all(p == 0 or p == s for p, s in zip(puz, sol))

# ---- edge ----
def test_count_solved_grid_is_one():
    assert count_solutions(parse(SOLUTION)) == 1

def test_count_empty_grid_stops_at_limit():
    assert count_solutions([0] * 81, limit=2) == 2

def test_two_solution_puzzle():
    g = parse(SOLUTION)
    for i in (0, 1, 9, 10):          # swap-able 2x2 pattern may give 2+ solutions
        g[i] = 0
    assert count_solutions(g) >= 1

# ---- impossible ----
def test_count_zero_for_contradiction():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    g[17] = 9
    assert count_solutions(g) == 0

