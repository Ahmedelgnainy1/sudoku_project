import pytest
from csp.grid import PEERS, box_of, parse, conflicts

def test_every_cell_has_20_peers():
    assert all(len(p) == 20 for p in PEERS)

def test_box_of_corners():
    assert [box_of(i) for i in (0, 8, 40, 72, 80)] == [0, 2, 4, 6, 8]

def test_parse_accepts_dots_and_zeros():
    assert parse("." * 81) == parse("0" * 81) == [0] * 81

def test_parse_rejects_wrong_length():
    with pytest.raises(ValueError):
        parse("123")

def test_conflicts_finds_duplicate_in_row():
    g = [0] * 81
    g[0] = g[8] = 5                      # same row
    assert conflicts(g) == [(0, 8)]