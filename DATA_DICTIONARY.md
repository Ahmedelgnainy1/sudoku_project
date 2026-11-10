# Data Dictionary

## data/generated/puzzles.csv (made by `python make_data.py --seed 42 --n 20`)

| Column | Type | Real/Generated | How made |
|---|---|---|---|
| puzzle_id | str | generated | `g<givens>_<index>`, e.g. `g30_4` |
| seed | int | generated | `base_seed*1000 + index`, base seed from `--seed` |
| n_givens | int | generated | target: 25, 30 or 35 |
| puzzle | str (81 chars, 0 = empty) | generated | randomised-backtracking full grid, cells removed in random order while the solution stays unique |
| solution | str (81 chars) | generated | the full grid before removal |

All columns are generated. No real Sudoku data is used.

## MNIST rows
To be added in Phase 5.
