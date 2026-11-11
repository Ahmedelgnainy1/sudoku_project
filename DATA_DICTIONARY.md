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

## MNIST (downloaded by make_data.py, git-ignored)
| Item | Real/Generated | How made |
|---|---|---|
| image 28x28 | real | MNIST pixels / 255 |
| label 1-9 | real | MNIST label, digit 0 dropped |
| train/val split | generated | seeded shuffle, 45,064 / 9,013 |
| test set | real | MNIST test, 9,020 images, untouched |

## Mosaic (built in memory)
| Item | Real/Generated | How made |
|---|---|---|
| mosaic 252x252 | generated | 9x9 tiles of test images |
| blank cell | generated | zero tile, found by rule max pixel < 0.1 |
| hard pool | generated | test images the CNN misreads but whose true digit is its 2nd choice |
| injected misreads | generated | m given cells use hard images, others use correctly read images |
