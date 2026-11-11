# P7: Handwritten Sudoku Reader and Solver (Search + CSP + Neural Network)

Pipeline: handwritten mosaic -> CNN reads digits (MNIST) -> Search (UCS/Greedy/A*) finds the cheapest correction set (max 3) -> CSP (BT, FC, AC-3) checks/solves -> correction log. If nothing works: "unreadable" plus the least confident cells.

## Setup (Windows PowerShell, Python 3.14.4)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run (from the project root, in this order)
    python make_data.py --seed 42 --n 20
    python -m nn.run_train
    python -m nn.run_calibrate
    python -m pytest -q
    python -m eval.exp1 --per-seed 20
    python -m eval.exp2 --per-seed 10
    python -m eval.exp3 --per-seed 20
    python run_demo.py

Results (CSV + PNG) are written to results/.

## Notes
- Misreads are injected with a documented rule; this is not photographed Sudoku.
- "Solvable" is not "unique": the cheapest solvable fix can be a different valid puzzle.
- See DATA_DICTIONARY.md for real vs generated data.
