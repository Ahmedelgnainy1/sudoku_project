# P7: Handwritten Sudoku Reader and Solver (Search + CSP + Neural Network)

Pipeline: picture or handwritten mosaic -> CNN reads digits -> Search (UCS/Greedy/A*) finds the cheapest correction set (max 3) -> CSP (BT, FC, AC-3) checks and solves -> correction log. If nothing works: "unreadable" plus the least confident cells.

## 1. Setup (Windows PowerShell, Python 3.14.4)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Always run commands from the project root (the folder that contains `models/`).

## 2. Solve your own Sudoku picture (extra, beyond the project PDF)
    python solve_image.py path\to\picture.png
    python solve_image.py path\to\picture.png --debug      # saves outputs/debug_tiles.png (the 81 tiles the CNN sees)
    python solve_image.py path\to\picture.png --save       # saves outputs/solution.png (black = read, blue = solved)

It prints each step: load and trim, cut cells and remove grid lines, CNN reading (* = confidence below 0.90), conflict check and AC-3, A* correction search, corrections made, CSP solution.

Picture requirements: straight and cropped Sudoku, printed or handwritten, one digit per cell, no text around the grid. Tilted photos are not supported.

Model used: `models/cnn_print.pt` if it exists, otherwise `models/cnn.pt`.

Things to remember:
- Check the "read grid" first. If a digit was misread, the answer is for a different puzzle: the search checks "solvable", not "matches the picture".
- With few clues (under 25) many solutions exist; one valid solution is returned.
- Search can change a digit into another digit, but cannot fix a blank read as a digit or a digit read as blank.

## 3. Reproduce the project (in this order)
    python make_data.py --seed 42 --n 20      # puzzles + MNIST + checksums
    python -m nn.run_train                    # logistic baseline + CNN -> models/cnn.pt
    python -m nn.run_calibrate                # accuracy, ECE, NLL, confusion matrix
    python -m vision.train_print              # CNN + printed digits -> models/cnn_print.pt (for solve_image.py)
    python -m pytest -q                       # all tests
    python -m eval.exp1 --per-seed 20         # UCS vs Greedy vs A*
    python -m eval.exp2 --per-seed 10         # BT vs BT+FC vs BT+FC+AC-3
    python -m eval.exp3 --per-seed 20         # NN-guided vs none vs random
    python run_demo.py                        # MNIST demo (also: --misreads 4 --seed 2)

Results (CSV + PNG) are written to results/. Experiments 1-3 use `models/cnn.pt`.

## 4. Folders
- csp/: grid, solver (BT, FC, AC-3, MRV, degree switches, counters), ac3, generator
- data/: MNIST loader. nn/: baseline, CNN, calibration. search/: correction search
- integration/: mosaic, reader, pipeline. eval/: experiments. tests/: pytest
- vision/: picture to tiles (extract.py), printed-digit training data (printed.py, train_print.py)

## 5. Notes and limits
- Misreads in the experiments are injected with a documented rule; the MNIST data is not photographed Sudoku.
- "Solvable" is not "unique": the cheapest solvable fix can be a different valid puzzle.
- vision/ and solve_image.py use external techniques (connected-component blob extraction with scipy, generated printed-digit tiles) and were only tested on a few clean pictures.
- See DATA_DICTIONARY.md for real vs generated data.

See examples/ for an example input picture and its solved output.

