# 🧩 Advanced Sudoku Reader and Solver

An AI-powered Handwritten Sudoku Reader and Solver that combines **Neural Networks, Search Algorithms, and Constraint Satisfaction Problems (CSP)** to recognize, validate, correct, and solve Sudoku puzzles.

> **Project Status:** 🚧 Under Development

## 📌 Project Overview

This project aims to build an intelligent system capable of solving handwritten Sudoku puzzles by combining multiple AI techniques.

Instead of simply solving a given Sudoku grid, the system will recognize handwritten digits, estimate recognition confidence, detect inconsistencies, and intelligently correct recognition errors before producing the final solution.

### 🎯 Main Objectives

- Recognize handwritten Sudoku digits using a trained Neural Network.
- Estimate prediction probabilities for each cell.
- Detect conflicts and inconsistencies in the recognized grid.
- Apply Search algorithms to identify the most likely recognition corrections.
- Solve the corrected puzzle using CSP techniques.
- Compare different AI algorithms through measurable experiments.

## 🏗️ System Architecture

The planned system consists of four main components:

### 1. Neural Network (NN)
Responsible for handwritten digit recognition.

- MNIST-based digit classification.
- Probability-based predictions.
- Recognition accuracy evaluation.
- Confidence scores for individual cells.

### 2. Search Algorithms
Responsible for finding the most likely corrections when the recognized Sudoku contains errors.

- Uniform Cost Search (UCS).
- Greedy Best-First Search.
- A* Search.
- Probability-based correction costs.

### 3. Constraint Satisfaction Problems (CSP)
Responsible for validating and solving Sudoku puzzles.

- Backtracking Search.
- Forward Checking.
- AC-3 (Arc Consistency).
- Variable and value ordering heuristics.

### 4. Integration
Connects all components into one complete pipeline.

**Planned execution flow:**

```text
Handwritten Sudoku Image
          |
          v
   Neural Network
          |
          v
  Digit Predictions
  + Confidence Scores
          |
          v
    CSP Validation
          |
          v
  Search-Based Correction
          |
          v
    CSP Sudoku Solver
          |
          v
    Final Solution
```

## 📂 Project Structure

```text
sudoku_project/
│
├── csp/                 # CSP algorithms and Sudoku solver
├── data/                # Generated datasets and data files
├── eval/                # Experiments and evaluation
├── integration/         # Integration of NN, Search, and CSP
├── nn/                  # Neural Network implementation
├── search/              # Search algorithms
├── tests/               # Automated tests
│
├── make_data.py         # Reproducible dataset generation
├── DATA_DICTIONARY.md   # Dataset documentation
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore
```

## 🛠️ Technologies

- **Language:** Python
- **Machine Learning:** Neural Networks
- **Dataset:** MNIST
- **AI Techniques:** Search Algorithms and CSP
- **Testing:** Automated Python tests
- **Version Control:** Git and GitHub

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later (depending on dependency compatibility).
- Git.
- pip.

### 1. Clone the repository

```bash
git clone https://github.com/Ahmedelgnainy1/sudoku_project.git
```

```bash
cd sudoku_project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python make_data.py
```

Check `make_data.py` for available generation options and configuration.

## 📊 Evaluation and Experiments

An important part of this project is evaluating the effectiveness of different AI techniques.

Planned comparisons include:

| Component | Approaches |
|---|---|
| Search | UCS vs Greedy vs A* |
| CSP | Backtracking vs Forward Checking vs AC-3 |
| Recognition Correction | NN-guided vs No Correction vs Random Correction |
| Heuristics | Baseline vs Advanced Heuristics |

### Evaluation Metrics

- Recognition Accuracy.
- Correction Success Rate.
- Exact-Solution Recovery Rate.
- Number of Expanded Search Nodes.
- Number of Backtracks.
- Number of Assignments.
- Execution Time.

Experimental results will be documented and compared to demonstrate the contribution of each technique.

## 📖 Documentation

- [Data Dictionary](DATA_DICTIONARY.md) — Dataset structure, fields, and data definitions.

Additional documentation, implementation details, and experimental results will be added as development progresses.

## 🗺️ Development Progress

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Project skeleton and grid utilities | ✅ Completed |
| Phase 2–3 | CSP implementation and improvements | 🚧 In Progress |
| Phase 4 | Seeded data generator and data dictionary | ✅ Completed |
| Phase 5 | Search algorithms | ⏳ Planned |
| Phase 6 | Neural Network implementation | ⏳ Planned |
| Phase 7 | System integration | ⏳ Planned |
| Phase 8 | Experiments and evaluation | ⏳ Planned |
| Phase 9 | Final demo and documentation | ⏳ Planned |

*Progress is updated as new components are implemented and tested.*

## 👨‍💻 Author

**Ahmed Elgnainy**

Computer Science Student | Egypt-Japan University of Science and Technology (E-JUST)

- GitHub: [@Ahmedelgnainy1](https://github.com/Ahmedelgnainy1)
- Project Repository: [sudoku_project](https://github.com/Ahmedelgnainy1/sudoku_project)

## 📄 License

This project is developed for academic purposes as part of an Artificial Intelligence course project.
