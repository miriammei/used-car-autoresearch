# GEMINI.md - Used Car AutoResearch

## Project Overview
This project is an automated research environment for predicting used car prices. It is designed to allow for rapid, iterative experimentation on machine learning models while keeping the data preparation and evaluation logic "frozen" to ensure consistent benchmarks.

### Architecture
The project follows the **AutoResearch Template**:
- **Frozen (Ground Truth)**: `prepare.py` (data loading/eval) and `run.py` (orchestration).
- **Editable (Sandbox)**: `model.py` (model architecture and preprocessing).
- **Configuration**: `program.md` contains the rules and objectives for the research agent.
- **Artifacts**: `results.tsv` (history) and `performance.png` (visualization).

### Technologies
- **Language**: Python 3.x
- **Libraries**: scikit-learn, pandas, numpy, matplotlib

## Building and Running

### Environment Setup
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### Execution
To run a single experiment (train, evaluate, and log):
```bash
python run.py
```

### Evaluation
The project uses the following datasets:
- `train.csv`: Used for model training.
- `test.csv`: Locked test set for final evaluation.

## Development Conventions

### Model Implementation
- All model-related code must reside in `model.py`.
- The `build_model(X)` function must be implemented to return a scikit-learn compatible estimator or a `Pipeline`.
- The function receives the training feature set `X` to allow for dynamic feature discovery (e.g., identifying numeric vs. categorical columns).

### Workflow Rules
- **DO NOT** modify `prepare.py` or `run.py`. These files define the "rules of the game" and the evaluation metric (RMSE).
- New experiments are automatically appended to `results.tsv`.
- A performance plot `performance.png` is generated/updated after each run.

### Target Performance
- **Objective**: Minimize RMSE.
- **Target**: < $5,263.80 (10% of average price).
