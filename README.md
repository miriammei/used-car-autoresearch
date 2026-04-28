# Used Car Price AutoResearch

This repository is set up for automated, iterative machine learning research to predict used car prices. It follows a specialized structure designed for AI agents to improve model performance autonomously.

## Project Goal
The primary objective is to minimize **RMSE**.
- **Current Best RMSE**: $27,736.39 (GradientBoosting)
- **Target RMSE**: < $5,263.80 (10% of average price)
- **Gap to Target**: $22,472.59

## Baseline Comparison

| Model | RMSE | R² | Description |
|-------|------|----|-------------|
| **GradientBoosting** | **27,736.39** | -0.0155 | Best baseline performer with default settings. |
| Ridge | 27,794.24 | -0.0198 | Standard linear model with L2 regularization. |
| Lasso | 27,798.16 | -0.0200 | Standard linear model with L1 regularization. |
| Linear Regression | 27,799.87 | -0.0202 | Initial baseline benchmark. |
| RandomForest | 28,083.88 | -0.0411 | Ensemble tree-based model (100 estimators). |
| XGBoost | 31,578.52 | -0.3163 | Default XGBoost regressor (performed poorly). |

*Detailed logs can be found in `results.tsv`.*

## Project Structure

- `prepare.py`: **FROZEN**. Handles data loading and evaluation metrics.
- `model.py`: **EDITABLE**. Contains the model pipeline definition.
- `run.py`: **FROZEN**. Orchestrates training, evaluation, and logging.
- `program.md`: Instructions and constraints for the AutoResearch agent.
- `GEMINI.md`: Contextual metadata for the Gemini CLI agent.
- `results.tsv`: Log of all experiments.
- `performance.png`: Visualization of performance over time.

## Setup and Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Experiment**:
   To execute a baseline model:
   ```bash
   BASE_MODEL=GradientBoosting python run.py "Baseline GradientBoosting"
   ```
   *Available BASE_MODEL options: Ridge, Lasso, XGBoost, RandomForest, GradientBoosting*

3. **Iterate**:
   Modify `model.py` to improve the pipeline, then run `python run.py` again.
