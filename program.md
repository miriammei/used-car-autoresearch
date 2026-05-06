# AutoResearch Agent Instructions: Used Car Price Prediction

## Objective
Minimize **RMSE** on the used car price regression task.

## Rules
1. You may **ONLY** modify `model.py`
2. `prepare.py` and `run.py` are **FROZEN** — do not touch them
3. `build_model(X)` must return an sklearn-compatible estimator (Pipeline preferred)
4. Training + evaluation must complete in **under 60 seconds** on CPU
5. No additional data sources or external downloads

## Workflow
```
1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run: python run.py
5. Check RMSE in output
6. Log findings in experiment_log.md
7. Verify entry in results.tsv
8. If improved: keep changes
9. If worse: revert model.py
10. Repeat from step 1
```

## Ideas to explore
- Different regressors: Ridge, Lasso, RandomForest, GradientBoosting, XGBoost
- Feature engineering: Date extraction from 'Year', Brand encoding, mileage bins
- Preprocessing: RobustScaler, PowerTransformer
- Hyperparameter tuning within the pipeline

## What NOT to do
- Do not modify `prepare.py` (data loading, evaluation logic)
- Do not modify `run.py` (orchestration)
- Do not add new files or external dependencies without updating requirements.txt
- Do not hard-code test data into the model
- Do not change the function signature of `build_model(X)`
- Do not forget to log all experimental outcomes in `experiment_log.md` and `results.tsv`
