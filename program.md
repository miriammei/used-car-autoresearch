# AutoResearch Program: Used Car Price Prediction

## Objective
Minimize RMSE of the used car price model.

## Rules
1. **Editable File**: You may ONLY modify `model.py`.
2. **Frozen Files**: DO NOT modify `prepare.py` or `run.py`.
3. **Execution**: Run `python run.py` to train and evaluate your model.
4. **Logging**: Results are automatically logged to `results.tsv`.
5. **Constraints**:
   - Each training run should ideally take less than 60 seconds.
   - Use only CPU-based scikit-learn compatible models.
   - No additional data sources or external downloads.


## Workflow

```
1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run:  python run.py "description of change"
5. Check val_rmse in output
6. If improved:  git add model.py && git commit -m "feat: <description>"
7. If worse:     git checkout model.py   (revert)
8. Repeat from step 1
```

## Ideas to explore

- Different regressors: Ridge, Lasso, ElasticNet, SVR
- Ensemble methods: RandomForest, GradientBoosting, HistGradientBoosting
- Feature engineering: PolynomialFeatures, interaction terms
- Preprocessing: RobustScaler, QuantileTransformer
- Target transform: TransformedTargetRegressor with log
- Hyperparameter tuning within the pipeline

## What NOT to do

- Do not modify `prepare.py` (data split, metric)
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of `build_model()`

