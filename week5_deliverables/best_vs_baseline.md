# Best Result vs. Baseline Report

This report compares the best-performing model from the Week 4 experiments against the project baselines.

## Performance Comparison

| Model | RMSE | R2 Score | Delta (RMSE) |
|-------|------|----------|--------------|
| **Initial Baseline (LinReg)** | 27,799.87 | -0.020 | 0.00 |
| **Strong Baseline (GradBoost)** | 27,736.39 | -0.015 | -63.48 |
| **Best Result (EXP-02: Ridge + Ordinal)** | **27,439.67** | **0.006** | **-360.20** |

## Key Metrics
*   **Best Model**: Ridge Regression with Ordinal Encoding (`Condition`) and Target Encoding (`Model`, `Brand`).
*   **Total Improvement**: 1.30% reduction in RMSE compared to the initial baseline.
*   **Status**: Target RMSE ($5,263.80) has NOT been met.

## Analysis
The "Best Result" represents the first time the model has achieved a positive R2 score (0.006). While statistically an improvement, the model still explains less than 1% of the variance in used car prices. This confirms that the current feature set is insufficient for high-accuracy prediction.
