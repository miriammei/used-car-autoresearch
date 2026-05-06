# Experiment Result Matrix

| Experiment ID | Description | Model | Baseline RMSE | Observed RMSE | Delta | Outcome |
|---------------|-------------|-------|---------------|---------------|-------|---------|
| EXP-01 | Target Encoding (Model/Brand) | Ridge | 27788.75 | 27555.58 | -233.17 | Success |
| EXP-02 | Ordinal Encoding (Condition) | Ridge | 27788.75 | 27439.67 | -349.08 | Best Result |
| EXP-03 | Feature Interaction (Miles/Year) | Ridge | 27788.75 | 27446.01 | -342.74 | Marginal |
| EXP-04 | Robust Scaling | Ridge | 27788.75 | 27458.92 | -329.83 | Marginal |
| EXP-05 | Huber Regression | Huber | 27788.75 | 27555.63 | -233.12 | Success |

*Note: Baseline RMSE is from Phase 3 (Ridge with Car_Age).*
