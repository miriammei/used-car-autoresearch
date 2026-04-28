# AutoResearch Log: Used Car Price Prediction

## Research Goal
Achieve an RMSE < $5,263.80 (10% of the average price of $52,638.02).

---

## Phase 1: Baseline Benchmarking
**Status**: Completed
**Key Finding**: `GradientBoosting` is the strongest baseline performer with default settings, but all models are currently far from the target RMSE.

### Experiments Summary
- **Linear Regression**: RMSE 27,799.87. Standard baseline.
- **RandomForest**: RMSE 28,083.88. Surprisingly worse than Linear Regression with default settings.
- **GradientBoosting**: RMSE 27,736.39. Best current performance.
- **XGBoost**: RMSE 31,578.52. Significantly underperformed, likely needs tuning.

---

## Phase 2: Hypothesis & Strategy
**Current Hypothesis**: 
1. The `Year` column is being treated as a raw numeric feature, but price-age relationships are often non-linear (e.g., depreciation curves).
2. `Brand` and `Model` have high cardinality; one-hot encoding might be creating a very sparse and high-dimensional space that simple models struggle with.
3. Feature scaling might not be enough for models like XGBoost/GBR if the target distribution is skewed.

**Next Steps**:
1. [ ] **Feature Engineering**: Convert `Year` to `Car_Age`.
2. [ ] **Target Transformation**: Try predicting `log(Price)` to handle potential skewness.
3. [ ] **Encoding**: Explore Target Encoding or Ordinal Encoding for `Brand`/`Model`.
4. [ ] **Hyperparameter Tuning**: Focus on tuning the `GradientBoosting` and `XGBoost` models.
