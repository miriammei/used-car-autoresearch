# AutoResearch Log: Used Car Price Prediction

## Research Goal
Achieve an RMSE < ,263.80 (10% of the average price of 2,638.02).

---

## Phase 1: Baseline Benchmarking
**Status**: Completed
**Key Finding**: `GradientBoosting` is the strongest baseline performer with default settings.

### Experiments Summary
- **Ridge**: RMSE 27,794.24.
- **GradientBoosting**: RMSE 27,736.39. Best current performance.

---

## Phase 2: Feature Engineering & Target Transformation
**Status**: Completed
**Key Finding**: Log-transforming the target variable significantly worsened performance across all models.

### Experiments Summary
- **Ridge (Log-Target)**: RMSE 29,542.27.
- **GradientBoosting (Log-Target)**: RMSE 29,664.70.

---

## Phase 3: Refining Feature Engineering (No Log-Transform)
**Status**: Completed
**Key Finding**: Removing the log-transform restored performance. `Ridge` showed a minor improvement with `Car_Age` and dropping `Car ID`. However, overall performance remains far from the target.

### Experiments Summary
- **Ridge (Age Only)**: RMSE 27,788.75. (Slight improvement over baseline)
- **GradientBoosting (Age Only)**: RMSE 28,024.78.

---

## Phase 4: Strategy for Improvement
**Current Hypothesis**: 
1. The model is struggling with high-cardinality features (`Brand`, `Model`). One-hot encoding is creating a sparse matrix that might be overfitting or failing to capture relationships.
2. `Condition` is ordinal but being treated as nominal.

**Next Steps**:
1. [ ] **Target Encoding**: Use Target Encoding for `Model` and `Brand`.
2. [ ] **Ordinal Encoding**: Apply `OrdinalEncoder` to `Condition` with a defined order.
3. [ ] **Feature Selection**: Evaluate if some features are purely noise.
