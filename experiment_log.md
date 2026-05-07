# AutoResearch Log: Used Car Price Prediction

## Research Goal
Achieve an RMSE < $5,263.80 (10% of the average price of $52,638.02).

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
**Status**: Completed
**Key Finding**: Information density improvements (Target and Ordinal Encoding) provided the most significant gains, but the model remains far from the target RMSE. Robust scaling and Huber regression did not provide additional benefits over Ridge.

### Experiments Summary
- **EXP-01: Target Encoding**: RMSE 27,555.58.
- **EXP-02: Ordinal Condition**: RMSE 27,439.67 (Best Result).
- **EXP-03: Miles_Per_Year Interaction**: RMSE 27,446.01.
- **EXP-04: Robust Scaling**: RMSE 27,458.92.
- **EXP-05: Huber Regression**: RMSE 27,555.63.
- **Verification Run: Auto-logging test**: RMSE 27474.07
