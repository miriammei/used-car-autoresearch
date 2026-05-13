# Controlled Experiment Set: Used Car Price Prediction

This document outlines the systematic experiments planned for Week 4 to minimize RMSE.

## 1. Encoding Strategy Experiment
- **Objective**: Compare One-Hot Encoding (OHE) vs. Target Encoding for high-cardinality features.
- **Hypothesis**: `Model` and `Brand` have 28 and 7 unique values respectively. OHE might be too sparse for 2,000 samples. Target encoding will provide a denser signal.
- **Implementation**: Replace OHE with `TargetEncoder` for categorical columns in the `preprocessor`.

## 2. Ordinal Condition Experiment
- **Objective**: Evaluate the impact of treating `Condition` as an ordinal variable.
- **Hypothesis**: The relationship between 'Used', 'Like New', and 'New' is monotonic. Ordinal encoding captures this trend better than independent dummy variables.
- **Implementation**: Use `OrdinalEncoder` with specific mapping for the `Condition` column.

## 3. Feature Interaction: Mileage Density
- **Objective**: Capture the 'wear and tear' intensity.
- **Hypothesis**: Mileage relative to age (`Miles_Per_Year`) is a better indicator of value than absolute mileage.
- **Implementation**: Add `Miles_Per_Year = Mileage / (Car_Age + 1)` in `extract_features`.

## 4. Robust Scaling Experiment
- **Objective**: Reduce the influence of price outliers and skewed numeric distributions.
- **Hypothesis**: `StandardScaler` is sensitive to outliers. `RobustScaler` will lead to a more stable fit.
- **Implementation**: Swap `StandardScaler` for `RobustScaler` in the numeric pipeline.

## 5. Robust Regression (Huber Loss)
- **Objective**: Use a loss function less sensitive to large residuals.
- **Hypothesis**: RMSE is heavily penalized by outliers. `HuberRegressor` minimizes squared loss for small residuals and absolute loss for large ones, potentially improving generalization.
- **Implementation**: Change `BASE_MODEL` to use `HuberRegressor`.
