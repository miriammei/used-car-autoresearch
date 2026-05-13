# "What Actually Worked" Memo

**To**: Research Team  
**From**: Gemini CLI (Autonomous Agent)  
**Date**: May 12, 2026  
**Subject**: Post-Mortem on Week 4 Wins and Path Forward

## 1. What Actually Worked
The primary "win" of Week 4 was the shift from **architectural complexity** to **information density**.

*   **Mapping Domain Knowledge**: The most significant gain came from manually mapping the `Condition` column to an ordinal scale. This suggests that the relationship between condition and price is strongly monotonic and that the model previously struggled to "learn" this from dummy variables.
*   **Managing Sparsity**: Replacing One-Hot Encoding with Target Encoding for the `Model` feature allowed the Ridge regressor to leverage the average price of specific car models without exploding the feature space.
*   **Linear Regularization**: Simple `Ridge` regression consistently outperformed more complex models like `XGBoost` and `RandomForest`. In a high-noise, low-signal environment, the bias of a linear model acts as a necessary regularizer.

## 2. What Failed to Move the Needle
*   **Outlier Mitigation**: Neither Robust Scaling nor Huber Regression provided a competitive advantage. This indicates that the high RMSE is not driven by a few "bad" data points, but by a general lack of correlation across the entire dataset.
*   **Non-Linearity**: Boosting and bagging models failed to find non-linear patterns, reinforcing the hypothesis that the provided features (`Mileage`, `Year`, `Engine Size`) are only weakly coupled to the `Price` in this specific data generation process.

## 3. Strategic Recommendations
We have reached the point of diminishing returns for model tuning and basic feature engineering. 

**Recommendation for Week 5**: 
1.  **Stop** hyperparameter optimization.
2.  **Focus** on feature synthesis (e.g., brand-tier clustering, engine-to-weight ratios if data available).
3.  **Investigate** the data source; the R2 of 0.006 suggests that the target variable `Price` may contain significant random noise or is missing a "silver bullet" feature (like `Original MSRP` or `Location`).
