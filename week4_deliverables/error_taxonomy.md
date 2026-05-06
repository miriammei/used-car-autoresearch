# Error Taxonomy: Used Car Price Prediction

Analysis of model failures based on historical performance in `results.tsv`.

| Error Class | Symptoms | Root Cause | Frequency |
|------------|----------|------------|-----------|
| **Sparse Signal** | High RMSE, negative R2 | One-hot encoding high-cardinality features (e.g., `Model`) with limited training data (2,000 rows). | High |
| **Over-Transformation** | Significant RMSE spike | Log-transforming a target variable that is not strictly log-normal or has low variance relative to noise. | Phase 2 |
| **Linear Limitations** | RMSE stalled at ~$27k | Simple linear models (Ridge/Lasso) failing to capture non-linear interactions between Mileage, Age, and Brand. | Constant |
| **Outlier Sensitivity** | Inconsistent performance | RMSE (squared error) being dominated by a small number of "luxury" or "classic" car outliers in the test set. | Likely |
| **Feature Noise** | Baseline ~ Constant Mean | Features like `Engine Size` or `Fuel Type` showing near-zero correlation with `Price` in current linear mappings. | High |
