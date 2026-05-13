# Week 5 Summary: Keep / Discard / Crash

This document summarizes the strategic decisions for the modeling pipeline based on Week 4 experimental results.

## ✅ Keep (Integrate into Main Pipeline)
*   **Ordinal Encoding for `Condition`**: Treat as `Used` < `Like New` < `New`. Provided the single largest reduction in RMSE (~350 points).
*   **Target Encoding for `Model` and `Brand`**: Effectively managed high-cardinality features where One-Hot Encoding introduced too much noise/sparsity.
*   **Miles_Per_Year Interaction**: Captures vehicle usage intensity better than raw mileage.

## ❌ Discard (Ineffective or Counter-productive)
*   **Log-Target Transformation**: Tested in Phase 2; resulted in a significant performance drop (RMSE increased from ~27k to ~29k).
*   **Robust Scaling**: While slightly better than the absolute baseline, it offered no advantage over standard scaling when combined with the best-performing encoding strategies.
*   **One-Hot Encoding for `Model`**: Discarded in favor of Target Encoding due to sparsity issues.

## ⚠️ Crash / Failures (Non-Competitive)
*   **Huber Regression**: Not a technical "crash," but failed to outperform `Ridge` once ordinal features were properly mapped. The robustness to outliers did not compensate for the loss of fit in the central distribution.
*   **XGBoost (Default Settings)**: Consistently underperformed linear models (Ridge/Lasso) on this specific dataset, likely due to small sample size (2,000) and high noise.
