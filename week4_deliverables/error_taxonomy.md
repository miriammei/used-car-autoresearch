# Error Taxonomy: Week 4 Controlled Experiment Loop

Analysis of model failures and performance plateaus observed during experiments EXP-01 through EXP-05.

| Error Class | Symptoms | Root Cause | Frequency |
|------------|----------|------------|-----------|
| **Interaction Noise** | Slight RMSE increase (+6.34) after adding `Miles_Per_Year`. | Adding interaction features that introduce more variance/noise than predictive signal for linear models. | EXP-03 |
| **Scaling Drift** | RMSE increase (+19.25) when switching to `RobustScaler`. | `RobustScaler` desensitizing the model to signals in the tails of the distribution that are actually predictive of price. | EXP-04 |
| **Loss Function Mismatch** | Significant RMSE spike (+115.96) with `HuberRegressor`. | Using a robust loss function (Huber) in a dataset where the squared error (Ridge) better captures the global price trends. | EXP-05 |
| **Model Saturation** | RMSE plateauing at ~$27,400$ regardless of feature tweaks. | A fundamental lack of predictive signal in the provided features, where architectural changes yield diminishing returns. | EXP-02 to EXP-05 |
| **Ordinal Over-simplification** | RMSE floor at $27,439$ (Best but insufficient). | While ordinal mapping helped, the linear decay assumption may still be too simple to reach the $5,263$ target. | EXP-02 |

## Observed Performance Dynamics
The Week 4 loop demonstrated that **Information Density** (Encoding) provided the only real gain. Subsequent attempts to add **Robustness** (RobustScaler, Huber) or **Complexity** (Interaction Terms) actually degraded performance, suggesting the model is extremely sensitive to noise and that the current linear approach has been fully exhausted.
