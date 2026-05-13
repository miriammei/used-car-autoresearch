# Failure Analysis Memo: Week 4 (Post-Experiment Update)

**To**: Research Lead  
**From**: Miriam Mei  
**Date**: May 6, 2026  
**Subject**: Evaluation of Week 4 Controlled Experiments

## 1. Executive Summary
The Week 4 experiments successfully broke the ~$27,700$ RMSE floor, achieving a new best of **$27,439.67$** via Ordinal Encoding of the `Condition` feature and Target Encoding of high-cardinality categorical variables. This represents a total improvement of ~$350$ over the previous phase.

## 2. Key Findings
1. **Ordinal Mapping**: Treating `Condition` as an ordinal feature (`Used` < `Like New` < `New`) provided the most significant boost (EXP-02). This confirms that the model benefits from explicit domain knowledge about value decay.
2. **Information Density**: Target Encoding (EXP-01) outperformed One-Hot Encoding, likely by reducing the noise introduced by sparse high-cardinality features (`Model`).
3. **Robustness Plateaus**: Both Robust Scaling (EXP-04) and Huber Regression (EXP-05) showed success in improving over the baseline but did not exceed the performance of simple Ridge with improved encoding. This suggests that while outliers exist, they are not the primary bottleneck.

## 3. Persistent Challenges
Despite these improvements, the RMSE remains far from the target of $5,263.80$. The R2 scores are near zero (0.006), indicating that the features provided still explain less than 1% of the variance in `Price`. 

## 4. Final Conclusion
The current dataset appears to have extremely high noise or the target variable `Price` is essentially random relative to the provided features (`Mileage`, `Engine Size`, etc.). Further progress likely requires external data or a radical change in feature generation.
