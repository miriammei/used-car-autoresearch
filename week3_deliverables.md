# Week 3 Deliverables: AutoResearch Agent Reflection

## Project Context & Navigation
- **Instructional Context:** [`program.md`](program.md) (Located in project root)
- **Research Journal:** [`experiment_log.md`](experiment_log.md) (Located in project root)
- **Detailed Metrics:** [`results.tsv`](results.tsv) (Tab-separated logs)

## Key Experiments Conducted
1. **Linear Regression Baseline:** Established initial benchmark with standard scaling and one-hot encoding. (**RMSE: 27,799.87**)
2. **RandomForest Transition:** Attempted switching to an ensemble method with default settings. (**RMSE: 28,083.88**)
3. **Competitive Model Selection (Grid Search):** Evaluated Ridge, Lasso, RF, GBR, and XGBoost in a single run to identify the most promising architecture. (**Winner: GradientBoosting**)
4. **Baseline Individual Benchmarking:** Ran a clean comparison of all five models without search-induced bias to confirm default performance. (**Best: GradientBoosting, RMSE: 27,736.39**)

---

## Agent Reflection

### What went well

- **Architectural Restructuring:** I successfully converted a flat repository into a structured "AutoResearch" template (Frozen vs. Editable) using the `demo-autoresearch` repository as a template.
- **Iterative Logging:** I refined the logging system multiple times based on user feedback—moving from simple hardcoded strings to dynamic CLI arguments, and eventually to a grouped "best-per-model" summary for grid searches.
- **Dependency Management:** I correctly identified and integrated new libraries like `xgboost` into the workflow when expanding the search space.

### What went badly

- **Speculative Implementation:** I initially implemented a Random Forest model that performed worse than the baseline because I prioritized switching the algorithm over analyzing the feature set (e.g., handling the `Year` column or high-cardinality categorical variables).
- **Information Density:** My first attempt at logging `GridSearchCV` results recorded all 18 candidates, which made the `results.tsv` file difficult to read and required a manual cleanup turn.
- **Baseline Assumption:** I assumed newer/more complex models (XGBoost) would naturally outperform Linear Regression with default settings, which proved incorrect for this specific dataset.

---

## Common Failure Modes
- **The "Replace" String Mismatch:** Subtle differences in indentation or trailing whitespace frequently cause the `replace` tool to fail. *Mitigation:* Always performing a surgical `read_file` immediately before a `replace`.
- **Hyperparameter Blindness:** Relying on default settings for powerful models like XGBoost can lead to significantly worse performance than simpler, well-scaled linear models.
- **Categorical sparse-space explosion:** Using One-Hot Encoding on high-cardinality features (like `Model`) creates a high-dimensional sparse matrix that can slow down tree-based models and degrade linear model performance without regularization.
- **Convergence Errors:** Linear models (Lasso/Ridge) often fail to converge on this dataset's scale, indicating a need for higher iterations or better feature scaling that was initially overlooked.
- **Log Bloat:** Automated search loops generate a high volume of data; failing to aggregate or summarize these results early leads to "context pollution" in the project artifacts.
