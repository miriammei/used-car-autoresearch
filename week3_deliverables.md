# Week 3 Deliverables: AutoResearch Agent Reflection

## Project Context & Navigation

- First version of [`program.md`](program.md)
- Experiment log: [`experiment_log.md`](experiment_log.md)
- Logged results: [`results.tsv`](results.tsv)

## Key Experiments Conducted

1. **Linear Regression Baseline:** Established initial benchmark with standard scaling and one-hot encoding. (**RMSE: 27,799.87**)
2. **RandomForest Transition:** Attempted switching to an ensemble method with default settings. (**RMSE: 28,083.88**)
3. **Competitive Model Selection (Grid Search):** Evaluated Ridge, Lasso, RF, GBR, and XGBoost in a single run to identify the most promising architecture. (**Winner: GradientBoosting**)
4. **Baseline Individual Benchmarking:** Ran a clean comparison of all five models without search-induced bias to confirm default performance. (**Best: GradientBoosting, RMSE: 27,736.39**)

---

## Agent Reflection

### What went well

I was able to successfully convert a flat repository into a structured "AutoResearch" template (Frozen vs. Editable) using the `demo-autoresearch` repository as a template. The agent is also able to adapt easily to new requirements and experiments when specified and integrates these elements seamlessly into the workflow while also logging them efficiently. Additionally, it was really easy to specify exactly which files I wanted to update and how I wanted the models to be run based on my prompts.

### What went badly

The newer models did not perform as well as I thought they would, but it makes sense since they might not be using optimal hyperparameters and features. I assumed that more complex models like XGBoost would naturally outperform Linear Regression with default settings, which proved incorrect for this specific dataset. Not only that, the Random Forest model also performed worse than the baseline because I prioritized switching the algorithm over analyzing the feature set (like handling the `Year` column or high-cardinality categorical variables). Aside from poor model performance, my first attempt at logging the experiments resulted in results that were difficult to read since it recorded all the possible `GridSearchCV` candidates. Once I informed the agent to only use the baseline Ridge, Lasso, XGBoost, etc. models, it was able to log those instead. 

---

## Common Failure Modes

- **Hyperparameter Blindness:** Relying on default settings for powerful models like XGBoost can lead to significantly worse performance than simpler, well-scaled linear models.
- **Computational Resources:** Using One-Hot Encoding on categorical features with lots of categories (like `Model`) creates a high-dimensional sparse matrix that can slow down tree-based models and degrade linear model performance without regularization - and subsequently drain computational resources.
- **Convergence Errors:** Linear models (Lasso/Ridge) often fail to converge on this dataset's scale, indicating a need for higher iterations or better feature scaling that was initially overlooked.
- **Overload of Data:** Automated search loops generate a high volume of data so failing to aggregate or summarize these results early leads to an overload of context in the project artifacts.
