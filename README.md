# Used Car Price Prediction - Baseline

This project provides a baseline machine learning pipeline to predict used car prices using Linear Regression.

## Project Goal
The primary objective is to achieve a model performance where the **Root Mean Squared Error (RMSE) is less than 10% of the average car price** in the dataset.

- **Current Average Price:** $52,638.02
- **Target RMSE:** < $5,263.80
- **Status:** The baseline is currently at $27,799.87 (~52.8% of average).

## Setup Instructions
...
1. **Install Dependencies**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset**
   Download the used car dataset from [Kaggle](https://www.kaggle.com/datasets/ayeshasiddiqa123/cars-pre?utm_source=chatgpt.com).
   Save it as `car_price_prediction_.csv` (or your preferred name) in this directory.

3. **Run the Baseline Model**
   Run the script by specifying your data file and the target column name:
   ```bash
   python baseline.py --data car_data.csv --target price
   ```
   *Note: The script will automatically split the data and save `test.csv` to lock the test set for future iterations.*

## Baseline Experiment Results

The first baseline run was conducted to establish a performance benchmark.

| Metric | Value |
|--------|-------|
| Model  | Linear Regression |
| RMSE   | 27799.8747 |
| R²     | -0.0202 |
| Runtime| 0.0106s |

*Detailed logs can be found in `experiments.md`.*
