import pandas as pd
import numpy as np
import argparse
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

def log_experiment(model_name, rmse, r2, runtime):
    log_file = 'experiments.md'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if file exists to add header
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("# Experiment Log\n\n")
            f.write("| Timestamp | Model | RMSE | R² | Runtime (s) |\n")
            f.write("|-----------|-------|------|----|-------------|\n")
            
    with open(log_file, 'a') as f:
        f.write(f"| {timestamp} | {model_name} | {rmse:.4f} | {r2:.4f} | {runtime:.4f} |\n")

def main():
    parser = argparse.ArgumentParser(description='Baseline Used Car Price Prediction')
    parser.add_argument('--data', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--target', type=str, default='price', help='Target column name (default: price)')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Ensure target column exists
    if args.target not in df.columns:
        # Check for case variations
        possible_targets = [c for c in df.columns if c.lower() == args.target.lower()]
        if possible_targets:
            args.target = possible_targets[0]
            print(f"Target column not found exactly, using '{args.target}' instead.")
        else:
            raise ValueError(f"Target column '{args.target}' not found in dataset. Columns: {list(df.columns)}")

    # Lock the Test Set
    print("Splitting data into train and test sets (locking test.csv)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target]
    X_test = test_df.drop(columns=[args.target])
    y_test = test_df[args.target]

    # Preprocessing
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train and Measure Runtime
    print("Training Linear Regression model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - start_time
    print(f"Training complete in {runtime:.4f} seconds.")

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nEvaluation Metrics (on locked test.csv):")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Log Experiment
    log_experiment("Linear Regression Baseline", rmse, r2, runtime)
    print("\nExperiment logged to experiments.md")

if __name__ == "__main__":
    main()
