import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_data():
    """Loads the train and test data."""
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        raise FileNotFoundError("train.csv and test.csv must exist.")
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    target = 'Price'
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    
    return X_train, y_train, X_test, y_test

def evaluate(y_true, y_pred):
    """Calculates RMSE and R2."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def plot_performance(results_file='results.tsv', output_file='performance.png'):
    """Plots the performance history from results.tsv."""
    if not os.path.exists(results_file):
        return
    
    try:
        df = pd.read_csv(results_file, sep='\t')
        if df.empty:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['RMSE'], marker='o', label='RMSE')
        plt.title('Model Performance History (RMSE)')
        plt.xlabel('Experiment Index')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_file)
        plt.close()
    except Exception as e:
        print(f"Error plotting performance: {e}")
