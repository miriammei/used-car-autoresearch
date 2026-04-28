import time
import os
import pandas as pd
from datetime import datetime
from prepare import load_data, evaluate, plot_performance
from model import build_model

def main():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    
    print("Building model...")
    model = build_model(X_train)
    
    print("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - start_time
    print(f"Training complete in {runtime:.4f} seconds.")
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    rmse, r2 = evaluate(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Log results
    results_file = 'results.tsv'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    description = "Linear Regression Baseline" # This could be passed as an argument
    
    new_result = {
        'Timestamp': timestamp,
        'RMSE': rmse,
        'R2': r2,
        'Runtime': runtime,
        'Description': description
    }
    
    if not os.path.exists(results_file):
        df = pd.DataFrame([new_result])
        df.to_csv(results_file, sep='\t', index=False)
    else:
        df = pd.read_csv(results_file, sep='\t')
        df = pd.concat([df, pd.DataFrame([new_result])], ignore_index=True)
        df.to_csv(results_file, sep='\t', index=False)
    
    print(f"Results logged to {results_file}")
    
    # Update plot
    plot_performance()
    print("Performance plot updated.")

if __name__ == "__main__":
    main()
