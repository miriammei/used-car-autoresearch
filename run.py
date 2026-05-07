import time
import os
import pandas as pd
import argparse
import numpy as np
import re
from datetime import datetime
from prepare import load_data, evaluate, plot_performance
from model import build_model

def update_experiment_log(results_to_log):
    log_file = 'experiment_log.md'
    if not os.path.exists(log_file):
        return
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        marker = "### Experiments Summary"
        if marker in content:
            lines = content.split('\n')
            last_marker_idx = -1
            for i, line in enumerate(lines):
                if marker in line:
                    last_marker_idx = i
            
            if last_marker_idx != -1:
                # Find the end of the current list following the marker
                insert_idx = last_marker_idx + 1
                while insert_idx < len(lines) and (lines[insert_idx].strip().startswith('-') or lines[insert_idx].strip() == ''):
                    insert_idx += 1
                
                # Back up over trailing empty lines
                while insert_idx > last_marker_idx + 1 and lines[insert_idx-1].strip() == '':
                    insert_idx -= 1

                for res in results_to_log:
                    lines.insert(insert_idx, f"- **{res['Description']}**: RMSE {res['RMSE']:.2f}")
                    insert_idx += 1
                
                with open(log_file, 'w') as f:
                    f.write('\n'.join(lines))
        else:
            with open(log_file, 'a') as f:
                f.write("\n### Automated Log Entry\n")
                for res in results_to_log:
                    f.write(f"- **{res['Description']}**: RMSE {res['RMSE']:.2f}\n")
    except Exception as e:
        print(f"Error updating experiment log: {e}")

def update_readme(results_file):
    readme_file = 'README.md'
    if not os.path.exists(readme_file) or not os.path.exists(results_file):
        return
    
    try:
        df = pd.read_csv(results_file, sep='\t')
        if df.empty:
            return
        
        best_idx = df['RMSE'].idxmin()
        best_rmse = df.loc[best_idx, 'RMSE']
        best_desc = df.loc[best_idx, 'Description']
        
        target_rmse = 5263.80
        gap = best_rmse - target_rmse
        
        with open(readme_file, 'r') as f:
            content = f.read()
        
        # Update Current Best RMSE
        content = re.sub(r'(- \*\*Current Best RMSE\*\*:\s*)\$[0-9,.]+\s*\(.*\)', 
                         rf'\1${best_rmse:,.2f} ({best_desc})', content)
        
        # Update Gap to Target
        content = re.sub(r'(- \*\*Gap to Target\*\*:\s*)\$[0-9,.]+', 
                         rf'\1${gap:,.2f}', content)
        
        with open(readme_file, 'w') as f:
            f.write(content)
    except Exception as e:
        print(f"Error updating README: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run AutoResearch Experiment')
    parser.add_argument('description', type=str, nargs='?', default='Unnamed Experiment', 
                        help='Description of the experiment/change')
    args = parser.parse_args()

    print(f"Starting Experiment: {args.description}")
    
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    
    print("Building model...")
    model = build_model(X_train)
    
    print("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - start_time
    print(f"Training complete in {runtime:.4f} seconds.")
    
    # Prepare list for logging
    results_to_log = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_file = 'results.tsv'

    # Check if it was a search (GridSearch/RandomizedSearch)
    if hasattr(model, 'cv_results_'):
        print("Search detected. Grouping by model type...")
        cv_res = model.cv_results_
        
        # Create a DataFrame of results
        df_cv = pd.DataFrame(cv_res['params'])
        df_cv['mean_test_score'] = cv_res['mean_test_score']
        df_cv['mean_fit_time'] = cv_res['mean_fit_time']
        df_cv['model_type'] = df_cv['regressor'].apply(lambda x: type(x).__name__)
        
        # Get the best index for each model type
        # Higher is better for neg_root_mean_squared_error
        best_indices = df_cv.groupby('model_type')['mean_test_score'].idxmax()
        
        for model_type, idx in best_indices.items():
            cv_rmse = -cv_res['mean_test_score'][idx]
            params = cv_res['params'][idx].copy()
            params.pop('regressor', None) # Remove the object for cleaner logging
            
            # For the overall best model, we also have the test set evaluation
            test_rmse = np.nan
            test_r2 = np.nan
            if idx == model.best_index_:
                print(f"Evaluating overall best model ({model_type}) on test set...")
                y_pred = model.predict(X_test)
                test_rmse, test_r2 = evaluate(y_test, y_pred)
                print(f"Best Model Test RMSE: {test_rmse:.4f}")

            results_to_log.append({
                'Timestamp': timestamp,
                'RMSE': test_rmse if idx == model.best_index_ else cv_rmse,
                'R2': test_r2 if idx == model.best_index_ else np.nan,
                'Runtime': cv_res['mean_fit_time'][idx],
                'Description': f"{model_type} | {args.description} | Best Params: {params} | CV_RMSE: {cv_rmse:.2f}"
            })
    else:
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        rmse, r2 = evaluate(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        
        results_to_log.append({
            'Timestamp': timestamp,
            'RMSE': rmse,
            'R2': r2,
            'Runtime': runtime,
            'Description': args.description
        })
    
    # Log results
    new_df = pd.DataFrame(results_to_log)
    if not os.path.exists(results_file):
        new_df.to_csv(results_file, sep='\t', index=False)
    else:
        df = pd.read_csv(results_file, sep='\t')
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(results_file, sep='\t', index=False)
    
    print(f"Logged {len(results_to_log)} model(s) to {results_file}")
    
    # Auto-update other logs
    update_experiment_log(results_to_log)
    update_readme(results_file)
    
    # Update plot
    plot_performance()
    print("Performance plot updated.")

if __name__ == "__main__":
    main()
