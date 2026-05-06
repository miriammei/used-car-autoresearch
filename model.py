import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, TargetEncoder, FunctionTransformer, OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def extract_features(df):
    """
    Feature engineering: drop Car ID, calculate Car_Age from Year, and Miles_Per_Year.
    """
    df = df.copy()
    if 'Car ID' in df.columns:
        df = df.drop(columns=['Car ID'])
    if 'Year' in df.columns:
        # Assuming current year is 2026 as per session context
        df['Car_Age'] = 2026 - df['Year']
        df = df.drop(columns=['Year'])
    
    if 'Mileage' in df.columns and 'Car_Age' in df.columns:
        df['Miles_Per_Year'] = df['Mileage'] / (df['Car_Age'] + 1)
        
    return df

def build_model(X):
    """
    Builds a pipeline with feature engineering, preprocessing, and a regressor.
    The regressor is selected via the BASE_MODEL environment variable.
    """
    model_type = os.getenv('BASE_MODEL', 'Ridge')
    
    # Discovery of features after transformation to pass to ColumnTransformer
    X_transformed = extract_features(X)
    numeric_features = X_transformed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Separate categorical features into ordinal and target encoded
    all_categorical = X_transformed.select_dtypes(include=['object', 'category']).columns.tolist()
    ordinal_features = ['Condition'] if 'Condition' in all_categorical else []
    target_enc_features = [c for c in all_categorical if c != 'Condition']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # Ordinal mapping for Condition
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[['Used', 'Like New', 'New']], handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    target_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_enc', TargetEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('target', target_transformer, target_enc_features)
        ])

    regressors = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Huber': HuberRegressor(),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }

    regressor = regressors.get(model_type, Ridge())
    
    # Main pipeline
    model = Pipeline(steps=[
        ('feat_eng', FunctionTransformer(extract_features)),
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    print(f"Building model with: {model_type} (Robust Preprocessing)")

    return model
