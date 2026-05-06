import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def extract_features(df):
    """
    Feature engineering: drop Car ID, calculate Car_Age from Year.
    """
    df = df.copy()
    if 'Car ID' in df.columns:
        df = df.drop(columns=['Car ID'])
    if 'Year' in df.columns:
        # Assuming current year is 2026 as per session context
        df['Car_Age'] = 2026 - df['Year']
        df = df.drop(columns=['Year'])
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
    categorical_features = X_transformed.select_dtypes(include=['object', 'category']).columns.tolist()

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

    regressors = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
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
    
    print(f"Building model with: {model_type} (Feature Eng Only)")

    return model
