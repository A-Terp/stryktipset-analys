"""
Model training module for Stryktips analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Tuple, Dict, Any
import os

def prepare_data(df: pd.DataFrame, target_col: str = 'result') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for model training.
    
    Args:
        df: DataFrame with features
        target_col: Name of the target column
        
    Returns:
        Tuple of (X, y) arrays
    """
    # Select features (excluding target and non-feature columns)
    exclude_cols = [target_col, 'match_date', 'home_team', 'away_team']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y

def train_model(X: np.ndarray, y: np.ndarray, 
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (best_model, metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=random_state)
    
    # Perform grid search
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'test_accuracy': best_model.score(X_test_scaled, y_test),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return best_model, metrics

def save_model(model: Any, scaler: Any, model_dir: str = '../data/models'):
    """
    Save trained model and scaler.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        model_dir: Directory to save models
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))

def load_model(model_dir: str = '../data/models') -> Tuple[Any, Any]:
    """
    Load trained model and scaler.
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        Tuple of (model, scaler)
    """
    # Check if model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory {model_dir} does not exist. Created it.")
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    # Check if model files exist
    model_path = os.path.join(model_dir, 'model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model files not found in {model_dir}")
        raise FileNotFoundError(f"Model files not found in {model_dir}")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler 