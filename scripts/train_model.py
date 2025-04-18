"""
Script to train a model using actual match data.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loading import load_match_data, load_odds_data, prepare_training_data

def train_model():
    """Train a model using actual match data."""
    # Load data
    print("Loading match data...")
    matches_df = load_match_data()
    
    print("Loading odds data...")
    odds_df = load_odds_data()
    
    print("Preparing training data...")
    X, y = prepare_training_data(matches_df, odds_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and fit scaler
    print("Fitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Create models directory if it doesn't exist
    model_dir = os.path.join('data', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and scaler
    print(f"Saving model to {model_dir}...")
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    print("Done!")

if __name__ == "__main__":
    train_model() 