"""
Script to train a dummy model for testing purposes.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def create_dummy_data(n_samples=1000):
    """Create dummy data for training."""
    np.random.seed(42)
    
    # Create random features
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Create random target (0: home win, 1: draw, 2: away win)
    y = np.random.randint(0, 3, size=n_samples)
    
    return X, y

def train_dummy_model():
    """Train a dummy model and save it."""
    # Create dummy data
    X, y = create_dummy_data()
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Create models directory if it doesn't exist
    model_dir = os.path.join('data', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    print(f"Dummy model saved to {model_dir}")

if __name__ == "__main__":
    train_dummy_model() 