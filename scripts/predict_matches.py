"""
Script to make predictions for upcoming matches.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loading import load_match_data, load_odds_data

def load_model():
    """Load the trained model and scaler."""
    model_dir = os.path.join('data', 'models')
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    return model, scaler

def prepare_match_data(match_data: dict) -> pd.DataFrame:
    """
    Prepare a single match's data for prediction.
    
    Args:
        match_data: Dictionary containing match information
        
    Returns:
        DataFrame with prepared match data
    """
    # Create DataFrame with required columns
    df = pd.DataFrame([{
        'home_goals': 0,  # Will be filled with average
        'away_goals': 0,  # Will be filled with average
        'home_shots': match_data.get('home_shots', 0),
        'away_shots': match_data.get('away_shots', 0),
        'home_shots_target': match_data.get('home_shots_target', 0),
        'away_shots_target': match_data.get('away_shots_target', 0),
        'home_corners': match_data.get('home_corners', 0),
        'away_corners': match_data.get('away_corners', 0),
        'home_odds': match_data.get('home_odds', 0),
        'draw_odds': match_data.get('draw_odds', 0),
        'away_odds': match_data.get('away_odds', 0)
    }])
    
    return df

def predict_match(model, scaler, match_data: dict) -> dict:
    """
    Make prediction for a single match.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        match_data: Dictionary containing match information
        
    Returns:
        Dictionary with prediction probabilities
    """
    # Prepare match data
    X = prepare_match_data(match_data)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get probabilities
    probabilities = model.predict_proba(X_scaled)[0]
    
    # Map to outcomes
    outcome_map = {0: 'draw', 1: 'home_win', 2: 'away_win'}
    predictions = {
        outcome_map[i]: prob for i, prob in enumerate(probabilities)
    }
    
    # Add predicted outcome
    predictions['predicted_outcome'] = outcome_map[np.argmax(probabilities)]
    
    return predictions

def load_upcoming_matches(filename: str = None) -> list:
    """
    Load upcoming matches from a JSON file.
    If no filename is provided, use the most recent file in data/upcoming/.
    """
    if filename is None:
        # Get the most recent file in data/upcoming/
        upcoming_dir = os.path.join('data', 'upcoming')
        if not os.path.exists(upcoming_dir):
            raise FileNotFoundError("No upcoming matches directory found. Run input_upcoming_matches.py first.")
        
        files = [f for f in os.listdir(upcoming_dir) if f.endswith('.json')]
        if not files:
            raise FileNotFoundError("No match data files found. Run input_upcoming_matches.py first.")
        
        filename = os.path.join(upcoming_dir, sorted(files)[-1])
    
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    """Main function to make predictions for upcoming matches."""
    # Load model and scaler
    print("Loading model...")
    model, scaler = load_model()
    
    try:
        # Load upcoming matches
        print("Loading upcoming matches...")
        upcoming_matches = load_upcoming_matches()
        
        # Make predictions
        print("\nPredictions for upcoming matches:")
        print("-" * 50)
        
        for match in upcoming_matches:
            predictions = predict_match(model, scaler, match)
            
            print(f"\n{match['home_team']} vs {match['away_team']}")
            print(f"Home win: {predictions['home_win']:.1%}")
            print(f"Draw: {predictions['draw']:.1%}")
            print(f"Away win: {predictions['away_win']:.1%}")
            print(f"Predicted outcome: {predictions['predicted_outcome']}")
            print("-" * 50)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run input_upcoming_matches.py first to input match data.")

if __name__ == "__main__":
    main() 