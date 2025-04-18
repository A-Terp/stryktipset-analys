"""
Model prediction module for Stryktips analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import random

def predict_match(model: Any, scaler: Any, match_data: pd.DataFrame) -> Dict[str, float]:
    """
    Make predictions for a single match.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        match_data: DataFrame with match features
        
    Returns:
        Dictionary with predicted probabilities for each outcome
    """
    try:
        # Prepare features
        feature_cols = [col for col in match_data.columns 
                       if col not in ['match_date', 'home_team', 'away_team', 'result']]
        X = match_data[feature_cols].values
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Get probabilities
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Map to outcomes
        outcome_map = {0: 'draw', 1: 'home_win', 2: 'away_win'}
        predictions = {
            outcome_map[i]: prob for i, prob in enumerate(probabilities)
        }
        
        return predictions
    except Exception as e:
        print(f"Error in predict_match: {e}")
        # Return random probabilities as fallback
        probs = [random.random() for _ in range(3)]
        total = sum(probs)
        return {
            'home_win': probs[0] / total,
            'draw': probs[1] / total,
            'away_win': probs[2] / total
        }

def predict_multiple_matches(model: Any, scaler: Any, 
                          matches_data: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions for multiple matches.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        matches_data: DataFrame with multiple matches
        
    Returns:
        DataFrame with predictions for each match
    """
    predictions = []
    
    for _, match in matches_data.iterrows():
        match_pred = predict_match(model, scaler, pd.DataFrame([match]))
        predictions.append({
            'match_id': match.get('match_id', None),
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'match_date': match['match_date'],
            'home_win_prob': match_pred['home_win'],
            'draw_prob': match_pred['draw'],
            'away_win_prob': match_pred['away_win'],
            'predicted_outcome': max(match_pred.items(), key=lambda x: x[1])[0]
        })
    
    return pd.DataFrame(predictions)

def evaluate_predictions(predictions: pd.DataFrame, 
                       actual_results: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate prediction accuracy.
    
    Args:
        predictions: DataFrame with predictions
        actual_results: DataFrame with actual match results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Merge predictions with actual results
    evaluation = pd.merge(
        predictions,
        actual_results[['match_id', 'result']],
        on='match_id'
    )
    
    # Calculate metrics
    total_matches = len(evaluation)
    correct_predictions = sum(
        evaluation['predicted_outcome'] == evaluation['result']
    )
    
    metrics = {
        'accuracy': correct_predictions / total_matches,
        'total_matches': total_matches,
        'correct_predictions': correct_predictions
    }
    
    return metrics

def get_value_bets(predictions: pd.DataFrame, 
                  odds: pd.DataFrame,
                  threshold: float = 0.1) -> pd.DataFrame:
    """
    Identify potential value bets based on predictions and odds.
    
    Args:
        predictions: DataFrame with match predictions
        odds: DataFrame with betting odds
        threshold: Minimum value required to consider a bet
        
    Returns:
        DataFrame with identified value bets
    """
    # Merge predictions with odds
    value_bets = pd.merge(predictions, odds, on='match_id')
    
    # Calculate value for each outcome
    value_bets['home_value'] = value_bets['home_win_prob'] - (1 / value_bets['home_odds'])
    value_bets['draw_value'] = value_bets['draw_prob'] - (1 / value_bets['draw_odds'])
    value_bets['away_value'] = value_bets['away_win_prob'] - (1 / value_bets['away_odds'])
    
    # Filter for value bets
    value_bets = value_bets[
        (value_bets['home_value'] > threshold) |
        (value_bets['draw_value'] > threshold) |
        (value_bets['away_value'] > threshold)
    ]
    
    return value_bets 