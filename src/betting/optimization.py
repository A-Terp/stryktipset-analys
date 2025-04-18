"""
Betting optimization module for Stryktips analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from .systems import Bet, KellyCriterion

def optimize_bet_size(predictions: pd.DataFrame,
                     odds: pd.DataFrame,
                     bankroll: float = 10000.0) -> pd.DataFrame:
    """
    Optimize bet sizes using Kelly Criterion.
    
    Args:
        predictions: DataFrame with match predictions
        odds: DataFrame with betting odds
        bankroll: Available bankroll
        
    Returns:
        DataFrame with optimized bet sizes
    """
    kelly = KellyCriterion(bankroll=bankroll)
    optimized_bets = []
    
    for _, row in predictions.iterrows():
        match_odds = odds[odds['match_id'] == row['match_id']].iloc[0]
        
        # Calculate stakes for each outcome
        home_stake = kelly.calculate_stake(row['home_win_prob'], match_odds['home_odds'])
        draw_stake = kelly.calculate_stake(row['draw_prob'], match_odds['draw_odds'])
        away_stake = kelly.calculate_stake(row['away_win_prob'], match_odds['away_odds'])
        
        # Only include bets with positive expected value
        if home_stake > 0:
            optimized_bets.append({
                'match_id': row['match_id'],
                'outcome': 'home_win',
                'stake': home_stake,
                'odds': match_odds['home_odds'],
                'probability': row['home_win_prob']
            })
        
        if draw_stake > 0:
            optimized_bets.append({
                'match_id': row['match_id'],
                'outcome': 'draw',
                'stake': draw_stake,
                'odds': match_odds['draw_odds'],
                'probability': row['draw_prob']
            })
        
        if away_stake > 0:
            optimized_bets.append({
                'match_id': row['match_id'],
                'outcome': 'away_win',
                'stake': away_stake,
                'odds': match_odds['away_odds'],
                'probability': row['away_win_prob']
            })
    
    return pd.DataFrame(optimized_bets)

def optimize_portfolio(bets: pd.DataFrame,
                      bankroll: float = 10000.0,
                      risk_tolerance: float = 0.5) -> pd.DataFrame:
    """
    Optimize betting portfolio using modern portfolio theory.
    
    Args:
        bets: DataFrame with potential bets
        bankroll: Available bankroll
        risk_tolerance: Risk tolerance parameter (0-1)
        
    Returns:
        DataFrame with optimized portfolio weights
    """
    def objective(weights):
        # Calculate expected return
        expected_return = np.sum(weights * (bets['odds'] - 1) * bets['probability'])
        
        # Calculate risk (standard deviation)
        risk = np.sqrt(np.sum(weights**2 * bets['probability'] * (1 - bets['probability'])))
        
        # Return negative Sharpe ratio (to minimize)
        return -(expected_return / risk)
    
    # Initial guess
    n_bets = len(bets)
    x0 = np.ones(n_bets) / n_bets
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'ineq', 'fun': lambda x: bankroll - np.sum(x * bankroll)}  # Within bankroll
    ]
    
    # Bounds
    bounds = [(0, 1) for _ in range(n_bets)]
    
    # Optimize
    result = minimize(objective, x0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    # Create optimized portfolio
    portfolio = bets.copy()
    portfolio['weight'] = result.x
    portfolio['stake'] = portfolio['weight'] * bankroll
    
    return portfolio

def calculate_expected_value(bet: Bet) -> float:
    """
    Calculate expected value of a bet.
    
    Args:
        bet: Bet object
        
    Returns:
        Expected value of the bet
    """
    return (bet.odds - 1) * bet.probability - (1 - bet.probability)

def calculate_roi(bets: pd.DataFrame, results: pd.DataFrame) -> float:
    """
    Calculate return on investment for a series of bets.
    
    Args:
        bets: DataFrame with placed bets
        results: DataFrame with actual results
        
    Returns:
        Return on investment as a percentage
    """
    total_stake = bets['stake'].sum()
    total_return = 0.0
    
    for _, bet in bets.iterrows():
        result = results[results['match_id'] == bet['match_id']].iloc[0]
        if result['outcome'] == bet['outcome']:
            total_return += bet['stake'] * bet['odds']
    
    return (total_return - total_stake) / total_stake * 100 