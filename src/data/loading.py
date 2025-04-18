"""
Data loading module for Stryktips analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime

def load_match_data(pl_path: str = 'data/raw/pl.csv', 
                   cs_path: str = 'data/raw/cs.csv') -> pd.DataFrame:
    """
    Load and combine match data from Premier League and Championship.
    
    Args:
        pl_path: Path to Premier League data
        cs_path: Path to Championship data
        
    Returns:
        Combined DataFrame with match data
    """
    # Load both datasets
    pl_data = pd.read_csv(pl_path)
    cs_data = pd.read_csv(cs_path)
    
    # Combine datasets
    combined_data = pd.concat([pl_data, cs_data], ignore_index=True)
    
    # Convert date columns
    combined_data['Date'] = pd.to_datetime(combined_data['Date'], format='%d/%m/%Y')
    
    # Rename columns to match application requirements
    combined_data = combined_data.rename(columns={
        'Date': 'match_date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'FTR': 'result',
        'HS': 'home_shots',
        'AS': 'away_shots',
        'HST': 'home_shots_target',
        'AST': 'away_shots_target',
        'HC': 'home_corners',
        'AC': 'away_corners'
    })
    
    # Create match_id
    combined_data['match_id'] = combined_data.apply(
        lambda x: f"{x['Div']}_{x['match_date'].strftime('%Y%m%d')}_{x['home_team']}_{x['away_team']}", 
        axis=1
    )
    
    return combined_data

def load_odds_data(pl_path: str = 'data/raw/pl.csv', 
                  cs_path: str = 'data/raw/cs.csv') -> pd.DataFrame:
    """
    Load and combine odds data from Premier League and Championship.
    
    Args:
        pl_path: Path to Premier League data
        cs_path: Path to Championship data
        
    Returns:
        Combined DataFrame with odds data
    """
    # Load both datasets
    pl_data = pd.read_csv(pl_path)
    cs_data = pd.read_csv(cs_path)
    
    # Combine datasets
    combined_data = pd.concat([pl_data, cs_data], ignore_index=True)
    
    # Convert date columns
    combined_data['Date'] = pd.to_datetime(combined_data['Date'], format='%d/%m/%Y')
    
    # Create match_id
    combined_data['match_id'] = combined_data.apply(
        lambda x: f"{x['Div']}_{x['Date'].strftime('%Y%m%d')}_{x['HomeTeam']}_{x['AwayTeam']}", 
        axis=1
    )
    
    # Select and rename odds columns
    odds_data = combined_data[[
        'match_id',
        'B365H', 'B365D', 'B365A',  # Bet365 odds
        'BWH', 'BWD', 'BWA',        # BetWin odds
        'PSH', 'PSD', 'PSA',        # Pinnacle odds
        'WHH', 'WHD', 'WHA'         # William Hill odds
    ]].rename(columns={
        'B365H': 'home_odds',
        'B365D': 'draw_odds',
        'B365A': 'away_odds'
    })
    
    # Calculate average odds across bookmakers
    odds_data['home_odds'] = combined_data[['B365H', 'BWH', 'PSH', 'WHH']].mean(axis=1)
    odds_data['draw_odds'] = combined_data[['B365D', 'BWD', 'PSD', 'WHD']].mean(axis=1)
    odds_data['away_odds'] = combined_data[['B365A', 'BWA', 'PSA', 'WHA']].mean(axis=1)
    
    return odds_data

def prepare_training_data(matches_df: pd.DataFrame, 
                        odds_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training.
    
    Args:
        matches_df: DataFrame with match data
        odds_df: DataFrame with odds data
        
    Returns:
        Tuple of (X, y) where X is the feature matrix and y is the target
    """
    # Merge match and odds data
    data = pd.merge(matches_df, odds_df, on='match_id', how='left')
    
    # Select features for training
    feature_columns = [
        'home_goals', 'away_goals', 'home_shots', 'away_shots',
        'home_shots_target', 'away_shots_target', 'home_corners', 'away_corners',
        'home_odds', 'draw_odds', 'away_odds'
    ]
    
    # Create target variable (0: draw, 1: home win, 2: away win)
    target_map = {'D': 0, 'H': 1, 'A': 2}
    y = data['result'].map(target_map)
    
    # Select features
    X = data[feature_columns]
    
    return X, y 