"""
Data cleaning module for Stryktips analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the raw data
    """
    return pd.read_csv(file_path)

def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean match data by handling missing values and standardizing formats.
    
    Args:
        df: Raw match data DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    df_clean = df_clean.fillna({
        'home_goals': 0,
        'away_goals': 0,
        'attendance': df_clean['attendance'].median()
    })
    
    # Convert date columns to datetime
    date_columns = ['match_date', 'draw_date']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col])
    
    # Standardize team names
    df_clean['home_team'] = df_clean['home_team'].str.strip().str.lower()
    df_clean['away_team'] = df_clean['away_team'].str.strip().str.lower()
    
    return df_clean

def create_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from match data.
    
    Args:
        df: Cleaned match data DataFrame
        
    Returns:
        DataFrame with additional features
    """
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # Calculate total goals
    df_features['total_goals'] = df_features['home_goals'] + df_features['away_goals']
    
    # Calculate goal difference
    df_features['goal_difference'] = df_features['home_goals'] - df_features['away_goals']
    
    # Create result column (1: home win, 0: draw, 2: away win)
    df_features['result'] = np.where(df_features['goal_difference'] > 0, 1,
                                   np.where(df_features['goal_difference'] < 0, 2, 0))
    
    return df_features

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Tuple of (X, y) where X is the feature matrix and y is the target
    """
    # Select features for training
    feature_columns = [
        'home_goals', 'away_goals', 'total_goals', 'goal_difference',
        'attendance'
    ]
    
    X = df[feature_columns]
    y = df['result']
    
    return X, y 