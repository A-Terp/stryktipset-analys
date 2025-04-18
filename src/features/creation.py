"""
Feature creation module for Stryktips analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta

def create_team_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Create team form features based on recent matches.
    
    Args:
        df: Match data DataFrame
        window: Number of recent matches to consider
        
    Returns:
        DataFrame with additional form features
    """
    df_form = df.copy()
    
    # Sort by date
    df_form = df_form.sort_values('match_date')
    
    # Initialize form columns
    df_form['home_team_form'] = 0.0
    df_form['away_team_form'] = 0.0
    
    # Calculate form for each team
    for idx, row in df_form.iterrows():
        # Home team form
        home_team = row['home_team']
        home_matches = df_form[
            (df_form['match_date'] < row['match_date']) &
            ((df_form['home_team'] == home_team) | (df_form['away_team'] == home_team))
        ].tail(window)
        
        if len(home_matches) > 0:
            home_form = calculate_team_form(home_matches, home_team)
            df_form.loc[idx, 'home_team_form'] = home_form
        
        # Away team form
        away_team = row['away_team']
        away_matches = df_form[
            (df_form['match_date'] < row['match_date']) &
            ((df_form['home_team'] == away_team) | (df_form['away_team'] == away_team))
        ].tail(window)
        
        if len(away_matches) > 0:
            away_form = calculate_team_form(away_matches, away_team)
            df_form.loc[idx, 'away_team_form'] = away_form
    
    return df_form

def calculate_team_form(matches: pd.DataFrame, team: str) -> float:
    """
    Calculate team form based on recent matches.
    
    Args:
        matches: DataFrame of recent matches
        team: Team name
        
    Returns:
        Form score between 0 and 1
    """
    form_score = 0.0
    total_matches = len(matches)
    
    if total_matches == 0:
        return form_score
    
    for _, match in matches.iterrows():
        # Determine if team was home or away
        is_home = match['home_team'] == team
        
        # Get goals
        team_goals = match['home_goals'] if is_home else match['away_goals']
        opponent_goals = match['away_goals'] if is_home else match['home_goals']
        
        # Calculate match points
        if team_goals > opponent_goals:
            form_score += 1.0
        elif team_goals == opponent_goals:
            form_score += 0.5
    
    return form_score / total_matches

def create_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from betting odds.
    
    Args:
        df: DataFrame with odds data
        
    Returns:
        DataFrame with additional odds-based features
    """
    df_odds = df.copy()
    
    # Calculate implied probabilities
    df_odds['home_implied_prob'] = 1 / df_odds['home_odds']
    df_odds['draw_implied_prob'] = 1 / df_odds['draw_odds']
    df_odds['away_implied_prob'] = 1 / df_odds['away_odds']
    
    # Calculate value bets
    df_odds['home_value'] = df_odds['home_implied_prob'] - df_odds['home_team_form']
    df_odds['draw_value'] = df_odds['draw_implied_prob'] - 0.33  # Assuming 33% draw probability
    df_odds['away_value'] = df_odds['away_implied_prob'] - df_odds['away_team_form']
    
    return df_odds

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.
    
    Args:
        df: DataFrame with match data
        
    Returns:
        DataFrame with additional time-based features
    """
    df_time = df.copy()
    
    # Convert match_date to datetime if not already
    df_time['match_date'] = pd.to_datetime(df_time['match_date'])
    
    # Extract time features
    df_time['year'] = df_time['match_date'].dt.year
    df_time['month'] = df_time['match_date'].dt.month
    df_time['day_of_week'] = df_time['match_date'].dt.dayofweek
    df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
    
    return df_time 