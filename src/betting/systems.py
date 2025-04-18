"""
Betting systems module for Stryktips analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Bet:
    """Class for representing a bet."""
    match_id: str
    outcome: str
    odds: float
    stake: float
    probability: float
    expected_value: float

class KellyCriterion:
    """Implementation of the Kelly Criterion for bet sizing."""
    
    def __init__(self, bankroll: float = 10000.0, kelly_fraction: float = 0.5):
        """
        Initialize Kelly Criterion calculator.
        
        Args:
            bankroll: Initial bankroll
            kelly_fraction: Fraction of Kelly to bet (0-1)
        """
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
    
    def calculate_fraction(self, probability: float, odds: float) -> float:
        """
        Calculate Kelly fraction for a bet.
        
        Args:
            probability: Estimated probability of winning
            odds: Decimal odds
            
        Returns:
            Fraction of bankroll to bet
        """
        q = 1 - probability
        b = odds - 1
        
        if b <= 0 or probability <= 0:
            return 0.0
        
        kelly = (b * probability - q) / b
        return max(0.0, min(kelly * self.kelly_fraction, 1.0))
    
    def calculate_stake(self, probability: float, odds: float) -> float:
        """
        Calculate optimal stake for a bet.
        
        Args:
            probability: Estimated probability of winning
            odds: Decimal odds
            
        Returns:
            Optimal stake amount
        """
        fraction = self.calculate_fraction(probability, odds)
        return self.bankroll * fraction

class ValueBetting:
    """Implementation of value betting strategy."""
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize value betting strategy.
        
        Args:
            threshold: Minimum value required to consider a bet
        """
        self.threshold = threshold
    
    def find_value_bets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Find value bets in the dataset.
        
        Args:
            data: DataFrame with predictions and odds
            
        Returns:
            DataFrame with identified value bets
        """
        value_bets = []
        
        for _, row in data.iterrows():
            # Calculate value for each outcome
            home_value = row['home_win_prob'] - (1 / row['home_odds'])
            draw_value = row['draw_prob'] - (1 / row['draw_odds'])
            away_value = row['away_win_prob'] - (1 / row['away_odds'])
            
            # Check if any outcome has value
            if max(home_value, draw_value, away_value) > self.threshold:
                value_bets.append({
                    'match_id': row['match_id'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'home_value': home_value,
                    'draw_value': draw_value,
                    'away_value': away_value,
                    'best_value': max(home_value, draw_value, away_value)
                })
        
        return pd.DataFrame(value_bets)

class MartingaleSystem:
    """Implementation of the Martingale betting system."""
    
    def __init__(self, initial_stake: float = 100.0, max_stakes: int = 5):
        """
        Initialize Martingale system.
        
        Args:
            initial_stake: Initial stake amount
            max_stakes: Maximum number of consecutive losses before stopping
        """
        self.initial_stake = initial_stake
        self.max_stakes = max_stakes
        self.current_stake = initial_stake
        self.consecutive_losses = 0
    
    def calculate_next_stake(self, won_last_bet: bool) -> Optional[float]:
        """
        Calculate stake for the next bet.
        
        Args:
            won_last_bet: Whether the last bet was won
            
        Returns:
            Stake for next bet, or None if should stop betting
        """
        if won_last_bet:
            self.current_stake = self.initial_stake
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_stakes:
                return None
            self.current_stake *= 2
        
        return self.current_stake
    
    def reset(self):
        """Reset the system to initial state."""
        self.current_stake = self.initial_stake
        self.consecutive_losses = 0 