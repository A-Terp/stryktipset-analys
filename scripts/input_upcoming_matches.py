"""
Script to help users input upcoming match data interactively.
"""

import json
import os
from datetime import datetime

def get_float_input(prompt: str) -> float:
    """Get a float input from the user with error handling."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def input_match_data() -> dict:
    """Get match data from user input."""
    print("\nEnter match details:")
    print("-" * 50)
    
    match_data = {
        'home_team': input("Home team: ").strip(),
        'away_team': input("Away team: ").strip(),
        'home_shots': get_float_input("Home team average shots per game: "),
        'away_shots': get_float_input("Away team average shots per game: "),
        'home_shots_target': get_float_input("Home team average shots on target per game: "),
        'away_shots_target': get_float_input("Away team average shots on target per game: "),
        'home_corners': get_float_input("Home team average corners per game: "),
        'away_corners': get_float_input("Away team average corners per game: "),
        'home_odds': get_float_input("Home win odds: "),
        'draw_odds': get_float_input("Draw odds: "),
        'away_odds': get_float_input("Away win odds: ")
    }
    
    return match_data

def main():
    """Main function to collect upcoming match data."""
    matches = []
    
    while True:
        match_data = input_match_data()
        matches.append(match_data)
        
        if input("\nAdd another match? (y/n): ").lower() != 'y':
            break
    
    # Create directory if it doesn't exist
    os.makedirs('data/upcoming', exist_ok=True)
    
    # Save matches to file
    filename = f"data/upcoming/matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(matches, f, indent=2)
    
    print(f"\nMatch data saved to {filename}")
    print("You can now run predict_matches.py with this data.")

if __name__ == "__main__":
    main() 