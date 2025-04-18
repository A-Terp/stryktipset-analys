"""
Script to parse Stryktips format and convert it to model input format.
"""

import json
import os
import re
from datetime import datetime

def parse_stryktips_format(input_text: str) -> list:
    """
    Parse the Stryktips format text into a list of match dictionaries.
    
    Args:
        input_text: String containing the Stryktips format text
        
    Returns:
        List of match dictionaries
    """
    # Split the input into lines and remove empty lines
    lines = [line.strip() for line in input_text.split('\n') if line.strip()]
    
    matches = []
    current_match = {}
    
    for i, line in enumerate(lines):
        # Check if this is a new match (starts with a number)
        if re.match(r'^\d+$', line):
            # If we have a previous match, add it to matches
            if current_match:
                matches.append(current_match)
            
            # Start a new match
            current_match = {'match_number': int(line)}
        
        # Home team (after match number)
        elif i % 7 == 1:
            current_match['home_team'] = line.strip('-')
        
        # Away team (after home team)
        elif i % 7 == 3:
            current_match['away_team'] = line.strip('-')
        
        # Home win probability (after "Svenska folket")
        elif i % 7 == 5 and 'home_win_prob' not in current_match:
            current_match['home_win_prob'] = float(line.strip('%')) / 100
        
        # Draw probability
        elif i % 7 == 6 and 'draw_prob' not in current_match:
            current_match['draw_prob'] = float(line.strip('%')) / 100
        
        # Away win probability
        elif i % 7 == 0 and 'away_win_prob' not in current_match:
            current_match['away_win_prob'] = float(line.strip('%')) / 100
        
        # Home odds (after "Odds")
        elif i % 7 == 2 and 'home_odds' not in current_match:
            current_match['home_odds'] = float(line.replace(',', '.'))
        
        # Draw odds
        elif i % 7 == 3 and 'draw_odds' not in current_match:
            current_match['draw_odds'] = float(line.replace(',', '.'))
        
        # Away odds
        elif i % 7 == 4 and 'away_odds' not in current_match:
            current_match['away_odds'] = float(line.replace(',', '.'))
    
    # Add the last match
    if current_match:
        matches.append(current_match)
    
    return matches

def convert_to_model_format(matches: list) -> list:
    """
    Convert Stryktips format matches to model input format.
    
    Args:
        matches: List of matches in Stryktips format
        
    Returns:
        List of matches in model input format
    """
    model_matches = []
    
    for match in matches:
        # Use default values for stats based on league averages
        model_match = {
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'home_shots': 12.0,  # Default average
            'away_shots': 10.0,  # Default average
            'home_shots_target': 4.0,  # Default average
            'away_shots_target': 3.0,  # Default average
            'home_corners': 5.0,  # Default average
            'away_corners': 4.0,  # Default average
            'home_odds': match['home_odds'],
            'draw_odds': match['draw_odds'],
            'away_odds': match['away_odds']
        }
        
        model_matches.append(model_match)
    
    return model_matches

def main():
    """Main function to parse Stryktips format and save to model input format."""
    print("Enter Stryktips format data (press Ctrl+D or Ctrl+Z when done):")
    
    # Read input until EOF
    input_lines = []
    try:
        while True:
            line = input()
            input_lines.append(line)
    except EOFError:
        pass
    
    input_text = '\n'.join(input_lines)
    
    # Parse the input
    print("\nParsing Stryktips format...")
    stryktips_matches = parse_stryktips_format(input_text)
    
    # Convert to model format
    print("Converting to model format...")
    model_matches = convert_to_model_format(stryktips_matches)
    
    # Create directory if it doesn't exist
    os.makedirs('data/upcoming', exist_ok=True)
    
    # Save to file
    filename = f"data/upcoming/matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(model_matches, f, indent=2)
    
    print(f"\nMatch data saved to {filename}")
    print(f"Found {len(model_matches)} matches")
    print("\nYou can now run predict_matches.py to get predictions.")

if __name__ == "__main__":
    main() 