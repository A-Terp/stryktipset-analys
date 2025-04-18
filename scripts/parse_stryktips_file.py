"""
Script to parse Stryktips format from a file and convert it to model input format.
"""

import json
import os
import re
import sys
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
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a new match (starts with a number)
        if re.match(r'^\d+$', line):
            current_match = {'match_number': int(line)}
            
            # Next 4 lines are: home team, separator, away team, "Svenska folket"
            if i + 4 < len(lines):
                current_match['home_team'] = lines[i + 1].strip()
                current_match['away_team'] = lines[i + 3].strip()
                
                # Next 3 lines are probabilities
                if i + 7 < len(lines) and lines[i + 4] == "Svenska folket":
                    current_match['home_win_prob'] = float(lines[i + 5].strip('%')) / 100
                    current_match['draw_prob'] = float(lines[i + 6].strip('%')) / 100
                    current_match['away_win_prob'] = float(lines[i + 7].strip('%')) / 100
                
                # Next 4 lines are: "Odds" and the three odds
                if i + 11 < len(lines) and lines[i + 8] == "Odds":
                    odds_str = lines[i + 9].replace(',', '.')
                    current_match['home_odds'] = float(odds_str) if odds_str != '-' else 0.0
                    
                    odds_str = lines[i + 10].replace(',', '.')
                    current_match['draw_odds'] = float(odds_str) if odds_str != '-' else 0.0
                    
                    odds_str = lines[i + 11].replace(',', '.')
                    current_match['away_odds'] = float(odds_str) if odds_str != '-' else 0.0
                
                matches.append(current_match)
                i += 12  # Move to the next match
            else:
                i += 1
        else:
            i += 1
    
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
    """Main function to parse Stryktips format from a file."""
    if len(sys.argv) < 2:
        print("Usage: python parse_stryktips_file.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    # Read the input file
    with open(input_file, 'r') as f:
        input_text = f.read()
    
    # Parse the input
    print(f"Parsing Stryktips format from {input_file}...")
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