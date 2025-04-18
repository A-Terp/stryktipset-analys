"""
Web application for Stryktips analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import functions but don't load model yet
from src.models.predict import predict_match, predict_multiple_matches
from src.betting.systems import KellyCriterion, ValueBetting
from src.betting.optimization import optimize_bet_size, optimize_portfolio

app = FastAPI(title="Stryktips Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and scaler as None
model = None
scaler = None

# Try to load the model, or create a dummy one if it doesn't exist
try:
    from src.models.train import load_model
    model, scaler = load_model()
    print("Loaded existing model")
except (FileNotFoundError, ImportError):
    print("No model found. Creating a dummy model for demonstration purposes.")
    # Create a dummy model for demonstration
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Create some dummy data to fit the model
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)
    model.fit(X, y)
    
    # Create a dummy scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    print("Created dummy model for demonstration")

class Match(BaseModel):
    """Match data model."""
    match_id: str
    home_team: str
    away_team: str
    match_date: str
    home_goals: Optional[int] = 0
    away_goals: Optional[int] = 0
    attendance: Optional[int] = 0
    home_team_form: Optional[float] = 0.0
    away_team_form: Optional[float] = 0.0

class Odds(BaseModel):
    """Odds data model."""
    match_id: str
    home_odds: float
    draw_odds: float
    away_odds: float

class Prediction(BaseModel):
    """Prediction response model."""
    match_id: str
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_outcome: str

class BetRecommendation(BaseModel):
    """Bet recommendation model."""
    match_id: str
    outcome: str
    stake: float
    odds: float
    probability: float
    expected_value: float

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Stryktips Analysis API"}

@app.post("/predict/match", response_model=Prediction)
async def predict_single_match(match: Match):
    """
    Predict outcome for a single match.
    """
    try:
        # Convert match data to DataFrame
        match_data = pd.DataFrame([match.dict()])
        
        # Make prediction
        prediction = predict_match(model, scaler, match_data)
        
        # Format response
        return {
            "match_id": match.match_id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_win_prob": prediction['home_win'],
            "draw_prob": prediction['draw'],
            "away_win_prob": prediction['away_win'],
            "predicted_outcome": max(prediction.items(), key=lambda x: x[1])[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/matches", response_model=List[Prediction])
async def predict_matches(matches: List[Match]):
    """
    Predict outcomes for multiple matches.
    """
    try:
        # Convert matches to DataFrame
        matches_data = pd.DataFrame([match.dict() for match in matches])
        
        # Make predictions
        predictions = predict_multiple_matches(model, scaler, matches_data)
        
        # Format response
        return predictions.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/betting/recommendations", response_model=List[BetRecommendation])
async def get_bet_recommendations(matches: List[Match], odds: List[Odds]):
    """
    Get betting recommendations for matches.
    """
    try:
        # Convert data to DataFrames
        matches_data = pd.DataFrame([match.dict() for match in matches])
        odds_data = pd.DataFrame([odd.dict() for odd in odds])
        
        # Make predictions
        predictions = predict_multiple_matches(model, scaler, matches_data)
        
        # Get optimized bet sizes
        recommendations = optimize_bet_size(predictions, odds_data)
        
        # Format response
        return recommendations.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/betting/portfolio")
async def optimize_betting_portfolio(bets: List[BetRecommendation]):
    """
    Optimize betting portfolio.
    """
    try:
        # Convert bets to DataFrame
        bets_data = pd.DataFrame([bet.dict() for bet in bets])
        
        # Optimize portfolio
        portfolio = optimize_portfolio(bets_data)
        
        # Format response
        return {
            "portfolio": portfolio.to_dict('records'),
            "total_stake": portfolio['stake'].sum(),
            "expected_return": (portfolio['stake'] * (portfolio['odds'] - 1) * portfolio['probability']).sum()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 