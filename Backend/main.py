from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import os
import requests
import logging
import sys
import json
from pathlib import Path

# Add your RL module to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_trader import DQNAgent, MultiStockEnv, get_data  # Import your actual module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
MODEL_DIR = "rl_trader_models"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn.weights.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
NORM_PATH = os.path.join(MODEL_DIR, "normalization_stats.pkl")
TF_SERVING_URL = os.getenv("TF_SERVING_URL", "http://localhost:8501")

class PredictionRequest(BaseModel):
    state: list
    holdings: list
    cash: float

class PortfolioResponse(BaseModel):
    portfolio_values: list
    final_value: float

@app.on_event("startup")
async def load_assets():
    try:
        # Load normalization stats
        with open(NORM_PATH, 'rb') as f:
            app.state.norm_stats = pickle.load(f)
            logger.info("Loaded normalization parameters")
        
        # Load scaler
        with open(SCALER_PATH, 'rb') as f:
            app.state.scaler = pickle.load(f)
            logger.info("Loaded scaler")
        
        # Initialize agent
        state_size = app.state.scaler.n_features_in_
        action_size = 27  # 3^3 actions
        app.state.agent = DQNAgent(state_size, action_size)
        app.state.agent.load(MODEL_PATH)
        logger.info("Loaded agent model")
        
        # Initialize test environment
        test_data = get_data()
        n_timesteps = test_data.shape[0]
        n_train = n_timesteps * 2 // 3
        test_data = test_data[n_train:]
        
        app.state.test_env = MultiStockEnv(
            test_data, 
            alpha=0.5,
            beta=0.5,
            initial_investment=20000
        )
        app.state.test_env.set_normalization_params(*app.state.norm_stats)
        logger.info("Initialized test environment")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Initialization error: {str(e)}")

def predict_via_tf_serving(state):
    payload = {
        "signature_name": "serving_default",
        "instances": [state.tolist()]
    }
    
    try:
        response = requests.post(
            f"{TF_SERVING_URL}/v1/models/rl_trader:predict",
            json=payload,
            timeout=1.0
        )
        response.raise_for_status()
        return response.json()['predictions'][0]
    except Exception as e:
        logger.error(f"TF Serving error: {str(e)}")
        return None

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Create full state vector: portfolio + market features
        portfolio = [
            request.holdings[0] / 100,
            request.holdings[1] / 100,
            request.holdings[2] / 100,
            request.cash / 20000  # Normalized by initial investment
        ]
        state_vector = np.array(portfolio + request.state, dtype=np.float32)
        scaled_state = app.state.scaler.transform([state_vector])
        
        # Get prediction
        q_values = predict_via_tf_serving(scaled_state)
        if q_values is None:
            q_values = app.state.agent.model.predict(scaled_state, verbose=0)[0]
        
        action = np.argmax(q_values)
        return {"action": int(action)}
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/run-test-episode", response_model=PortfolioResponse)
def run_test_episode():
    try:
        portfolio_values = []
        state = app.state.test_env.reset()
        done = False
        
        while not done:
            scaled_state = app.state.scaler.transform([state])
            action = app.state.agent.act(scaled_state)
            next_state, _, done, info = app.state.test_env.step(action)
            portfolio_values.append(info['curr_val'])
            state = next_state
        
        return PortfolioResponse(
            portfolio_values=portfolio_values,
            final_value=portfolio_values[-1] if portfolio_values else 0
        )
        
    except Exception as e:
        logger.error(f"Test episode failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))