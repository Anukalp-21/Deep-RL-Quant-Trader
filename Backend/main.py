from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import os
from datetime import datetime
from your_rl_module import MultiStockEnv, DQNAgent, get_data  # Import your existing code
import requests
import json

TF_SERVING_URL = os.getenv("TF_SERVING_URL", "http://localhost:8501")

app = FastAPI()

# Load models and scalers
MODEL_PATH = "rl_trader_models/dqn.weights.h5"
SCALER_PATH = "rl_trader_models/scaler.pkl"
CONFIG_PATH = "rl_trader_models/config.pkl"

# Initialize on startup
@app.on_event("startup")
async def load_model():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]):
        raise RuntimeError("Model files not found!")
    
    with open(CONFIG_PATH, 'rb') as f:
        state_size, action_size = pickle.load(f)
    
    app.state.agent = DQNAgent(state_size, action_size)
    app.state.agent.load(MODEL_PATH)
    
    with open(SCALER_PATH, 'rb') as f:
        app.state.scaler = pickle.load(f)
    
    # Initialize test environment
    test_data = get_data()
    app.state.test_env = MultiStockEnv(test_data, 0.5, 0.5, 20000)

class PredictionRequest(BaseModel):
    state: list

@app.get("/run-test-episode")
def run_test_episode():
    try:
        portfolio_values = []
        state = app.state.test_env.reset()
        done = False
        
        while not done:
            scaled_state = app.state.scaler.transform([state])
            action = app.state.agent.act(scaled_state)
            next_state, _, done, _ = app.state.test_env.step(action)
            portfolio_values.append(app.state.test_env._get_val())
            state = next_state
        
        return {
            "portfolio_values": portfolio_values,
            "final_value": portfolio_values[-1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def predict_via_tf_serving(state):
    """Send prediction request to TensorFlow Serving"""
    payload = {
        "signature_name": "serving_default",
        "inputs": state.tolist()
    }
    
    try:
        response = requests.post(
            f"{TF_SERVING_URL}/v1/models/rl_trader:predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()['outputs'][0]
    except Exception as e:
        print(f"TF Serving error: {str(e)}")
        # Fallback to local model
        return app.state.agent.model.predict(state)[0]

@app.post("/predict")
def predict(request: PredictionRequest):
    state = np.array(request.state, dtype=np.float32).reshape(1, -1)
    scaled_state = app.state.scaler.transform(state)
    
    # Use TensorFlow Serving for predictions
    q_values = predict_via_tf_serving(scaled_state)
    
    action = np.argmax(q_values)
    return {"action": int(action)}