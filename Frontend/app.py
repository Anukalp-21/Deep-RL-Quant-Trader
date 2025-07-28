import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np

# Backend configuration
BACKEND_URL = "http://localhost:8000"  # Update if deployed

st.title("RL Stock Trading Dashboard")

# Real-time Prediction Section
st.header("Real-time Trading")
with st.form("prediction_form"):
    st.subheader("Enter Market State")
    
    # Create input fields based on your state dimensions
    col1, col2 = st.columns(2)
    with col1:
        reliance_owned = st.number_input("RELIANCE Shares", value=0.0)
        infy_owned = st.number_input("INFY Shares", value=0.0)
        sbin_owned = st.number_input("SBIN Shares", value=0.0)
        cash = st.number_input("Cash", value=20000.0)
    
    with col2:
        # Add technical indicators (simplified example)
        reliance_rsi = st.slider("RELIANCE RSI", 0, 100, 50)
        nifty_macd = st.number_input("NIFTY MACD", value=0.0)
        market_quality = st.slider("Market Quality", 0, 100, 50)
    
    submit = st.form_submit_button("Get Trading Action")
    
    if submit:
        # Construct state vector (match your environment's state structure)
        state = [
            reliance_owned/100,
            infy_owned/100,
            sbin_owned/100,
            cash/20000,
            # Add other technical indicators here...
            0.02,  # Example: RELIANCE log return
            1.05,  # Example: RELIANCE SMA ratio
            reliance_rsi,
            0.1,   # Example: RELIANCE MACD hist
            # ... repeat for other stocks and indicators
        ]
        
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"state": state}
        )
        
        if response.status_code == 200:
            action = response.json()["action"]
            action_map = {
                0: "SELL ALL",
                1: "HOLD",
                2: "BUY"
            }
            st.success(f"Recommended Action: **{action_map[action]}**")
        else:
            st.error("Prediction failed")

# Test Simulation Section
st.header("Test Trading Strategy")
if st.button("Run Full Simulation"):
    with st.spinner("Running test episode..."):
        response = requests.get(f"{BACKEND_URL}/run-test-episode")
        
        if response.status_code == 200:
            result = response.json()
            final_value = result["final_value"]
            portfolio_values = result["portfolio_values"]
            
            st.subheader(f"Final Portfolio Value: ₹{final_value:,.2f}")
            
            # Plot results
            fig, ax = plt.subplots()
            ax.plot(portfolio_values)
            ax.set_title("Portfolio Value Over Time")
            ax.set_xlabel("Trading Step")
            ax.set_ylabel("Portfolio Value (₹)")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.error("Simulation failed")

# Training Control Section (if implemented)
st.header("Model Training")
if st.button("Retrain Model"):
    with st.spinner("Training in progress..."):
        # This would call a training endpoint if implemented
        st.warning("Training endpoint not implemented in this example")