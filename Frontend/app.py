import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import os
# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("RL Stock Trading Dashboard")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'reliance': 0,
        'sbin': 0,
        'infy': 0,
        'cash': 20000
    }

# Real-time Prediction Section
st.header("Real-time Trading")
with st.form("prediction_form"):
    st.subheader("Current Portfolio")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        reliance_shares = st.number_input("RELIANCE Shares", 
                                         min_value=0, 
                                         value=st.session_state.portfolio['reliance'])
    with col2:
        sbin_shares = st.number_input("SBIN Shares", 
                                      min_value=0, 
                                      value=st.session_state.portfolio['sbin'])
    with col3:
        infy_shares = st.number_input("INFY Shares", 
                                      min_value=0, 
                                      value=st.session_state.portfolio['infy'])
    
    cash = st.number_input("Cash (₹)", 
                           min_value=0.0, 
                           value=st.session_state.portfolio['cash'],
                           step=1000.0)
    
    date = st.date_input("Select Date", datetime.today())
    
    submit = st.form_submit_button("Get Trading Action")
    
    if submit:
        # Update session state
        st.session_state.portfolio = {
            'reliance': reliance_shares,
            'sbin': sbin_shares,
            'infy': infy_shares,
            'cash': cash
        }
        
        # Get market features
        try:
            # Get market features (implementation below)
            market_features = get_market_features(date)
            
            # Prepare request
            request_data = {
                "state": market_features,
                "holdings": [reliance_shares, sbin_shares, infy_shares],
                "cash": cash
            }
            
            # Get prediction
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json=request_data
            )
            
            if response.status_code == 200:
                action = response.json()["action"]
                action_map = {
                    0: "SELL ALL",
                    1: "HOLD",
                    2: "BUY"
                }
                st.success(f"Recommended Action: **{action_map.get(action, 'UNKNOWN')}**")
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

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
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(portfolio_values)
            ax.set_title("Portfolio Value Over Time")
            ax.set_xlabel("Trading Step")
            ax.set_ylabel("Portfolio Value (₹)")
            ax.grid(True)
            ax.ticklabel_format(style='plain', axis='y')
            st.pyplot(fig)
        else:
            st.error(f"Simulation failed: {response.text}")

# Market feature extraction
def get_market_features(date):
    """Get the 16 market features for a given date"""
    def get_stock_features(ticker, label, end_date):
        end_date = pd.Timestamp(end_date)
        start_date = end_date - pd.Timedelta(days=30)
        df = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1))
        
        if df.empty:
            raise ValueError(f"No data for {ticker} on {end_date}")
        
        # Calculate features
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_ratio'] = df['Close'] / df['SMA_20']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Return last available values
        return df[['log_return', 'SMA_ratio', 'RSI_14', 'MACD_hist']].iloc[-1].values.tolist()
    
    # Get features for all required stocks
    features = []
    for ticker, label in [('RELIANCE.NS', 'RELIANCE'), 
                         ('SBIN.NS', 'SBIN'), 
                         ('INFY.NS', 'INFY'),
                         ('^NSEI', 'NSEI')]:
        features += get_stock_features(ticker, label, date)
    
    return features