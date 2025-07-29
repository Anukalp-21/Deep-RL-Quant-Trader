import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np

import yfinance as yf
import pandas as pd
import numpy as np

def preprocess_stock(ticker, label,end_date_str):
    # Download data
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.Timedelta(days=30)
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=(end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    df = df[['Close', 'Volume']].copy()
    df.columns = [f'{label}_Close', f'{label}_Volume']

    # Log return
    df[f'log_return_{label}'] = np.log(df[f'{label}_Close'] / df[f'{label}_Close'].shift(1)).fillna(0)

    # SMA and SMA ratio
    df[f'SMA_20_{label}'] = df[f'{label}_Close'].rolling(window=20).mean()
    df[f'SMA_ratio_{label}'] = df[f'{label}_Close'] / df[f'SMA_20_{label}']

    # RSI
    delta = df[f'{label}_Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df[f'RSI_14_{label}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df[f'{label}_Close'].ewm(span=12, adjust=False).mean()
    ema26 = df[f'{label}_Close'].ewm(span=26, adjust=False).mean()
    df[f'MACD_{label}'] = (ema12 - ema26)
    df[f'MACD_signal_{label}'] = df[f'MACD_{label}'].ewm(span=9, adjust=False).mean()
    df[f'MACD_hist_{label}'] = df[f'MACD_{label}'] - df[f'MACD_signal_{label}']

    # Drop warm-up rows
    df.dropna(inplace=True)

    return df

def state(date):
    reliance_df=preprocess_stock('RELIANCE.NS', 'RELIANCE',date)
    sbin_df = preprocess_stock('SBIN.NS', 'SBIN',date)
    infy_df = preprocess_stock('INFY.NS', 'INFY',date)
    multiStock_df=pd.merge(reliance_df,infy_df,left_index=True,right_index=True)
    multiStock_df=pd.merge(multiStock_df,sbin_df,left_index=True,right_index=True)
    nifty_df = preprocess_stock('^NSEI', 'NSEI',date)
    nifty_df=nifty_df[['log_return_NSEI','SMA_20_NSEI','SMA_ratio_NSEI','RSI_14_NSEI','MACD_NSEI','MACD_signal_NSEI','MACD_hist_NSEI']]
    aligned_df = multiStock_df.merge(nifty_df, left_index=True, right_index=True, how='inner')
    return aligned_df[-1:]

# Backend configuration
BACKEND_URL = "http://localhost:8000"  # Update if deployed

st.title("RL Stock Trading Dashboard")

# Real-time Prediction Section
st.header("Real-time Trading")
with st.form("prediction_form"):
    st.subheader("Enter Market State")
    
    date = st.date_input("Select Date", pd.to_datetime("today"))
    
    submit = st.form_submit_button("Get Trading Action")
    
    if submit:
        # Construct state vector (match your environment's state structure)
        state =np.array(state(date))
        state = state.values.flatten().tolist()
        if state.empty:
            st.error("No data available for the selected date.")
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