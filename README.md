# 🤖 Deep Reinforcement Learning Quantitative Trading Agent

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Validated-success)

A **systematic quantitative trading agent** powered by **Double DQN (DDQN)** and **LSTM** networks. This project implements a custom Gymnasium-style environment to autonomously trade a multi-asset portfolio (RELIANCE, INFY, SBIN) using advanced policy optimization, macro-regime filtering, and realistic market friction simulation.

## 🚀 Key Performance Highlights

In quantitative finance, the ultimate test is out-of-sample generalization. The agent was rigorously validated across distinct macroeconomic regimes to ensure it learned structural market mechanics rather than overfitting to historical noise.

| Metric | 📚 In-Sample (2015-2021) | 🐻 Bear Market (2022) | 🐂 Bull Market (2023-24) | 🛡️ Forward Walk (2025) |
| :--- | :--- | :--- | :--- | :--- |
| **Market Condition** | 7-Year Historical Gauntlet | High Inflation / Rate Hikes | Strong Uptrend | Choppy / Regime Shift |
| **Total Return (%)** | **+358.9%** (₹20k ➔ ₹91.7k) | **+31.5%** | **+86.0%** | **+14.3%** |
| **Annualized (CAGR)** | **~24.3%** | N/A (1-Year) | **~36.3%** | N/A (1-Year) |
| **Agent Sharpe Ratio** | **0.78** (7-Year Gauntlet) | **1.01** | **1.43** | **0.56** (H2 Peak: **2.18**) |
| **Strategy Behavior** | Baseline Exploration | Capital Preservation | Aggressive Compounding | Defensive ➔ Trend-Following |

> **Verification:** See the `backtests/1_logs/` and `backtests/3_plots/` folders for detailed step-by-step transaction logs, Monte Carlo simulations, and equity curve distributions.

## 🛠️ Tech Stack & Architecture

* **Core:** Python, TensorFlow, Keras, Pandas, Streamlit
* **Model:** Double DQN (DDQN) with an **LSTM (64 units)** backbone for time-series memory retention.
* **Regularization:** `GaussianNoise(0.1)` and `Dropout(0.3)` to prevent curve-fitting on high-noise financial data.
* **Optimization:** **Prioritized Experience Replay (PER)** using a custom `O(log N)` SumTree data structure to focus training on high-TD-error events.
* **State Space (31 dims):** Portfolio Weights + Technicals (MACD, RSI, BB) + **Macro Regime Indicators (India VIX, NIFTY SMA_200)**.

## 📐 Mathematical Framework

The agent optimizes a custom **Risk-Adjusted Reward Function** that dynamically balances absolute profit, volatility smoothing, and capital preservation.

**Total Reward Equation:**

> Rₜ = 5.0 × tanh( 100·rₜ + 0.1·ΔSharpeₜ - 10.0·max(0, ΔDrawdownₜ) )

**Where:**

* **rₜ (Returns):** Portfolio returns bounded by a scaling factor.
* **ΔSharpeₜ:** The differential Sharpe Ratio. Rewards the agent for smoothing out the volatility of its equity curve.
* **ΔDrawdownₜ (Delta Drawdown Penalty):** Strictly penalizes the agent for creating *new* losses from the all-time high.
* **tanh():** Hyperbolic tangent squashes the final reward between `[-5, 5]`, preventing gradient explosions during highly volatile market epochs.

## 📉 Realistic Market Simulation

To prevent "paper trading bias," the environment mathematically enforces real-world Indian market friction:

* **Stochastic Slippage:** 70% probability of adverse execution (0-30 bps negative slippage).
* **Transaction Costs:** Calculates exact Zerodha Brokerage, STT (0.1%), Exchange Txn charges, SEBI fees, Stamp Duty, and 18% GST on every trade.
* **Capital Sizing:** Dynamically calculates maximum allowable shares based on real-time cash balances and simulated margin.

## 🧠 Engineering Challenges & Solutions

### 1. Concept Drift & The "Falling Knife" Trap

* **Problem:** Standard RL bots trained in bull markets learn to aggressively "Buy the Dip." During out-of-sample stress tests (like the 2022 crash), this logic fails catastrophically, causing the bot to catch falling knives.
* **Solution:** Engineered **Macro-Regime Filters** into the state vector, specifically the **India VIX (Turbulence Index)** and **NIFTY SMA_200 Ratio**. This granted the LSTM macro-awareness, allowing it to mathematically differentiate between a safe "bull market dip" and a dangerous "bear market crash."

### 2. The "Memoryless" Agent (Architecture Search)

* **Problem:** Early experiments using standard Multi-Layer Perceptrons (MLP) treated price points as isolated events, leading to severe overfitting. The bot could not understand sequences or momentum.
* **Solution:** Migrated the Q-Network architecture to an **LSTM (Long Short-Term Memory)** backbone. This allowed the agent to maintain a hidden state of historical price action, effectively letting it "remember" volatility clustering rather than just reacting to the current spot price.

### 3. Alpha Decay & Non-Stationarity (The 2025 Problem)

* **Problem:** Financial markets evolve. A model trained on 2015-2021 data inherently suffers from alpha decay when predicting 2025 markets due to shifts in retail participation and changing interest rate regimes.
* **Solution:** Conducted a rigorous 4-year Forward Walk analysis. Instead of catastrophically failing, the frozen model exhibited graceful degradation (0.56 overall Sharpe), successfully recognizing early 2025 chop to preserve capital, before seamlessly pivoting to capture a massive H2 trend with an elite 2.18 Sharpe.

### 4. The "Revenge Trading" Drawdown Bug

* **Problem:** Early reward functions penalized the agent based on absolute drawdown. When the market crashed and the bot safely retreated to 100% Cash, it continued receiving negative rewards every day. This inadvertently trained the bot to "revenge trade" back into crashing markets.
* **Solution:** Refactored the reward function to penalize **Delta Drawdown** (new, active losses). Sitting in cash yields a neutral 0 penalty, successfully teaching the neural network the concept of Capital Preservation.

### 5. Deterministic Illusion (Monte Carlo Variance)

* **Problem:** Standard backtests assume static slippage, creating deterministic illusions of profitability. A model might be profitable by sheer luck of a single execution path.
* **Solution:** Wrapped the evaluation script in a Monte Carlo Simulation module. The testing suite runs multiple independent paths, sampling from a custom stochastic probability distribution for slippage. Final metrics are aggregated averages, mathematically proving the strategy's edge survives extreme execution variance.

### 6. The Sparse Reward Problem

* **Problem:** In a multi-asset environment with 27 possible discrete actions, the agent struggled to converge because positive feedback was too infrequent.
* **Solution:** Implemented **Prioritized Experience Replay (PER)** from scratch. By using a binary SumTree to sample high-TD-error transitions, the agent learned from "surprising" market events significantly faster than uniform random sampling.

## 📂 Project Structure

```text
├── backtests/                 # Validation Artifacts & Output
│   ├── 1_logs/                # txt files (training, train_test, bear, bull, 2025 H1/H2/Overall)
│   ├── 2_raw_data/            # npy arrays (portfolio histories for Streamlit/Plots)
│   └── 3_plots/               # png distribution & equity charts
├── data/                      # Historical Market Data (Train, Test, Val CSVs)
├── models/                    # Serialized Agents (.h5) & Pickled Scalers (.pkl)
├── src/                       # Core Strategy Logic
│   ├── __init__.py
│   ├── agent.py               # DDQN Agent with PER logic
│   ├── environment.py         # Custom Gymnasium Trading Env
│   └── model.py               # LSTM Network Architecture
├── .gitignore
├── main.py                    # Entry Point (Training & Evaluation Loop)
├── requirements.txt           # Python Dependencies
└── README.md                  # Project Documentation

```

## ⚡ How to Run

### 1. Prerequisite

Ensure you have Python 3.8+ and TensorFlow installed.

### 2. Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/Anukalp-21/Deep-RL-Quant-Trader.git
cd Deep-RL-Quant-Trader
pip install -r requirements.txt

```

### 3. Execution (Terminal)

To run the evaluation via Monte Carlo Simulation:

```bash
python main.py --mode test

```

*(To initiate a new training loop, use `python main.py --mode train`).*

