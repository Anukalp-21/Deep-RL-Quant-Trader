# 🤖 Deep Reinforcement Learning Quantitative Trading Agent

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Validated-success)

A high-frequency quantitative trading agent powered by **Double DQN (DDQN)** and **LSTM** networks. This project implements a custom Gymnasium environment to autonomously trade a multi-asset portfolio (RELIANCE, INFY, SBIN) using advanced policy optimization and realistic market simulation.

## 🚀 Key Performance Highlights

The agent was rigorously validated using a **Monte Carlo Simulation (N=100 Episodes)** to ensure statistical significance.

| Metric | 🐂 Bull Market (2023-24) | 🛡️ Stress Test (2025) |
| :--- | :--- | :--- |
| **Market Condition** | Strong Uptrend | Volatile / Choppy Regime |
| **Benchmark Return** | +31.65% | +15.39% |
| **Bot Return (Cumulative)** | **+71.70%** 🚀 | **+16.80%** ✅ |
| **Alpha (Edge)** | **+40.05%** | **+1.41%** |
| **Strategy Behavior** | Aggressive Compounding | Capital Preservation |

> **Verification:** See `backtests/` folder for detailed logs and distribution histograms.

## 📊 Monte Carlo Validation
Unlike standard backtests that run once, this system was tested on **100 independent episodes** to map the probability distribution of returns.

* **Bull Market Distribution:** consistently converges on high returns (~34k portfolio value).
* **Stress Test Distribution:** Exhibits bimodal behavior, intelligently switching to "Cash Preservation" mode during high volatility to protect gains.

## 🛠️ Tech Stack & Architecture
* **Core:** Python, TensorFlow, Keras, Gymnasium
* **Model:** Double DQN (DDQN) with **LSTM layers** for time-series memory.
* **Optimization:** **Prioritized Experience Replay (PER)** using SumTree data structures to focus training on high-error events.
* **Risk Management:** Sharpe Ratio optimization + Maximum Drawdown penalties using Curriculum Learning.

## 📉 Realistic Market Simulation
To prevent "paper trading bias," the environment models real-world Indian market friction:
* **Stochastic Slippage:** 70% probability of adverse execution (0-30 bps).
* **Transaction Costs:** Includes STT (0.1%), Brokerage, and Stamp Duty on every trade.
* **Liquidity Constraints:** Simulates partial fills and volume limits.

## 📂 Project Structure
```text
├── backtests/                 # PROOF OF RESULTS
│   ├── 2023-24_Bull_Market_Performance.png
│   ├── 2025_Stress_Test_Volatile_Regime.png
│   ├── Logs_2023-24_Bull_Market.txt
│   └── Logs_2025_Stress_Test.txt
├── src/                       # Source Code
│   ├── agent.py               # DDQN Agent (TensorFlow + PER)
│   ├── environment.py         # Custom Multi-Stock Gym Environment
│   └── main.py                # Training & Testing Loop
├── models/                    # Pre-trained Weights
└── README.md
```
## ⚡ How to Run
### 1. Prerequisite
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone the repo and install dependencies:
```bash
git clone [https://github.com/Anukalp-21/Deep-RL-Quant-Trader.git](https://github.com/Anukalp-21/Deep-RL-Quant-Trader.git)
cd Deep-RL-Quant-Trader
pip install -r requirements.txt
```

### 3. Execution
To start the training or testing loop:
```bash
python main.py
```