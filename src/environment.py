import pandas as pd
import numpy as np
from datetime import datetime
import itertools
from collections import deque
def get_data():
  df=pd.read_csv('data/Train 2015-22.csv')
  print(df.head())
  return df.values
def get_test_data():
  df=pd.read_csv('data/Test 2023-24.csv')
  print(df.head())
  return df.values
# This list contains the names of all columns we want the agent to see.
feature_columns = [
    # RELIANCE Indicators
    'log_return_RELIANCE', 'SMA_ratio_RELIANCE', 'RSI_14_RELIANCE',
    'MACD_hist_RELIANCE', 'Volume_ratio_RELIANCE', 'BB_Position_RELIANCE',

    # INFY Indicators
    'log_return_INFY', 'SMA_ratio_INFY', 'RSI_14_INFY',
    'MACD_hist_INFY', 'Volume_ratio_INFY', 'BB_Position_INFY',

    # SBIN Indicators
    'log_return_SBIN', 'SMA_ratio_SBIN', 'RSI_14_SBIN',
    'MACD_hist_SBIN', 'Volume_ratio_SBIN', 'BB_Position_SBIN',

    # NIFTY Index Indicators
    'log_return_NSEI', 'SMA_ratio_NSEI', 'RSI_14_NSEI',
    'MACD_hist_NSEI', 'Volume_ratio_NSEI', 'BB_Position_NSEI',

    # Time/Seasonality Features
    'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos'
]
rsi_columns = [
        'RSI_14_RELIANCE','RSI_14_INFY','RSI_14_SBIN'
]
def get_feature_indices():
  df=pd.read_csv('data/Test 2023-24.csv')
  final_feature_indices = [df.columns.get_loc(c) for c in feature_columns]
  return final_feature_indices
def get_rsi_indices():
  df=pd.read_csv('data/Test 2023-24.csv')
  final_feature_indices = [df.columns.get_loc(c) for c in rsi_columns]
  return final_feature_indices

class MultiStockEnv:
  """
  A 3-stock trading environment.
  State: vector of size 32 ((n_stock+1(NSEI)) * 6 + 4(Time/Seasonality features)+4(portfolio))
    # portfolio constituents:
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - cash owned (can be used to purchase more stocks)
  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
  """
  def __init__(self,data,alpha,beta,initial_investment=20000,feature_indices=get_feature_indices(),rsi_index=get_rsi_indices()):
    #data
    self.stock_price_history=data
    self.n_step=self.stock_price_history.shape[0]
    self.n_stock=3
    self.alpha=alpha
    self.beta=beta
    self.use_dd_penalty = False
    self.sharpe=0.0
    self.initial_investment=initial_investment
    self.peak_val = initial_investment
    self.price_indices = [0, 15, 30]
    self.rsi_index=rsi_index
    print(f"Data shape: {data.shape}")
    print(f"Price indices: {self.price_indices}")
    print(f"Sample prices: {data[0, self.price_indices]}")
    self.curr_step=None
    self.stock_owned=None
    self.stock_price=None
    self.cash_in_hand=None
    self.state_feature_indices = feature_indices
    self.action_space=np.arange(3**self.n_stock)
    # action permutations
    # returns a nested list with elements like:
    # [0,0,0]
    # [0,0,1]
    # [0,0,2]
    # [0,1,0]
    # [0,1,1]
    # etc.
    # 0 = sell
    # 1 = hold
    # 2 = buy
    self.action_list=list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
    self.returns_window = deque(maxlen=21)
    self.window_size=21
    self.state_history=deque(maxlen=self.window_size)
    self.last_rsi_mean = 50.0  # For market quality default
    self.feature_step_dim=4+len(self.state_feature_indices)
    self.state_dim = (self.window_size,self.feature_step_dim)
    self.reset()

  def reset(self, mode='train'):
    if mode == 'train':
        # Random start for training to prevent memorization
        self.curr_step = np.random.randint(0,self.n_step-21)
    else: 
        # Always start at 0 for validation/testing
        self.curr_step = 0

    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.curr_step, self.price_indices]
    self.cash_in_hand = self.initial_investment
    self.peak_val = self.initial_investment
    self.returns_window.clear()
    self.state_history.clear()
    current_state=self._get_current_state()
    for i in range(self.window_size):
      self.state_history.append(current_state)
    obs = self._get_obs()
    assert obs.shape == self.state_dim, f"obs shape {obs.shape} != {self.state_dim}"
    return obs
  def set_normalization_params(self, return_mean, return_std, sharpe_mean, sharpe_std):
    self.norm_return_mean = return_mean
    self.norm_return_std = return_std
    self.norm_sharpe_mean = sharpe_mean
    self.norm_sharpe_std = sharpe_std

  @staticmethod
  def get_slippage_pct():
    # 70% chance of negative slippage (unfavorable)
    return np.random.choice(
        [np.random.uniform(0, 0.003), np.random.uniform(-0.001, 0)],
        p=[0.7, 0.3]
    )
  
  @staticmethod
  def calculate_transaction_cost(price, quantity, trade_type='delivery', broker='zerodha'):
    
    trade_value = price * quantity

    # --- Cost Rates ---
    brokerage_rate = 0.001  
    stt_rate = 0.001 if trade_type == 'delivery' else 0.00025
    exchange_txn_rate = 0.0000325
    sebi_fee_rate = 0.000001
    stamp_duty_rate = 0.00015  # Approximate
    gst_rate = 0.18  # On brokerage + exchange txn + sebi

    # --- Cost Components ---
    brokerage = min(trade_value * brokerage_rate, 20.0) if broker == 'zerodha' else trade_value * brokerage_rate
    stt = trade_value * stt_rate
    exchange_txn = trade_value * exchange_txn_rate
    sebi_fee = trade_value * sebi_fee_rate
    stamp_duty = trade_value * stamp_duty_rate
    gst = gst_rate * (brokerage + exchange_txn + sebi_fee)

    total_cost = brokerage + stt + exchange_txn + sebi_fee + stamp_duty + gst
    return total_cost
  
  @staticmethod
  def sharpe_ratio(returns, risk_free_rate=0.05):
    if len(returns) < 5:
        return 0.0

    # excess returns over the daily risk-free rate
    excess_returns = np.array(returns) - risk_free_rate / 252 # 252 trading days in a year

    # annualized sharpe ratio
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns) + 1e-8

    return (mean_return / std_return) * np.sqrt(252)
  
  def step(self,action):
    assert action in self.action_space 
    prev_val=self._get_val()
    #perform the trade
    self._trade(action)
    done=self.curr_step==self.n_step-1 
    self.curr_step+=1
    if not done:
        self.stock_price = self.stock_price_history[self.curr_step, self.price_indices]
    curr_val=self._get_val()
    self.peak_val = max(self.peak_val, curr_val)
    portfolio_return = (curr_val - prev_val) / prev_val 
    self.returns_window.append(portfolio_return)
    self.sharpe= self.sharpe_ratio(self.returns_window)
    reward =self.findReward(portfolio_return,self.sharpe)
    self.state_history.append(self._get_current_state())
    info={'curr_val':curr_val}
    next_obs = self._get_obs()
    assert next_obs.shape == self.state_dim, f"obs shape {next_obs.shape} != {self.state_dim}"
    return next_obs,reward,done,info
  
  def findReward(self, portfolio_return, sharpe):
    if np.isnan(portfolio_return) or np.isnan(sharpe):
        return 0.0

    return_reward = portfolio_return * 100
    sharpe_reward = np.tanh(sharpe)
    cash_ratio = self.cash_in_hand / (self._get_val() + 1e-8)
    if self.curr_step < self.n_step:
        rsi_indices =self.rsi_index
        rsi_values = self.stock_price_history[self.curr_step, rsi_indices]
        total_owned = np.sum(self.stock_owned)
        weights = np.ones(3) / 3 if total_owned < 0.1 else self.stock_owned / total_owned
        market_quality = np.clip(np.average(rsi_values, weights=weights), 30, 70)
        self.last_rsi_mean = market_quality
    else:
        market_quality = self.last_rsi_mean

    cash_penalty = 2.5 * cash_ratio * ((market_quality - 50) / 50)
    dd_penalty = 0
    if self.peak_val > 0:
      drawdown = max(0, (self.peak_val - self._get_val()) / self.peak_val)
      dd_penalty = 1.5 * drawdown

    reward = (
        self.alpha * return_reward +
        self.beta * sharpe_reward -
        cash_penalty -
        dd_penalty
    )
    return np.clip(reward, -10, 10)
  
  def _get_current_state(self):
    # Portfolio status
    portfolio = [
    self.stock_owned[0] / 100,  # ~0-5 range for typical holdings
    self.stock_owned[1] / 100,
    self.stock_owned[2] / 100,
    self.cash_in_hand / self.initial_investment  # 0-2 range
    ]
    if self.curr_step < self.n_step:
        features = self.stock_price_history[self.curr_step, self.state_feature_indices]
    else:
        features = self.stock_price_history[self.curr_step - 1, self.state_feature_indices]
    state = np.array(portfolio + features.tolist(), dtype=np.float32)
    return state
  
  def _get_obs(self):
    return np.array(self.state_history)
  
  def _get_val(self):
    return self.stock_owned.dot(self.stock_price)+self.cash_in_hand

  def _trade(self,action):
    # index the action we want to perform
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # e.g. [2,1,0] means:
    # buy first stock
    # hold second stock
    # sell third stock
    if self.curr_step >= self.n_step - 1:  
        return
    action_vec = self.action_list[action]
    sell_index=[] #stores indeces of stocks we want to sell
    buy_index=[]  #stores indeces of stocks we want to buy
    for i,a in enumerate(action_vec):
      if(a==0):
        sell_index.append(i)
      elif(a==2):
        buy_index.append(i)
    slippage_pct = self.get_slippage_pct()
    if sell_index:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        effective_price = self.stock_price[i] * (1 - slippage_pct)
        cost = self.calculate_transaction_cost(effective_price, self.stock_owned[i])
        self.cash_in_hand += effective_price * self.stock_owned[i] - cost
        self.stock_owned[i] = 0
    if buy_index:
       max_iterations = 1000 
       iteration = 0
       while iteration < max_iterations:
        bought_any = False
        for i in buy_index:
            effective_price = self.stock_price[i] * (1 + slippage_pct)
            cost = self.calculate_transaction_cost(effective_price, 1)
            total_cost = effective_price + cost
            if self.cash_in_hand >= total_cost:
                self.stock_owned[i] += 1
                self.cash_in_hand -= total_cost
                bought_any = True
        if not bought_any:
            break
        iteration += 1