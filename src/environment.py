import pandas as pd
import numpy as np
from datetime import datetime
import itertools
from collections import deque
def get_data():
  df=pd.read_csv('/content/MultiStock (3).csv')
  print(df.head())
  return df.values
def get_test_data():
  df=pd.read_csv('/content/MultiStock_test1.csv')
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
  df=pd.read_csv('/content/MultiStock (3).csv')
  final_feature_indices = [df.columns.get_loc(c) for c in feature_columns]
  return final_feature_indices
def get_rsi_indices():
  df=pd.read_csv('/content/MultiStock (3).csv')
  final_feature_indices = [df.columns.get_loc(c) for c in rsi_columns]
  return final_feature_indices
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1


    # In the SumTree class
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
class MultiStockEnv:
  """
  A 3-stock trading environment.
  State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
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
    #instance attributes
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
    #calculate size of state
    self.feature_step_dim=4+len(self.state_feature_indices)
    self.state_dim = (self.window_size,self.feature_step_dim)
    self.reset()
  # In your MultiStockEnv class

  def reset(self, mode='train'): # MODIFIED: Accept a mode
    # MODIFIED: Choose start step based on the mode
    if mode == 'train':
        # Random start for training to prevent memorization
        # This is the new way: only starting within the first year
        self.curr_step = np.random.randint(0,self.n_step-21)
        # self.curr_step = np.random.randint(0,256)
    else: # 'test' mode
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
    """
    Calculates detailed transaction costs for a trade.

    Parameters:
        price (float): Execution price per share
        quantity (int): Number of shares traded
        trade_type (str): 'delivery' or 'intraday'
        broker (str): Broker name (for flat fee logic, if needed)

    Returns:
        total_cost (float): Total transaction cost in ₹
        breakdown (dict): Dictionary of individual cost components
    """
    trade_value = price * quantity

    # --- Cost Rates ---
    brokerage_rate = 0.001  # 0.1%
    stt_rate = 0.001 if trade_type == 'delivery' else 0.00025
    exchange_txn_rate = 0.0000325
    sebi_fee_rate = 0.000001
    stamp_duty_rate = 0.00015  # Approximate, varies by state
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

    # Calculate excess returns over the daily risk-free rate
    excess_returns = np.array(returns) - risk_free_rate / 252 # 252 trading days in a year

    # Calculate annualized sharpe ratio
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns) + 1e-8

    return (mean_return / std_return) * np.sqrt(252)
  def step(self,action):
    assert action in self.action_space #- It checks at runtime whether the action is a valid member of the environment’s action_space.
    #- If the assertion fails (i.e., action is not in self.action_space), Python raises an AssertionError and halts the program.

    # get current value before performing the action.this is the the total money=each stock share price*shares of each stock+cash in hand (current portfolio)
    #like say my shares for each stocks are [10->shares of apple,1->share of msi,1->share of SBUX] and these are corresponding stock prices [2,3,1]
    #and 10000 in hand cash,then value=10*2+1*3+1*1+10000=10024
    prev_val=self._get_val()
    #perform the trade
    self._trade(action)

    # check if we are at the end of the data
    done=self.curr_step==self.n_step-1 # Check before incrementing

    # update price, i.e. go to the next day
    self.curr_step+=1

    # If not done, get the stock prices for the next step
    if not done:
        self.stock_price = self.stock_price_history[self.curr_step, self.price_indices]
    #get the new value after taking the action
    curr_val=self._get_val()
    self.peak_val = max(self.peak_val, curr_val)
    # At each step
    portfolio_return = (curr_val - prev_val) / prev_val  # Simple return
    self.returns_window.append(portfolio_return)
    self.sharpe= self.sharpe_ratio(self.returns_window)
    # Combined reward
    reward =self.findReward(portfolio_return,self.sharpe)
    self.state_history.append(self._get_current_state())
    #done if we have reached out of data or end of episode
    #store the curr value pof portfolio here
    info={'curr_val':curr_val}
    #conform to the Gym API
    next_obs = self._get_obs()
    # validate obs shape again
    assert next_obs.shape == self.state_dim, f"obs shape {next_obs.shape} != {self.state_dim}"
    return next_obs,reward,done,info
  # def findReward(self,portfolio_return,sharpe):
  #   # Normalize
  #   return_norm = (portfolio_return - self.norm_return_mean) / self.norm_return_std
  #   sharpe_norm = (sharpe - self.norm_sharpe_mean) / self.norm_sharpe_std
  #   cash_penalty = 0.1 * (self.cash_in_hand / self.initial_investment)
  #   reward = (alpha * return_norm + beta * sharpe_norm) - cash_penalty
  #   if self._get_val() < self.initial_investment * 0.8:
  #     reward -= 10
  #   return reward
  def findReward(self, portfolio_return, sharpe):
    # Handle edge cases for stability
    if np.isnan(portfolio_return) or np.isnan(sharpe):
        return 0.0

    # 1. SCALED Return Component
    # Changed from log return and simplified. Now, a 1% daily return (0.01) directly
    # translates to a reward of 1.0, making its scale comparable to the Sharpe reward.
    return_reward = portfolio_return * 100

    # 2. SCALED Sharpe Component
    # This calculation is unchanged. np.tanh() is an excellent method that already
    # squashes the Sharpe ratio into a stable [-1, 1] range.
    sharpe_reward = np.tanh(sharpe)

    # 3. SCALED Cash Penalty
    # The original penalty was too small. This multiplier makes it more impactful.
    # It now operates in a more meaningful ~[-0.5, 0.5] range.
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

    # 4. SCALED Drawdown Penalty
    # The original penalty was also too small. With a multiplier of 5.0, a 20%
    # drawdown now results in a significant -1.0 penalty, making risk management a priority.
    dd_penalty = 0
    if self.peak_val > 0:
      drawdown = max(0, (self.peak_val - self._get_val()) / self.peak_val)
      dd_penalty = 1.5 * drawdown

    # 5. Combine the BALANCED components
    # With all components on a similar scale, your dynamic alpha and beta
    # hyperparameters will now correctly shift the agent's focus during training.
    reward = (
        self.alpha * return_reward +
        self.beta * sharpe_reward -
        cash_penalty -
        dd_penalty
    )

    # Clipping the final reward is a good safety measure for training stability.
    return np.clip(reward, -10, 10)
  def _get_current_state(self):
    # Portfolio status
    portfolio = [
    self.stock_owned[0] / 100,  # Now ~0-5 range for typical holdings
    self.stock_owned[1] / 100,
    self.stock_owned[2] / 100,
    self.cash_in_hand / self.initial_investment  # Now 0-2 range
    ]
    # Technical + macro indicators
    # Check if curr_step is within bounds before accessing stock_price_history
    if self.curr_step < self.n_step:
        features = self.stock_price_history[self.curr_step, self.state_feature_indices]
    else:
        # If at the end, use the last available features (or handle as needed)
        features = self.stock_price_history[self.curr_step - 1, self.state_feature_indices]
    # Combine into final state vector
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
    if self.curr_step >= self.n_step - 1:  # Prevent out-of-bound trading
        return
    action_vec = self.action_list[action]
    # determine which stocks to buy or sell
    sell_index=[] #stores indeces of stocks we want to sell
    buy_index=[]  #stores indeces of stocks we want to buy
    for i,a in enumerate(action_vec):
      if(a==0):
        sell_index.append(i)
      elif(a==2):
        buy_index.append(i)
    # sell any stocks we want to sell
    # then buy any stocks we want to buy
    slippage_pct = self.get_slippage_pct()
    if sell_index:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        effective_price = self.stock_price[i] * (1 - slippage_pct)
        cost = self.calculate_transaction_cost(effective_price, self.stock_owned[i])
        self.cash_in_hand += effective_price * self.stock_owned[i] - cost
        self.stock_owned[i] = 0
    if buy_index:
       max_iterations = 1000  # safeguard
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