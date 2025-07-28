import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import itertools
def get_data():
  df=pd.read_csv('/kaggle/input/ghgkkgcgh/MultiStock.csv')
  print(df.head())
  return df.values
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

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
class PrioritizedReplayBuffer:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment = 0.001

    def __init__(self, obs_dim, act_dim, size):
        self.tree = SumTree(size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def store(self, error, obs, act, rew, next_obs, done):
        data = (obs, act, rew, next_obs, done)
        p = self._get_priority(error)
        self.tree.add(p, data)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        batch = {'s': [], 's2': [], 'a': [], 'r': [], 'd': [], 'weights': [], 'indices': []}
        segment = self.tree.total() / batch_size
        

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            obs, act, rew, next_obs, done = data
            batch['s'].append(obs)
            batch['s2'].append(next_obs)
            batch['a'].append(act)
            batch['r'].append(rew)
            batch['d'].append(done)
            batch['indices'].append(idx)
            
            # Importance sampling weight
            sample_prob = p / self.tree.total()
            weight = (self.size * sample_prob) ** -self.beta
            batch['weights'].append(weight)

        # Normalize weights
        max_weight = max(batch['weights'])
        batch['weights'] = [w / max_weight for w in batch['weights']]
        
        # Convert to arrays
        batch['s'] = np.array(batch['s'])
        batch['s2'] = np.array(batch['s2'])
        batch['a'] = np.array(batch['a'])
        batch['r'] = np.array(batch['r'])
        batch['d'] = np.array(batch['d'])
        return batch
    def increment_beta(self):
        self.beta=np.min([1., self.beta + self.beta_increment])
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)
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
  def __init__(self,data,alpha,beta,initial_investment=20000):
    #data
    self.stock_price_history=data
    self.n_step=self.stock_price_history.shape[0]
    self.n_stock=3
    #instance attributes
    self.alpha=alpha
    self.beta=beta
    self.sharpe=0.0
    self.initial_investment=initial_investment
    self.peak_val = initial_investment
    self.price_indices = [0, 9, 18]
    print(f"Data shape: {data.shape}")
    print(f"Price indices: {self.price_indices}")
    print(f"Sample prices: {data[0, self.price_indices]}")
    self.curr_step=None
    self.stock_owned=None
    self.stock_price=None
    self.cash_in_hand=None
    self.state_feature_indices = [
    2, 4, 5, 8,    # RELIANCE: log_return, SMA_ratio, RSI, MACD_hist
    11, 13, 14, 17, # INFY: log_return, SMA_ratio, RSI, MACD_hist
    20, 22, 23, 26, # SBIN: log_return, SMA_ratio, RSI, MACD_hist
    27, 29, 30, 33  # NIFTY: log_return, SMA_ratio, RSI, MACD_hist
    ]
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
    self.returns_window = deque(maxlen=252)
    # NEW: Normalization parameters
    self.norm_return_mean = 0
    self.norm_return_std = 1
    self.norm_sharpe_mean = 0
    self.norm_sharpe_std = 1
    self.last_rsi_mean = 50.0  # For market quality default
    #calculate size of state
    self.state_dim=20
    self.reset()
  def reset(self):
    self.curr_step=0
    self.stock_owned=np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[0, self.price_indices]
    self.cash_in_hand=self.initial_investment
    self.peak_val = self.initial_investment 
    self.returns_window.clear()

    obs = self._get_obs()
    # validate obs shape
    assert obs.shape == (self.state_dim,), f"obs shape {obs.shape} != {self.state_dim}"
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
    
    excess_returns = np.array(returns) - risk_free_rate/252
    mean_retrn = np.mean(excess_returns)
    std_retrn = np.std(excess_returns)+1e-8 
    
    if std_retrn < 1e-8:
        return 0.0
        
    return mean_retrn / std_retrn * np.sqrt(252)
    
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
    #done if we have reached out of data or end of episode
    #store the curr value pof portfolio here
    info={'curr_val':curr_val}
    #conform to the Gym API
    next_obs = self._get_obs()
    # validate obs shape again
    assert next_obs.shape == (self.state_dim,), f"obs shape {next_obs.shape} != {self.state_dim}"
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
    # Handle edge cases
    if np.isnan(portfolio_return) or np.isnan(sharpe):
        return 0.0

    return_z = (portfolio_return - self.norm_return_mean) / self.norm_return_std
    sharpe_z = (sharpe - self.norm_sharpe_mean) /self.norm_sharpe_std
    # 2. Smart cash penalty
    cash_ratio = self.cash_in_hand / (self._get_val() + 1e-8)

    # Calculate market quality with safe weights
    if self.curr_step < self.n_step:
        rsi_indices = [5, 14, 23]
        rsi_values = self.stock_price_history[self.curr_step, rsi_indices]
        total_owned = np.sum(self.stock_owned)
        if total_owned < 0.1:
            weights = np.ones(3) / 3  # Equal weights if no holdings
        else:
            weights = self.stock_owned / total_owned
        market_quality = np.clip(np.average(rsi_values, weights=weights), 30, 70)
        self.last_rsi_mean = market_quality
    else:
        market_quality = self.last_rsi_mean

    cash_penalty = 0.5 * cash_ratio * ((market_quality - 50) / 50)
    if self.peak_val > 0:
        drawdown = max(0, (self.peak_val - self._get_val()) / self.peak_val)
    else:
        drawdown = 0
    dd_penalty = 10 * (drawdown**1.5)
    reward = (
        self.alpha * return_z +
        self.beta * sharpe_z -
        cash_penalty -
        dd_penalty
    )

    return np.clip(reward, -10, 10)
  def _get_obs(self):
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
class DQNAgent(object):
  def __init__(self,state_size,action_size):
    self.state_size=state_size
    self.action_size=action_size
    # self.memory=ReplayBuffer(state_size,action_size,size=800)
    self.memory = PrioritizedReplayBuffer(state_size, action_size, size=800)
    self.gamma=0.95 #discount rate
    self.adaptive_epsilon=0.3 #exploration rate
    self.original_epsilon=0.3
    self.epsilon_min=0.01
    self.epsilon_decay=0.995
    self.decay_frequency = 2
    self.model=mlp(state_size,action_size)
    self.target_model = mlp(state_size, action_size)  # Add this
    self.update_target()
  # Pre-compile prediction function
    self.predict_fn = tf.function(
        lambda x: self.model(x),
        input_signature=[tf.TensorSpec(shape=(None, state_size), dtype=tf.float32)]
    )
  def update_replay_memory(self,state,action,reward,next_state,done):
    self.memory.store(1.0, state, action, reward, next_state, done)
  # def adapt_epsilon(self, portfolio_value, initial_investment, episode,
  #                min_epsilon=0.01, max_epsilon=1.0,
  #                danger_zone=0.7, recovery_zone=0.9):
  #   """
  #   Adaptive ε-greedy policy with portfolio-aware adjustment

  #   Args:
  #       portfolio_value: Current portfolio value
  #       initial_investment: Starting capital
  #       episode: Current episode number
  #       min_epsilon: Minimum exploration rate (default: 1%)
  #       max_epsilon: Maximum exploration rate (default: 100%)
  #       danger_zone: Portfolio threshold for conservative mode (default: 70%)
  #       recovery_zone: Portfolio threshold for normal mode (default: 90%)
  #   """
  #   # Dynamic baseline adjusts with training progress
  #   progress = min(1.0, episode / 2000)  # Adjust denominator based on total episodes
    # dynamic_threshold = danger_zone + (recovery_zone - danger_zone) * (1 - progress)

    # # Portfolio performance assessment
    # performance_ratio = portfolio_value / initial_investment

    # if performance_ratio < dynamic_threshold:
    #     # Conservative mode - slower decay when performing poorly
    #     decay_rate = 0.997
    #     self.epsilon = max(min_epsilon,
    #                       self.epsilon * decay_rate)
    # else:
    #     # Normal mode - standard decay
    #     decay_rate = 0.99
    #     self.epsilon = max(min_epsilon,
    #                       self.epsilon * decay_rate)

    # # Episode-based override (ensure eventual exploitation)
    # self.epsilon = max(min_epsilon,
    #                   min(max_epsilon,
    #                      self.epsilon * (0.9995 ** episode)))
  def decay_base_epsilon(self):
    if(self.original_epsilon>0.03):
        self.original_epsilon = max(self.original_epsilon * self.epsilon_decay, 
                                    self.epsilon_min)
    else:
        self.original_epsilon = max(self.original_epsilon *0.999, 
                                    self.epsilon_min)
  def adapt_epsilon(self, portfolio_value, initial_investment, episode):
    # Base decay
    progress = episode / 4000
    base_epsilon = self.epsilon_min + (self.original_epsilon - self.epsilon_min) * (1 - progress)**2
    self.adaptive_epsilon=base_epsilon
    # Performance boost
    perf_ratio = portfolio_value / initial_investment
    if perf_ratio < 0.95:
        self.adaptive_epsilon=min(base_epsilon * 1.5, 0.5)
    elif perf_ratio > 1.2:
        self.adaptive_epsilon=max(base_epsilon * 0.8, self.epsilon_min)
    return self.adaptive_epsilon

  # def adapt_epsilon(self, portfolio_value, initial_investment, episode):
  #   baseline = 0.9 - (episode / 10000)  # Gradually relax the threshold
  #   if portfolio_value < initial_investment * max(0.7, baseline):
  #       self.epsilon = min(0.3, self.epsilon)  # More aggressive reduction
  #       self.epsilon_decay = 0.998  # Slower decay during trouble
  #   else:
  #       self.epsilon *= self.epsilon_decay
  #       self.epsilon_decay = 0.995  # Normal decay
  def act(self,state):
    if np.random.rand()<=self.adaptive_epsilon:
      return np.random.choice(self.action_size)
    state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
    act_values = self.predict_fn(state_tensor).numpy()
    return np.argmax(act_values[0])
  def update_target(self):
    self.target_model.set_weights(self.model.get_weights())
  @tf.function      # TensorFlow traces the function once, builds a computation graph, and then runs that graph for future calls—much faster and more portable.
  def replay(self,batch_size=256):
    #first check if replay contains enough data
    if self.memory.size<batch_size:
      return
    # sample a batch of data from the replay memory
    minibatch = self.memory.sample_batch(batch_size)
    states = minibatch['s']
    actions = minibatch['a']
    rewards = minibatch['r']
    next_states = minibatch['s2']
    done = minibatch['d']
    weights = minibatch['weights']
    indices = minibatch['indices']
    # Calculate the tentative target: Q(s',a)
    target = rewards + (1 - done) * self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1)
    # With the Keras API, the target (usually) must have the same
    # shape as the predictions.
    # However, we only need to update the network for the actions
    # which were actually taken.
    # We can accomplish this by setting the target to be equal to
    # the prediction for all values.
    # Then, only change the targets for the actions taken.
    # Q(s,a)
    current_Q = self.model.predict(states, verbose=0)
    target_full = current_Q.copy()
    target_full[np.arange(batch_size), actions] = target
    # Run one training step
    td_errors = np.abs(target - current_Q[np.arange(batch_size), actions])+1e-5
    self.model.train_on_batch(states, target_full,sample_weight=np.array(weights))
    self.memory.update_priorities(indices, td_errors)
  def load(self, name):
    self.model.load_weights(name)

  def save_model(self, name):
    self.model.save_weights(name)
  def save_target(self,name):
    self.target_model.save_weights(name)
