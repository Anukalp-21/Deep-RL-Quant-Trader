import pandas as pd
import numpy as np
from datetime import datetime
import itertools
from collections import deque

def get_data():
  df=pd.read_csv('data/Train_2015_2021.csv')
  return df

feature_columns=[
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

    'SMA_200_ratio_NSEI', 
    'India_VIX_Close'
]
rsi_columns=['RSI_14_RELIANCE','RSI_14_INFY','RSI_14_SBIN']
price_columns=['RELIANCE_Close','SBIN_Close','INFY_Close']
def get_feature_indices():
  df=get_data()
  final_feature_indices=[df.columns.get_loc(c) for c in feature_columns]
  return final_feature_indices
def get_rsi_indices():
  df=get_data()
  final_rsi_indices=[df.columns.get_loc(c) for c in rsi_columns]
  return final_rsi_indices
def get_price_indices():
  df=get_data()
  final_prices_indices=[df.columns.get_loc(c) for c in price_columns]
  return final_prices_indices
class MultiStockEnv:

  def __init__(self,data,initial_investment=20000,feature_indices=None,rsi_indices=None,price_indices=None):
    self.stock_price_history=data
    self.n_step=self.stock_price_history.shape[0]
    self.n_stock=3
    self.sharpe=0.0
    self.current_drawdown=0.0
    self.initial_investment=initial_investment
    self.peak_val=initial_investment
    self.price_indices=price_indices if price_indices is not None else get_price_indices()
    self.rsi_indices=rsi_indices if rsi_indices is not None else get_rsi_indices()
    print(f"Data shape:{data.shape}")
    print(f"Price indices:{self.price_indices}")
    print(f"Sample prices:{data[0,self.price_indices]}")
    self.curr_step=None
    self.stock_owned=None
    self.stock_price=None
    self.cash_in_hand=None
    self.state_feature_indices=feature_indices if feature_indices is not None else get_feature_indices()
    self.action_space=np.arange(3**self.n_stock)
    self.action_list=list(map(list,itertools.product([0, 1, 2],repeat=self.n_stock)))
    self.returns_window=deque(maxlen=21)
    self.window_size=21
    self.state_history=deque(maxlen=self.window_size)
    self.last_rsi_mean=50.0
    self.feature_step_dim=5+len(self.state_feature_indices)
    self.state_dim=(self.window_size,self.feature_step_dim)
    self.reset()
  def reset(self,mode='train'):
    self.episode_length=252
    if mode=='train':
        self.start_step=np.random.randint(0,max(1,self.n_step-self.episode_length))
        self.curr_step=self.start_step
        self.cash_in_hand=self.initial_investment*np.random.uniform(0.8,1.2) 
        self.stock_owned=np.random.randint(0,3,size=self.n_stock).astype(float) 
    else:
        self.start_step=0
        self.curr_step=0
        self.stock_owned=np.zeros(self.n_stock)
        self.cash_in_hand=self.initial_investment
    self.stock_price=self.stock_price_history[self.curr_step,self.price_indices]
    self.peak_val=self._get_val()
    self.sharpe=0.0
    self.current_drawdown=0.0
    self.returns_window.clear()
    self.state_history.clear()
    current_state=self._get_current_state()
    for i in range(self.window_size):
      self.state_history.append(current_state)
    obs=self._get_obs()
    assert obs.shape==self.state_dim,f"obs shape {obs.shape}!={self.state_dim}"
    return obs
  @staticmethod
  def get_slippage_pct():
    return np.random.choice(
        [np.random.uniform(0,0.003), np.random.uniform(-0.001,0)],
        p=[0.7,0.3]
    )
  @staticmethod
  def calculate_transaction_cost(price,quantity,trade_type='delivery',broker='zerodha'):
    trade_value=price*quantity
    brokerage_rate=0.001  
    stt_rate=0.001 if trade_type=='delivery' else 0.00025
    exchange_txn_rate=0.0000325
    sebi_fee_rate=0.000001
    stamp_duty_rate=0.00015
    gst_rate=0.18
    brokerage=min(trade_value*brokerage_rate, 20.0) if broker=='zerodha' else trade_value*brokerage_rate
    stt=trade_value*stt_rate
    exchange_txn=trade_value*exchange_txn_rate
    sebi_fee=trade_value*sebi_fee_rate
    stamp_duty=trade_value*stamp_duty_rate
    gst=gst_rate*(brokerage+exchange_txn+sebi_fee)
    total_cost=brokerage+stt+exchange_txn+sebi_fee+stamp_duty+gst
    return total_cost
  @staticmethod
  def sharpe_ratio(returns,risk_free_rate=0.05):
    if len(returns)<5:
        return 0.0
    excess_returns=np.array(returns)-risk_free_rate/252
    mean_return=np.mean(excess_returns)
    std_return=np.std(excess_returns)+1e-8
    if std_return<1e-6:
        return 0.0
    return (mean_return/std_return)*np.sqrt(252)
  def step(self,action,mode):
    assert action in self.action_space
    prev_val=self._get_val()
    self._trade(action)
    if mode=='train':
        done=(self.curr_step-self.start_step)>=(self.episode_length-1)
    else:
        done=self.curr_step==self.n_step-1
    self.curr_step+=1
    if not done:
        self.stock_price=self.stock_price_history[self.curr_step,self.price_indices]
    curr_val=self._get_val()
    self.peak_val=max(self.peak_val,curr_val)
    portfolio_return=(curr_val-prev_val)/(prev_val+1e-8)
    self.returns_window.append(portfolio_return)
    prev_sharpe=self.sharpe
    self.sharpe=self.sharpe_ratio(self.returns_window)
    differential_sharpe=self.sharpe-prev_sharpe
    prev_drawdown=self.current_drawdown
    self.current_drawdown=self.calc_drawdown()
    delta_drawdown=self.current_drawdown-prev_drawdown
    if mode=='train':
        reward=self.findReward(portfolio_return,differential_sharpe,delta_drawdown)
    else :
        reward=0.00
    self.state_history.append(self._get_current_state())
    info={'curr_val':curr_val}
    next_obs=self._get_obs()
    assert next_obs.shape==self.state_dim,f"obs shape {next_obs.shape}!={self.state_dim}"
    return next_obs,reward,done,info

  def calc_drawdown(self):
    current_drawdown=0.0
    if self.peak_val>0:
        current_drawdown=max(0,(self.peak_val-self._get_val())/self.peak_val)
    return current_drawdown

  def findReward(self,portfolio_return,differential_sharpe,delta_drawdown):
    if np.isnan(portfolio_return) or np.isnan(differential_sharpe):
        return 0.0
    reward=(portfolio_return*100.0) 
    reward+=(differential_sharpe*0.1)
    if delta_drawdown>0: 
        reward-=(delta_drawdown*10.0)
    return np.tanh(reward)*5.0

  def _get_current_state(self):
    total_val=self._get_val()+1e-8
    portfolio = [
            (self.stock_owned[0]*self.stock_price[0])/total_val,
            (self.stock_owned[1]*self.stock_price[1])/total_val,
            (self.stock_owned[2]*self.stock_price[2])/total_val, 
            self.cash_in_hand/total_val,
            self.current_drawdown
    ]
    if self.curr_step<self.n_step:
        features=self.stock_price_history[self.curr_step,self.state_feature_indices]
    else:
        features = self.stock_price_history[self.curr_step-1,self.state_feature_indices]
    state=np.array(portfolio+features.tolist(),dtype=np.float32)
    return state
  def _get_obs(self):
    return np.array(self.state_history)
  def _get_val(self):
    gross_stock_value=self.stock_owned.dot(self.stock_price)
    realizable_stock_value=gross_stock_value*0.9985 
    return realizable_stock_value+self.cash_in_hand

  def _trade(self,action):
    if self.curr_step>=self.n_step-1:
        return
    action_vec = self.action_list[action]
    sell_index=[]
    buy_index=[]
    for i,a in enumerate(action_vec):
      if(a==0):
        sell_index.append(i)
      elif(a==2):
        buy_index.append(i)
    if sell_index:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        effective_price=self.stock_price[i]*(1-self.get_slippage_pct())
        cost=self.calculate_transaction_cost(effective_price,self.stock_owned[i])
        self.cash_in_hand+=effective_price*self.stock_owned[i]-cost
        self.stock_owned[i]=0
    if buy_index:
        available_cash_per_stock=self.cash_in_hand/len(buy_index)
        for i in buy_index:
            effective_price=self.stock_price[i]*(1+self.get_slippage_pct())
            max_shares=int(available_cash_per_stock/effective_price)
            if max_shares>0:
                cost = self.calculate_transaction_cost(effective_price,max_shares)
                total_trade_value=(max_shares*effective_price)+cost
                while total_trade_value>available_cash_per_stock and max_shares>0:
                    max_shares-=1
                    cost=self.calculate_transaction_cost(effective_price,max_shares)
                    total_trade_value=(max_shares*effective_price)+cost
                if max_shares>0:
                    self.stock_owned[i]+=max_shares
                    self.cash_in_hand-=total_trade_value