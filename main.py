import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.environment import MultiStockEnv
from src.agent import DQNAgent
import argparse

def get_train_data():
  df=pd.read_csv('data/Train_2015_2021.csv')
  return df.values
def get_val_bear_data():
  df=pd.read_csv('data/Val_2022_Bear.csv')
  return df.values
def get_val_bull_data():
    df=pd.read_csv('data/Val_2023_2024_Bull.csv') 
    return df.values
def get_test_data():
  df=pd.read_csv('data/Test_2025.csv')
  print(df.head())
  return df.values

def make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def get_scaler(env):
  states=[]
  env.reset(mode='test')
  for _ in range(env.n_step):
    action=np.random.choice(env.action_space)
    current_state = env._get_current_state()
    states.append(current_state)
    state,reward,done,info=env.step(action,mode='test')
    if done:
      break
  scaler=StandardScaler()
  scaler.fit(states)
  return scaler

def play_one_episode(agent,scaler,env,is_train,batch_size):
    curr_epsilon = agent.epsilon
    if is_train == 'test':
        agent.epsilon = 0.0
    state = env.reset(mode=is_train)
    scaled_state = scaler.transform(state)
    done = False
    all_daily_returns = []
    prev_val = env.initial_investment
    while not done:
        action = agent.act(scaled_state[np.newaxis, :,:])
        next_state, reward, done, info = env.step(action,mode=is_train)
        scaled_next_state = scaler.transform(next_state)
        curr_val = info['curr_val']
        daily_return = (curr_val - prev_val) / (prev_val + 1e-8)
        all_daily_returns.append(daily_return)
        prev_val = curr_val
        if is_train == 'train':
            agent.update_replay_memory(scaled_state, action, reward, scaled_next_state, done)
            steps_taken = env.curr_step - env.start_step
            if steps_taken>0 and steps_taken% 21 == 0:
                for _ in range(1):
                    agent.replay(batch_size)
        scaled_state = scaled_next_state
    if is_train == 'test':
        agent.epsilon = curr_epsilon
    final_sharpe = MultiStockEnv.sharpe_ratio(all_daily_returns)
    return info['curr_val'], final_sharpe

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Deep RL Quant Trading Agent')
    parser.add_argument('--mode',type=str,default='test',choices=['train','test'],help='Run mode: "train" to train a new agent, "test" to evaluate existing weights.')
    args=parser.parse_args()
    mode=args.mode
    models_folder='models'
    rewards_folder='backtests'
    data_folder='data'
    model_file='dqn_BEST_MULTI_REGIME.weights.h5'
    target_file='target_BEST_MULTI_REGIME.weights.h5'
    num_episodes=750
    batch_size=64
    initial_investment=20000
    make_dir(models_folder)
    make_dir(rewards_folder)
    print(f"🚀 Initializing Agent in {mode.upper()} mode...")
    if mode=='train':
        train_data=get_train_data()
        val_bear_data=get_val_bear_data()
        val_bull_data=get_val_bull_data()
        env=MultiStockEnv(train_data,initial_investment)
        env_val_bear=MultiStockEnv(val_bear_data,initial_investment)
        env_val_bull=MultiStockEnv(val_bull_data,initial_investment)
        scaler=get_scaler(env)
        with open(f'{models_folder}/scaler.pkl','wb') as f:
            pickle.dump(scaler, f)
    else:
        test_data=get_test_data()
        env=MultiStockEnv(test_data,initial_investment)
        with open(f'{models_folder}/scaler.pkl','rb') as f:
            scaler=pickle.load(f)
    state_size=env.state_dim
    action_size=len(env.action_space)
    agent=DQNAgent(state_size, action_size)
    portfolio_value=[]
    if mode=='test':
        agent.epsilon=0.0
        agent.load(f'{models_folder}/{model_file}')
        num_test_runs=5
        print(f"\nEvaluating on 2025 Unseen Data (Running {num_test_runs} Monte Carlo simulations for slippage variance)...")
        test_sharpes=[]
        for i in range(num_test_runs):
            val, sharpe=play_one_episode(agent,scaler,env,mode,batch_size)
            portfolio_value.append(val)
            test_sharpes.append(sharpe)
            print(f"  Run {i+1}/{num_test_runs} -> Final Value: ₹{val:,.2f} | Sharpe: {sharpe:.2f}")
        avg_val=np.mean(portfolio_value)
        avg_sharpe=np.mean(test_sharpes)
        avg_return=((avg_val-initial_investment)/initial_investment)*100
        print("\n" + "="*50)
        print("🚀 TEST MODE RESULTS (AVERAGE OF 5 RUNS)")
        print("="*50)
        print(f"Initial Capital:        ₹{initial_investment:,.2f}")
        print(f"Average Final Value:    ₹{avg_val:,.2f} ({avg_return:+.2f}%)")
        print(f"Average Sharpe Ratio:   {avg_sharpe:.2f}")
        print("="*50 + "\n")
    elif mode == 'train':
        initial_lr = 7e-5
        lr_decay =  0.9997
        best_combined_score = -float('inf')
        for e in range(num_episodes):
            t0 = datetime.now()
            val, sharpe = play_one_episode(agent, scaler, env, mode, batch_size)
            new_lr = initial_lr * (lr_decay ** e)
            agent.model.optimizer.learning_rate.assign(new_lr)
            current_lr = agent.model.optimizer.learning_rate.numpy()
            agent.memory.increment_beta()
            if e % agent.decay_frequency == 0:
                agent.decay_epsilon()
            if e % 10 == 0:
                agent.update_target()
            if e > 0 and e % 5 == 0:
                bear_val, bear_sharpe = play_one_episode(agent, scaler, env_val_bear, 'test', batch_size)
                bull_val, bull_sharpe = play_one_episode(agent, scaler, env_val_bull, 'test', batch_size)
                combined_score = min(bear_sharpe, bull_sharpe) + ((bear_sharpe + bull_sharpe) * 0.1)
                print(f"\n--- Validation @ Episode {e} ---")
                print(f"  --> Bear (2022)  | Val: ₹{bear_val:.2f} | Sharpe: {bear_sharpe:.2f}")
                print(f"  --> Bull (23-24) | Val: ₹{bull_val:.2f} | Sharpe: {bull_sharpe:.2f}")
                print(f"  --> Combined Multi-Regime Score: {combined_score:.4f}")
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    agent.save_model(f'{models_folder}/{model_file}')
                    agent.save_target(f'{models_folder}/{target_file}')
                    print("  🌟 NEW BEST MULTI-REGIME MODEL SAVED! 🌟\n")
            dt = datetime.now() - t0
            print(f"Episode: {e+1}/{num_episodes}, Val: ₹{val:.2f}, Duration: {dt}, LR: {current_lr:.6f}, Eps: {agent.epsilon:.3f}, Train Sharpe: {sharpe:.2f}")
            portfolio_value.append(val)
    np.save(f'{rewards_folder}/{mode}_portfolio_history.npy', portfolio_value)