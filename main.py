import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.environment import MultiStockEnv
from src.agent import DQNAgent

def get_data():
  df=pd.read_csv('Data1/Train 2015-22.csv')
  print(df.head())
  return df.values

def get_test_data():
  df=pd.read_csv('Data1/Test 2025.csv')
  print(df.head())
  return df.values

def make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def get_scaler(env):
  states=[]
  for _ in range(env.n_step):
    action=np.random.choice(env.action_space)
    # Get the raw state from the environment
    current_state = env._get_current_state()
    # Append the raw state (which should be 1D) to the states list
    states.append(current_state)
    # Take a step to advance the environment, but we don't use the output state here for fitting
    state,reward,done,info=env.step(action)
    if done: # Stop if the environment is done
      break
  scaler=StandardScaler()
  scaler.fit(states)
  return scaler

def play_one_episode(agent, scaler, env, is_train, batch_size):
    curr_epsilon = agent.epsilon
    if is_train == 'test':
        agent.epsilon = 0.0

    # Get the initial raw state from the environment
    state = env.reset(mode=is_train)
    # Scale the initial state
    scaled_state = scaler.transform(state)

    done = False
    all_daily_returns = []
    prev_val = env.initial_investment

    while not done:
        # Use the scaled state for the agent's action
        action = agent.act(scaled_state[np.newaxis, :,:])

        # Get the next state and other information from the environment (raw state)
        next_state, reward, done, info = env.step(action)

        # Scale the next state
        scaled_next_state = scaler.transform(next_state)

        curr_val = info['curr_val']
        daily_return = (curr_val - prev_val) / (prev_val + 1e-8)
        all_daily_returns.append(daily_return)
        prev_val = curr_val

        if is_train == 'train':
            # Store the scaled states in the replay memory
            agent.update_replay_memory(scaled_state, action, reward, scaled_next_state, done)

        # Update the current scaled state for the next iteration
        scaled_state = scaled_next_state

    if is_train == 'test':
        agent.epsilon = curr_epsilon

    final_sharpe = MultiStockEnv.sharpe_ratio(all_daily_returns)
    return info['curr_val'], final_sharpe
def get_alpha_beta(portfolio_val, episode, initial_investment, total_episodes):
    # This is your first function, slightly renamed for clarity
    curriculum_duration = total_episodes * 0.8
    progress = min(1.0, episode / curriculum_duration)

    base_alpha = 0.3 + 0.4 * progress
    base_beta = 1.0 - base_alpha  # Simpler way to ensure they sum to 1

    performance_ratio = portfolio_val / initial_investment

    if performance_ratio > 1.5:
        alpha = max(0.3, base_alpha - 0.2)
    elif performance_ratio < 0.9:
        alpha = max(0.2, base_alpha - 0.1)
    else:
        alpha = base_alpha

    alpha = max(0.0, min(1.0, alpha))
    beta = 1.0 - alpha

    return alpha, beta
if __name__ == '__main__':
    # Config
    models_folder = 'models'
    rewards_folder = 'backtests'
    model_file = 'dqn.weights.h5'
    target_file='target.weights.h5'
    norm_file = 'normalization_stats.pkl'  # NEW: File for normalization stats
    num_episodes = 21000
    batch_size = 256
    initial_investment = 20000
    alpha = 0.6
    beta = 0.4
    mode = input('Enter mode (train/test): ').strip().lower()
    while mode not in ['train', 'test']:
        print('Invalid mode!')
        mode = input('Enter mode (train/test): ').strip().lower()
    # Create directories
    make_dir(models_folder)
    make_dir(rewards_folder)

    # Load data
    data = get_data()
    n_timesteps, n_stocks = data.shape
    n_train = n_timesteps *2// 3
    train_data = data
    test_data = get_test_data()

    # Create environment with CORRECTED parameter order
    env = MultiStockEnv(train_data, alpha, beta, initial_investment)  # FIXED ORDER

    # NEW: Warm-up phase for normalization statistics
    # norm_path ='/kaggle/input/pkl_file/tensorflow2/pkkkllll/1/normalization_stats_calibrated.pkl'

    # if mode == 'train' and not os.path.exists(norm_path):
    #     print("\n=== RUNNING WARM-UP PHASE ===")
    #     (return_mean, return_std, sharpe_mean, sharpe_std) = run_warmup_phase(env, num_episodes=warmup_episodes)

    #     # Save normalization stats
    #     with open(norm_path, 'wb') as f:
    #         pickle.dump((return_mean, return_std, sharpe_mean, sharpe_std), f)
    #     print(f"Saved normalization parameters to {norm_path}")
    # else:
    #     # Load existing stats
    #     with open(norm_path, 'rb') as f:
    #         calibrated_stats = pickle.load(f)
    #     return_mean = calibrated_stats[0]
    #     return_std = calibrated_stats[1]
    #     sharpe_mean = calibrated_stats[2]
    #     sharpe_std = calibrated_stats[3]
    #     print("✅ Successfully loaded calibrated normalization stats.")
    #     print("   - Environment updated with new stats.")

    # # Set normalization parameters in environment
    # env.set_normalization_params(return_mean, return_std, sharpe_mean, sharpe_std)

    # Initialize agent and scaler
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)
    initial_lr = 3e-4
    lr_decay =  0.99975
    portfolio_value = []
    sharpe_ratios = []

    if mode == 'test':
        # Load scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        num_episodes=100
        # Create test environment with CORRECTED parameter order
        test_env = MultiStockEnv(test_data, alpha, beta, initial_investment)  # FIXED ORDER
        # test_env.set_normalization_params(return_mean, return_std, sharpe_mean, sharpe_std)

        agent.epsilon=0.0
        agent.load(f'{models_folder}/{model_file}')
        agent.target_model.load_weights(f'{models_folder}/{target_file}')
        # Use test environment
        env = test_env
    if mode == 'train':
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    # Training loop
    for e in range(num_episodes):
        t0 = datetime.now()
        val, sharpe = play_one_episode(agent, scaler, env, mode, batch_size)
        current_lr=0.00
        # Update adaptive epsilon (only in train mode)
        if mode == 'train':
            agent.replay(batch_size)
            new_lr = initial_lr * (lr_decay ** e)
            agent.model.optimizer.learning_rate.assign(new_lr)
            alpha, beta = get_alpha_beta(val,e, initial_investment,num_episodes)
            env.alpha = alpha
            env.beta = beta
            # Get the *current* LR after assigning it
            current_lr = agent.model.optimizer.learning_rate.numpy()
            agent.memory.increment_beta()
            if e % agent.decay_frequency == 0:
                agent.decay_epsilon()
            if e%10==0:
                agent.update_target()
            if e % 50 == 0:
                test_env = MultiStockEnv(test_data, env.alpha, env.beta, initial_investment)
                # test_env.set_normalization_params(return_mean, return_std, sharpe_mean, sharpe_std)
                test_val,sharpe= play_one_episode(agent, scaler, test_env,'test', batch_size)
                print(f"Validation @ {e}: ₹{test_val:.2f},sharpe :{sharpe:.2f}")
                agent.save_model(f'{models_folder}/dqn_episode_{e}.weights.h5')
                agent.save_target(f'{models_folder}/target_episode_{e}.weights.h5')
        dt = datetime.now() - t0
        print(f"Episode: {e+1}/{num_episodes}, Portfolio Value: ₹{val:.2f}, Duration: {dt}, LR: {current_lr:.6f}, Epsilon: {agent.epsilon:.3f},sharpe :{sharpe:.2f},alpha: {env.alpha:.2f},beta: {env.beta:.2f}")
        portfolio_value.append(val)

    # Save results
    if mode == 'train':
        agent.save_model(f'{models_folder}/{model_file}')
        agent.save_target(f'{models_folder}/{target_file}')
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)

    # Plot results
    a = np.array(portfolio_value)
    print("\nPerformance Summary:")
    print(f"Average Portfolio Value: ₹{a.mean():.2f}")
    print(f"Minimum Portfolio Value: ₹{a.min():.2f}")
    print(f"Maximum Portfolio Value: ₹{a.max():.2f}")
    print(f"Final Portfolio Value: ₹{a[-1]:.2f}")

    plt.figure(figsize=(12, 6))
    if mode == 'train':
        plt.plot(a)
        plt.title('Training: Portfolio Value Over Episodes')
        plt.xlabel('Episode')
    else:
        plt.hist(a, bins=30)
        plt.title('Test: Portfolio Value Distribution')
        plt.xlabel('Portfolio Value (₹)')

    plt.ylabel('Portfolio Value (₹)')
    plt.grid(True)
    plt.savefig(f'{rewards_folder}/{mode}_results.png')
    plt.show()
