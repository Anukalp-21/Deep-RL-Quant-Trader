import numpy as np
import tensorflow as tf
from src.model import LSTM_Model
from tensorflow.keras.layers import Dense,Input,Dropout,BatchNormalization,Activation,LSTM,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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
        # Ensure data is stored as a tuple
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

class DQNAgent(object):
  def __init__(self,state_size,action_size):
    self.state_size=state_size
    self.action_size=action_size
    # self.memory=ReplayBuffer(state_size,action_size,size=800)
    self.memory = PrioritizedReplayBuffer(state_size[1], action_size, size=70000) # Correct memory state_size
    self.gamma=0.99 #discount rate
    self.epsilon=1
    self.epsilon_min=0.01
    self.epsilon_decay= 0.99975
    self.decay_frequency = 1
    self.model=LSTM_Model(state_size,action_size)
    self.target_model = LSTM_Model(state_size, action_size)  # Add this
    self.update_target()
  # Pre-compile prediction function
    self.predict_fn = tf.function(
        lambda x: self.model(x),
        input_signature=[tf.TensorSpec(shape=(None, state_size[0],state_size[1]), dtype=tf.float32)]
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
  def decay_epsilon(self):
    if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

  # def adapt_epsilon(self, portfolio_value, initial_investment, episode):
  #   baseline = 0.9 - (episode / 10000)  # Gradually relax the threshold
  #   if portfolio_value < initial_investment * max(0.7, baseline):
  #       self.epsilon = min(0.3, self.epsilon)  # More aggressive reduction
  #       self.epsilon_decay = 0.998  # Slower decay during trouble
  #   else:
  #       self.epsilon *= self.epsilon_decay
  #       self.epsilon_decay = 0.995  # Normal decay
  def act(self,state):
    if np.random.rand()<=self.epsilon:
      return np.random.choice(self.action_size)
    state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
    act_values = self.predict_fn(state_tensor).numpy()
    return np.argmax(act_values[0])
  def update_target(self):
    self.target_model.set_weights(self.model.get_weights())
  # In DQNAgent.replay()
  # In your DQNAgent class

  @tf.function
  def _train_step(self, states, actions, rewards, next_states, done, weights):
    with tf.GradientTape() as tape:
        q_next = self.target_model(next_states)
        target = rewards + (1 - done) * self.gamma * tf.reduce_max(q_next, axis=1)

        q_current = self.model(states)
        action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
        q_action = tf.gather_nd(q_current, action_indices)

        td_errors = tf.abs(target - q_action)
        loss = tf.keras.losses.huber(target, q_action)
        weighted_loss = tf.reduce_mean(loss * weights)

    grads = tape.gradient(weighted_loss, self.model.trainable_variables)
    self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    return td_errors
  # In your DQNAgent class

  def replay(self, batch_size=256):
    if self.memory.size < batch_size:
        return

    minibatch = self.memory.sample_batch(batch_size)
    states = tf.convert_to_tensor(minibatch['s'], dtype=tf.float32)
    actions = tf.convert_to_tensor(minibatch['a'], dtype=tf.int32)
    rewards = tf.convert_to_tensor(minibatch['r'], dtype=tf.float32)
    next_states = tf.convert_to_tensor(minibatch['s2'], dtype=tf.float32)
    done = tf.convert_to_tensor(minibatch['d'], dtype=tf.float32)
    weights = tf.convert_to_tensor(minibatch['weights'], dtype=tf.float32)
    indices = minibatch['indices']

    # Call the new, decorated function to perform the training step
    td_errors = self._train_step(states, actions, rewards, next_states, done, weights)

    # Now, we are outside the @tf.function scope, so calling .numpy() is safe
    # Ensure td_errors.numpy() is always iterable
    self.memory.update_priorities(indices, td_errors.numpy().tolist() if isinstance(td_errors.numpy(), np.ndarray) else [td_errors.numpy()])
  def load(self, name):
    self.model.load_weights(name)

  def save_model(self, name):
    self.model.save_weights(name)
  def save_target(self,name):
    self.target_model.save_weights(name)
