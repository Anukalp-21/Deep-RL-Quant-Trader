import numpy as np
import tensorflow as tf
from src.model import LSTM_Model
from tensorflow.keras.layers import Dense,Input,Dropout,BatchNormalization,Activation,LSTM,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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
    """
    Implementation of Prioritized Experience Replay (PER).

    Key parameters:
    - alpha (a): Controls how much prioritization is used (0 = uniform, 1 = full).
    - beta: Controls importance sampling weights (anneals to 1.0).
    """
    
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

            sample_prob = p / self.tree.total()
            weight = (self.size * sample_prob) ** -self.beta
            batch['weights'].append(weight)

        max_weight = max(batch['weights'])
        batch['weights'] = [w / max_weight for w in batch['weights']]

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


class DQNAgent(object):
  def __init__(self,state_size,action_size):
    self.state_size=state_size
    self.action_size=action_size
    self.memory = PrioritizedReplayBuffer(state_size[1], action_size, size=70000)
    self.gamma=0.99
    self.epsilon=1
    self.epsilon_min=0.01
    self.epsilon_decay= 0.99975
    self.decay_frequency = 1
    self.model=LSTM_Model(state_size,action_size)
    self.target_model = LSTM_Model(state_size, action_size) 
    self.update_target()
    self.predict_fn = tf.function(
        lambda x: self.model(x),
        input_signature=[tf.TensorSpec(shape=(None, state_size[0],state_size[1]), dtype=tf.float32)]
    )
  def update_replay_memory(self,state,action,reward,next_state,done):
    self.memory.store(1.0, state, action, reward, next_state, done)

  def decay_epsilon(self):
    if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

  def act(self,state):
    if np.random.rand()<=self.epsilon:
      return np.random.choice(self.action_size)
    state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
    act_values = self.predict_fn(state_tensor).numpy()
    return np.argmax(act_values[0])
  def update_target(self):
    self.target_model.set_weights(self.model.get_weights())

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

    td_errors = self._train_step(states, actions, rewards, next_states, done, weights)
    self.memory.update_priorities(indices, td_errors.numpy().tolist() if isinstance(td_errors.numpy(), np.ndarray) else [td_errors.numpy()])
  def load(self, name):
    self.model.load_weights(name)

  def save_model(self, name):
    self.model.save_weights(name)
  def save_target(self,name):
    self.target_model.save_weights(name)
