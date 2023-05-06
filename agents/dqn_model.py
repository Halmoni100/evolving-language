import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
from scipy.stats import entropy
from scipy.special import softmax


class ReplayBuffer():

    def __init__(self, max_size, input_dims):

        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)

        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)


    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1


    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims=16, fc2_dims=16):

    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(fc2_dims, kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dense(n_actions, activation=None)])

    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return model



class Agent():

    def __init__(self, id, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=0.999, epsilon_end=0.01,
                mem_size=1000000, fname='dqn_model.h5'):
        
        self.id = id
        self.action_space = [i for i in range(n_actions)]

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 32, 16)


    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    def choose_action(self, observation):
        out = self.q_eval.predict(np.array([observation]))
        prob = softmax(out)
        H = entropy(np.squeeze(prob))
        
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            dqn_command = False
        else:
            action = np.argmax(out)
            dqn_command = True

        return action, dqn_command, H


    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return
        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones

        self.q_eval.train_on_batch(states, q_target)


    def epsilon_decay(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min


    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file)
