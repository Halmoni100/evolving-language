import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


class Copier():
    def __init__(self, num_agents, ep_lookback, obs_dim, num_actions):

        self.mem_size = num_agents*ep_lookback
        self.state_memory = np.zeros((self.mem_size, obs_dim), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.obs_dim = obs_dim

        self.mem_cntr = 0

        self.model = Sequential()
        self.model.add(Dense(16, activation='tanh', kernel_initializer='he_normal', input_shape=(obs_dim,)))
        self.model.add(Dense(16, activation='tanh', kernel_initializer='he_normal'))
        self.model.add(Dense(num_actions, activation='softmax'))
        # compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def store_obs_action(self, state, action):
       index = self.mem_cntr % self.mem_size
       self.state_memory[index] = state
       self.action_memory[index] = action
       self.mem_cntr += 1

    
    def train(self):
       X_train = self.state_memory
       y_train = self.action_memory
       self.model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
       
       return 
    
    def predict(self, new_obs):
       new_obs = new_obs.reshape(1, self.obs_dim)
       yhat = self.model.predict(new_obs)
       predicted_action = argmax(yhat)

       return predicted_action
       
    
    
    



      