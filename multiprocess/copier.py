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
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import regularizers


class Copier():
    def __init__(self, config)
        activation_func = config["activation"]
        obs_dim = config["obs_dim"]
        l2_alpha = config["l2_alpha"]

        self.model = Sequential()
        self.model.add(Dense(16, activation=activation, kernel_initializer='he_normal', input_shape=(obs_dim,), kernel_regularizer=regularizers.l2(l2_alpha)))
        self.model.add(Dense(16, activation=activation, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_alpha)))
        self.model.add(Dense(num_actions, activation='softmax'))
        # compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, observations, actions):
       X_train = observations
       y_train = actions
       self.model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

    def predict(self, new_obs):
       new_obs = new_obs.reshape(1, self.obs_dim)
       yhat = self.model.predict(new_obs)
       predicted_action = argmax(yhat)

       return predicted_action
       
