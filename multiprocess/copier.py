import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class Copier():
    def __init__(self, config):
        activation = config["activation"]
        obs_dim = config["obs_dim"]
        act_dim = config["act_dim"]
        l2_alpha = config["l2_alpha"]
        width = config["width"]

        self.model = keras.Sequential()
        self.model.add(Dense(width, activation=activation, kernel_initializer='he_normal', input_shape=(obs_dim,), kernel_regularizer=keras.regularizers.l2(l2_alpha)))
        self.model.add(Dense(width, activation=activation, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_alpha)))
        self.model.add(Dense(act_dim, activation='softmax'))
        # compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

    def train(self, observations, actions):
       X_train = observations
       y_train = actions
       self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, new_obs):
       new_obs = new_obs.reshape(1, self.obs_dim)
       yhat = self.model.predict(new_obs)
       predicted_action = np.argmax(yhat)

       return predicted_action
       
