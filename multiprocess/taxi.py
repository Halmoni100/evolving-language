import numpy as np
from tensorflow import keras

def taxi_observation_transform(observation):
    observation_left = observation

    destination = observation_left % 4
    destination_one_hot = keras.utils.to_categorical(destination, num_classes=4)
    observation_left = observation_left // 4

    passenger_location = observation_left % 5
    passenger_location_one_hot = keras.utils.to_categorical(passenger_location, num_classes=5)
    observation_left = observation_left // 5

    taxi_col = observation_left % 5
    taxi_col_one_hot = keras.utils.to_categorical(taxi_col, num_classes=5)
    observation_left = observation_left // 5

    taxi_row = observation_left
    taxi_row_one_hot = keras.utils.to_categorical(taxi_row, num_classes=5)

    transformed_observation = np.concatenate((destination_one_hot, passenger_location_one_hot, taxi_col_one_hot, taxi_row_one_hot))
    return transformed_observation
