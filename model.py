import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn import model_selection
from pandas.io import parsers
import os
from data import generate_samples, preprocess


local_project_path = '/'
local_data_path = os.path.join(local_project_path, 'data')


if __name__ == '__main__':
    # Read the data
    df = parsers.read_csv(os.path.join(local_data_path, 'driving_log.csv'))
    # Filter out frames with steering angles close to 0
    df = df[np.absolute(df.steering) > 0.001]
    # Split data into training and validation sets
    df_train, df_valid = model_selection.train_test_split(df, test_size=0.2)

    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit_generator(
        generate_samples(df_train, local_data_path),
        samples_per_epoch=df_train.count()[0],
        nb_epoch=20,
        validation_data=generate_samples(df_valid, local_data_path),
        nb_val_samples=df_valid.count()[0]
    )

    model.save_weights(os.path.join(local_project_path, 'behavioral_cloning.h5'))
    with open(os.path.join(local_project_path, 'behavioral_cloning.json'), 'w') as file:
        file.write(model.to_json())
    print('Model saved.')