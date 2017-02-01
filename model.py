import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd
tf.python.control_flow_ops = tf
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from sklearn import model_selection
from data import generate_samples, preprocess
from weights_logger_callback import WeightsLogger

local_project_path = '/'
local_data_path = os.path.join(local_project_path, 'data')

if __name__ == '__main__':
    # Read the data
    df = pd.io.parsers.read_csv(os.path.join(local_data_path, 'driving_log_balanced_n100_bins1000.csv'))
    # Split data into training and validation sets
    df_train, df_valid = model_selection.train_test_split(df, test_size=.2)

    model = models.Sequential()
    model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(core.Flatten())
    model.add(core.Dense(500, activation='relu'))
    model.add(core.Dropout(.5))
    model.add(core.Dense(100, activation='relu'))
    model.add(core.Dropout(.25))
    model.add(core.Dense(20, activation='relu'))
    model.add(core.Dense(1))
    model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')

    history = model.fit_generator(
        generate_samples(df_train, local_data_path),
        samples_per_epoch=df_train.shape[0],
        nb_epoch=30,
        validation_data=generate_samples(df_valid, local_data_path, augment=False),
        callbacks=[WeightsLogger(root_path=local_project_path)]
        nb_val_samples=df_valid.shape[0],
    )

    with open(os.path.join(local_project_path, 'model.json'), 'w') as file:
        file.write(model.to_json())

    backend.clear_session()