"""
Title: Diesel_Multi_Var_In_Multi_Var_Out

Description: Runs a prediction of all variables of fuel from all variables 
  in the US EIA data 
  Original Source: https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm

Usage: Run direct from IDE

Arguments: n/a Data location is hardcoded

Functions: 

Classes: 

Examples:

Author: Brendon Wolff-Piggott 

Date: 6 June 2024 
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib

# Graphical backend to matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import os
import sys

current_dir = os.path.dirname(__file__)
common_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'common', 'utils'))
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
sys.path.append(common_dir)

import tfToolkit
from tfToolkit import customMetricCallback
import tfLearningRatePlot

#############
tf.keras.backend.clear_session()
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

##########################################
N_EPOCHS = 140
THRESHOLD_MAE = 0.02

N_FEATURES=1
SPLIT_TIME=1139
BATCH_SIZE=32 
N_PAST=10
N_FUTURE=10
SHIFT=1

DATAFILE_NAME = "PET_PRI_GND_DCUS_NUS_W.csv"

##########################################

def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def solution_model():
    global N_FEATURES
    global SPLIT_TIME
    global BATCH_SIZE
    global N_PAST
    global N_FUTURE
    global SHIFT

    file_name = os.path.abspath(os.path.join(current_dir, '..', 'data', DATAFILE_NAME))
    df = pd.read_csv(file_name, index_col='Date', header=0)

    N_FEATURES = len(df.columns)

    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    SPLIT_TIME = int(len(data) * 0.8) 
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    BATCH_SIZE = 32

    N_PAST = 10
    N_FUTURE = 10  
    SHIFT = 1

    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    model = tf.keras.models.Sequential([
         tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                               strides=1,
                               activation="relu",
                               padding='causal',
                               input_shape=[N_PAST, N_FEATURES]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(N_FUTURE*N_FEATURES, activation="relu"),
        tf.keras.layers.Reshape([N_FUTURE, N_FEATURES]),
    ])

    optimizer =  tf.keras.optimizers.Adam()
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=optimizer,
        metrics=["mae"]
    )

    tfLearningRatePlot.learning_rate_history(model, train_set)

    model_inp_shape=tfToolkit.model_input_shape(model)
    print("Input shape to model: ", model_inp_shape)
    print("\nModel Summary", model.summary())

    print("\nResult of passing an array with random content: ")
    tfToolkit.input_output_shape(model, N_PAST, N_FEATURES)

######################

    callbacks = customMetricCallback("mae", THRESHOLD_MAE)
    history_fit = model.fit(
        train_set, batch_size=BATCH_SIZE, epochs=N_EPOCHS, verbose=1, validation_data=valid_set,
                                                      callbacks=[callbacks]
    )

    print("\nCalling plot_train_val Plot Function..")
    tfToolkit.plot_train_val(history_fit)

    return x_train,x_valid,data,model

#############################################################################################


if __name__ == '__main__':
    x_train,x_valid,data,model = solution_model()

    rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
    rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]
    x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0]])
    x_valid_last = x_valid[:, -1]
    result = tf.keras.metrics.mean_absolute_error(x_valid_last, rnn_forecast).numpy()
    print("\n\nThe MAE on the prediction is: ", result, "\n")

    size = len(x_valid_last)    
    x_axis_array = np.arange(1, size + 1)
    tfToolkit.plot_line( x_axis_array, x_valid_last, rnn_forecast, "Time", "Price",
                         "Validation Series", "Forecast Series")

    print("\n**Run finished**")
