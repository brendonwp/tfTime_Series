# ==============================================================================
#
# TIME SERIES QUESTION
#
# Build and train a neural network to predict time indexed variables of
# the multivariate house hold electric power consumption time series dataset.
# Using a window of past 24 observations of the 7 variables, the model
# should be trained to predict the next 24 observations of the 7 variables.
#
# ==============================================================================
#
# ABOUT THE DATASET
#
# Original Source:
# https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
#
# The original Individual House Hold Electric Power Consumption Dataset
# has Measurements of electric power consumption in one household with
# a one-minute sampling rate over a period of almost 4 years.
#
# Different electrical quantities and some sub-metering values are available.
#
# For the purpose of the examination we have provided a subset containing
# the data for the first 60 days in the dataset. We have also cleaned the
# dataset beforehand to remove missing values. The dataset is provided as a
# CSV file in the project.
#
# The dataset has a total of 7 features ordered by time.
# ==============================================================================

"""
Title: Power_Multi_Var_In_Multi_Var_Out

Description: Runs a prediction of all variables of power consumption 
   from all input variables 
  Original Source:
    https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Usage: Run direct from IDE

Arguments: n/a 

Functions: 

Classes: 

Examples:

Author: Brendon Wolff-Piggott 

Date: 7 June 2024 
"""


import urllib
import zipfile

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

#########################
N_EPOCHS=25 # 10 
BATCH_SIZE=32 # 32 Increased this (see notes below) since a 90 epoch run shows validation 
  # going off wild after 10 epochs
"""
# E25 BS 16 MAE=0.03 - Validation OK until epoch 15

# E25 BS 32 MAE=0.024 - The loss curve steps down around 5, 15, 20
# E50 BS 32 MAE=0.024 - Training and validation MAE both drop between
#           epochs 45 and 50. But MAE shows no improvement

# E25 BS 64 MAE=0.02 - Loss curves still broken and separate from about Epoch 6
# E05    64     0.027 
#  TS graph shows validation series changes in steps and drops from close to 
#  the maximum values to 0 fairly often. I would guess that the data should be transformed 
#  somehow.  
#  If working further on this dataset I would explore the data more - see TF TS Tutorial -
#  and do some feature engineering (eg FFT analysis, rescaling)
#
# NOTE: NEED TO EXPLORE THIS MAE CALC (for all variables) vs CALC PROVIDED 
      IN DIESEL EXAM
"""
#########################

def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()

def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)


def mae(y_true, y_pred):
    return np.mean(abs(y_true.ravel() - y_pred.ravel()))


def model_forecast(model, series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
    forecast = model.predict(ds)
    return forecast

########################

def solution_model():
    global N_FEATURES
    global SPLIT_TIME
    global BATCH_SIZE
    global N_PAST
    global N_FUTURE
    global SHIFT
	
    download_and_extract_data()
    df = pd.read_csv('household_power_consumption.csv', sep=',', index_col='datetime', header=0)

    N_FEATURES = len(df.columns) 

    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    SPLIT_TIME = int(len(data) * 0.5)
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    BATCH_SIZE = 32
    N_PAST = 24
    N_FUTURE = 24 
    SHIFT = 1 

    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    # Code to define your model.
    print("Entering Model Definition\n")

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
        tf.keras.layers.Reshape([N_FUTURE, N_FEATURES])
    ])

    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["mae"]
    )


    tfLearningRatePlot.learning_rate_history(model, train_set)

    model_inp_shape=tfToolkit.model_input_shape(model)
    print("\nModel Summary\n", model.summary())
    print("\nResult of passing an array with random content: ")
    tfToolkit.input_output_shape(model, N_PAST, N_FEATURES)

    history_fit=model.fit(train_set, epochs=N_EPOCHS, batch_size= BATCH_SIZE, verbose=1, validation_data=valid_set,
                          shuffle=False
    )

    tfToolkit.plot_train_val(history_fit)

    return x_train, x_valid, data, model

#######################################################################

if __name__ == '__main__':
    x_train,x_valid,data,model = solution_model()
    
    rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
    rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, :]
    x_valid = x_valid[:rnn_forecast.shape[0]]
    
    print("The shape of x_valid is: ", x_valid.shape)           # (43191, 7)
    print("The shape of rnn_forecast is: ", rnn_forecast.shape) # (43191, 7)

    result = mae(x_valid, rnn_forecast)
    print("\n\nThe MAE on the prediction is: ", result, "\n")

    x_valid_last = x_valid[:, -1] # Extracts the final variable only -
                    # wrong when all output variables are relevant - ITERATE
    size = len(x_valid_last)        # n/a here - Plot multiple 
    x_axis_array = np.arange(1, size + 1) # train / valid lines 
    tfToolkit.plot_line( x_axis_array, x_valid_last, rnn_forecast[:, -1], "Time", "Price",
                         "Validation Series", "Forecast Series")
                       # Iterate over fields instead of hard-coding

    print("\n**Run finished**")
