"""
Title: Diesel_Multi_Var_In_Single_Var_Out

Description: Runs a prediction of the final variable of fuel (Diesel) 
  from all variables in the US EIA data. See note under Functions below 
  Original Source: https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm

Usage: Run direct from IDE

Arguments: n/a Data location is hardcoded

Functions: See ds.map in windowed_dataset for mapping of all input vars 
    to single (last) variable

Classes: 

Examples:

Author: Brendon Wolff-Piggott 

Date: 6 June 2024 
"""

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import os
import sys

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
tf.keras.backend.clear_session()
gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

current_dir = os.path.dirname(__file__)
common_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'common', 'utils'))
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
sys.path.append(common_dir)

import tfToolkit
from tfToolkit import customMetricCallback
import tfLearningRatePlot

##########################################
N_EPOCHS=70
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
    ds = ds.map(lambda w: (w[:n_past], w[n_past:, -1]))  
    return ds.batch(batch_size).prefetch(1)


class custommaeCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        # "not None" checks that "mae" exists in the logs dictionary before making the
        #  comparison to the cutoff value

        # Required MAE Value is 0.02 or less
        #  Will need to train to better than 0.02 to get prediction OK
        if(logs.get("mae") is not None and logs.get("mae") < 0.025):
            print("\nReached MAE < 0.025. Terminating training")
            self.model.stop_training = True

def mae(y_true, y_pred):
    return np.mean(abs(y_true.ravel() - y_pred.ravel()))

def plot_series(time, series, label, color):
    plt.plot(time, series, label=label, color=color)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

def plot_train_val(history_fit):
    mae = history_fit.history['mae']
    val_mae = history_fit.history['val_mae']
    loss = history_fit.history['loss']
    val_loss = history_fit.history['val_loss']
    epochs = range(len(mae))

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)

    plt.show()

    plt.plot(epochs, mae, 'r', label='Training MAE')
    plt.plot(epochs, val_mae, 'b', label='Validation MAE')
    plt.title('Training and validation MAE')
    plt.legend(loc=0)

    plt.show()


####################################

def model_forecast(model, series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
    forecast = model.predict(ds)
    return forecast

##########################################################################################

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
        tf.keras.layers.Dense(N_FUTURE * 1, activation="relu"),
         tf.keras.layers.Reshape([N_FUTURE, 1]),
    ])

    optimizer =  tf.keras.optimizers.Adam()
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=optimizer,
        metrics=["mae"]
    )

    tfToolkit.model_input_shape(model)
    print("Model Summary", model.summary())
    tfToolkit.input_output_shape(model, 10, 13)

    tfLearningRatePlot.learning_rate_history(model, train_set) 


    callbacks = custommaeCallback()

    history_fit = model.fit(
        train_set, batch_size=BATCH_SIZE, epochs=N_EPOCHS, verbose=1, validation_data=valid_set, callbacks=[callbacks]
    )

    tfToolkit.plot_train_val(history_fit)

    return x_train,x_valid,data,model



#############################################################################################


if __name__ == '__main__':
    x_train,x_valid,data,model = solution_model()

    rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
    rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]
    x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0], -N_FUTURE:])
    time_array = np.arange(1, len(x_valid) + 1)

    x_valid_last = x_valid[:, -1] # Extracts the value of the last (Diesel) variable to plot

    tfToolkit.plot_line(time_array, x_valid_last, rnn_forecast, x_axis_label="Time Step", y_axis_label="Value", y_series1_label="Observed D1",
              y_series2_label="Predicted D1", title="Time Series of Observed vs Predicted Diesel Prices ")

    result = tf.keras.metrics.mean_absolute_error(x_valid_last, rnn_forecast).numpy()
    print("\n\nThe MAE on the prediction is: ", result, "\n")

    exit(0)

