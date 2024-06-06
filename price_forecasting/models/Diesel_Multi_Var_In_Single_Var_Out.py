"""
Title: Diesel_Multi_Var_In_Single_Var_Out

Description: Runs a prediction of the final variable of fuel (Diesel) 
  from all variables in the US EIA data.  
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
N_EPOCHS=25 # 140
SPLIT_TIME=1139
BATCH_SIZE=32
N_PAST=10
N_FUTURE=10
SHIFT=1

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

# Plotting code provided by Gpt on 10 Aug
def plot_series(time, series, label, color):
    plt.plot(time, series, label=label, color=color)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

# Added May 2024 - Simple function to display numpy arrays of training and validation history
def plot_train_val(history_fit):
    mae = history_fit.history['mae']
    val_mae = history_fit.history['val_mae']
    loss = history_fit.history['loss']
    val_loss = history_fit.history['val_loss']
    epochs = range(len(mae))

# Plot loss - Does not plot a decreasing line from an initial maximum.
    #   Mimics accuracy??
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)

    plt.show()

# Plot accuracy
    plt.plot(epochs, mae, 'r', label='Training MAE')
    plt.plot(epochs, val_mae, 'b', label='Validation MAE')
    plt.title('Training and validation MAE')
    plt.legend(loc=0)

    plt.show()


###########################################################################################
# This function provided originally at the end of the template code
#############################################################################################

def model_forecast(model, series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
    forecast = model.predict(ds)
    return forecast

##########################################################################################

# COMPLETE THE CODE IN THIS FUNCTION
def solution_model():
    #Brendon
    global N_FEATURES
    global SPLIT_TIME
    global BATCH_SIZE
    global N_PAST
    global N_FUTURE
    global SHIFT

    # DO NOT CHANGE THIS CODE
    # Reads the dataset.
    # Original code from the exam
    # df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
    #                 infer_datetime_format=True, index_col='Week of', header=0)
    # Code for dataset downloaded from Kaggle
    df = pd.read_csv('/home/brendon/Insync/Data-Technical/Projects/tf2/data/PET_PRI_GND_DCUS_NUS_W.csv',
                     infer_datetime_format=True, index_col='Date', header=0)

    # Number of features in the dataset. We use all features as predictors to
    # predict all features of future time steps.
    N_FEATURES = len(df.columns) # DO NOT CHANGE THIS

    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.8) # DO NOT CHANGE THIS
    # Brendon
    print("\nSPLIT_TIME = ", SPLIT_TIME, "\n\n")

    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]


    # DO NOT CHANGE THIS CODE
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    # DO NOT CHANGE BATCH_SIZE IF YOU ARE USING STATEFUL LSTM/RNN/GRU.
    # THE TEST WILL FAIL TO GRADE YOUR SCORE IN SUCH CASES.
    # In other cases, it is advised not to change the batch size since it
    # might affect your final scores. While setting it to a lower size
    # might not do any harm, higher sizes might affect your scores.
    BATCH_SIZE = 32  # ADVISED NOT TO CHANGE THIS

    # DO NOT CHANGE N_PAST, N_FUTURE, SHIFT. The tests will fail to run
    # on the server.
    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 10  # DO NOT CHANGE THIS

    # Number of future time steps which are to be predicted.
    N_FUTURE = 10  # DO NOT CHANGE THIS

    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1  # DO NOT CHANGE THIS, why the MA`

    # Code to create windowed train and validation datasets.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    # Code to define your model.
    model = tf.keras.models.Sequential([

        # ADD YOUR LAYERS HERE.

        # If you don't follow the instructions in the following comments,
        # tests will fail to grade your code:
        # The input layer of your model must have an input shape of:
        # (BATCH_SIZE, N_PAST = 10, N_FEATURES = 1)
        # The model must have an output shape of:
        # (BATCH_SIZE, N_FUTURE = 10, N_FEATURES = 1).
        # Make sure that there are N_FEATURES = 1 neurons in the final dense
        # layer since the model predicts 1 feature.

        # HINT: Bidirectional LSTMs may help boost your score. This is only a
        # suggestion.

        # WARNING: After submitting the trained model for scoring, if you are
        # receiving a score of 0 or an error, please recheck the input and
        # output shapes of the model to see if it exactly matches our requirements.
        # The grading infrastructure is very strict about the shape requirements.
        # Most common issues occur when the shapes are not matching our
        # expectations.
        #
        # TIP: You can print the output of model.summary() to review the model
        # architecture, input and output shapes of each layer.
        # If you have made sure that you have matched the shape requirements
        # and all the other instructions we have laid down, and still
        # receive a bad score, you must work on improving your model.

        # WARNING: If you are using the GRU layer, it is advised not to use the
        # recurrent_dropout argument (you can alternatively set it to 0),
        # since it has not been implemented in the cuDNN kernel and may
        # result in much longer training times.

        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                               strides=1,
                               activation="relu",
                               padding='causal',
                               input_shape=[N_PAST, N_FEATURES]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(30, activation="relu"),
        # Brendon 10 May 2024 Changed to predict a single value
        tf.keras.layers.Dense(N_FUTURE * 1, activation="relu"),
#        tf.keras.layers.Dense(1)
         tf.keras.layers.Reshape([N_FUTURE, 1]),
#        tf.keras.layers.Dense(N_FUTURE*N_FEATURES, activation="relu"),
#        tf.keras.layers.Reshape([N_FUTURE, N_FEATURES]),
    ])

# Calling for plot of learning rate vs epoch
#    LearningRatePlot2.learning_rate_history(model, tf.keras.losses.Huber(),
#        tf.keras.optimizers.Adam(), ["mae"], train_set) 
    # Check that receive data back from called function
    # INSERT PLOT FUNCTION
#########################    
    
    # Code to train and compile the model
    optimizer =  tf.keras.optimizers.Adam()
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=optimizer,
        metrics=["mae"]
    )

# Brendon
    tfToolkit.model_input_shape(model)
    print("Model Summary", model.summary())
    tfToolkit.input_output_shape(model, 10, 13)

    
    # Learning rate calc and plot for compiled model
    # Debug
    tfLearningRatePlot.learning_rate_history(model, train_set) 


    callbacks = custommaeCallback()

    history_fit = model.fit(

        train_set, batch_size=BATCH_SIZE, epochs=N_EPOCHS, verbose=1, validation_data=valid_set, callbacks=[callbacks]
    )

    # May 2024 Insert plotting code to show fit
    tfToolkit.plot_train_val(history_fit)


# Brendon
    return x_train,x_valid,data,model



#############################################################################################


if __name__ == '__main__':
    x_train,x_valid,data,model = solution_model()
#    model.save("mymodel.h5")

# Brendon May 2024 Steps below are advised from the original code
    rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
    # Debug May 2024
    print("The shape of rnn_forecast as received from model_forecast is ", rnn_forecast.shape)
    # (1344, 1) for single variable
    # (1344, 10, 13) for multi-in and multi-out

    # 14 May 2024 Removed , 0 at end of slicing because there is only a single variable 
    #                    and I don't need to slice out one from among many 
    rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]

    print("The shape of x_valid before slicing is: ", x_valid.shape,"\n")
    x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0], -N_FUTURE:])

    time_array = np.arange(1, len(x_valid) + 1)
    print("The shape of the time array is: ", time_array.shape,"\n")

    # Debug May 2024
    print("The shape of the sliced rnn_forecast is: ", rnn_forecast.shape[0])
    print("The shape of x_valid is: ", x_valid.shape)

    x_valid_last = x_valid[:, -1]
    print("The shape of x_valid_last is: ", x_valid_last.shape)

    # Plot time series
    # Note: x_valid_last is the series of values for the last variable only
    tfToolkit.plot_line(time_array, x_valid_last, rnn_forecast, x_axis_label="Time Step", y_axis_label="Value", y_series1_label="Observed D1",
              y_series2_label="Predicted D1", title="Time Series of Observed vs Predicted Diesel Prices ")

    result = tf.keras.metrics.mean_absolute_error(x_valid_last, rnn_forecast).numpy()
    print("\n\nThe MAE on the prediction is: ", result, "\n")

    exit(0)

######################################################################################################################
    # FROM ORIGINAL DEBUGGING 2023

    print("\nGLOBAL VAR VALUES SINCE FN CALL: \n")
    print("N_FEATURES = ", N_FEATURES, flush=True)
    print("SPLIT_TIME = ", SPLIT_TIME, flush=True)
    print("BATCH_SIZE = ", BATCH_SIZE, flush=True)
    print("N_PAST = ", N_PAST, flush=True)
    print("N_FUTURE = ", N_FUTURE, flush=True)
    print("SHIFT = ", SHIFT, flush=True)
    print("\n\n", flush=True)

# Steps below work when only one output is generated

    rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)

# This line works:
    rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]

# This following line was provided
#    rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]
    # But fails with Traceback (most recent call last):
    #   File "/home/brendon/Insync/Data-Technical/Projects/tf2/202308-Cat5-Wrk.py", line 360, in <module>
    #     result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    #   File "/home/brendon/miniconda3/envs/tf-cert/lib/python3.8/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    #     raise e.with_traceback(filtered_tb) from None
    #   File "/home/brendon/miniconda3/envs/tf-cert/lib/python3.8/site-packages/keras/losses.py", line 1455, in mean_absolute_error
    #     return backend.mean(tf.abs(y_pred - y_true), axis=-1)
    # tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [214] vs. [214,13] [Op:Sub]
    #
    # Process finished with exit code 1

# Corrected above to:    rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1]

    x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0]])
    result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    print("\n\nThe MAE on the prediction is: ", result, "\n")

    exit(0) # So that there is a halt before the plot error pops up

# Define the time axis, it could be a range equal to the length of x_valid or rnn_forecast
    time = range(len(x_valid))

# Plot ground truth (actual values)
    plot_series(time, x_valid[:-1], label="Ground Truth", color="blue")

    plt.title("Training Set vs Model Prediction")
    plt.show()

# Plot fitted model values (predictions)
    plot_series(time, rnn_forecast, label="Model Prediction", color="red")

    plt.title("Validation Set vs Model Prediction")
    plt.show()

# Code below was provided in the exam for use in testing the model
#   Kept here for reference
# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.

# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS

#def model_forecast(model, series, window_size, batch_size):
#    ds = tf.data.Dataset.from_tensor_slices(series)
#    ds = ds.window(window_size, shift=1, drop_remainder=True)
#    ds = ds.flat_map(lambda w: w.batch(window_size))
#    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
#    forecast = model.predict(ds)
#    return forecast

# PASS THE NORMALIZED data IN THE FOLLOWING CODE

# rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
# rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]

# x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0]])
# result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
