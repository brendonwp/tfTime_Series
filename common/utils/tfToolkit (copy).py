# Common functions developed for toolkit use
# Last update: 22 May 2024

import tensorflow as tf 
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class customMetricCallback(tf.keras.callbacks.Callback):
    """
    Provides a generalised metric stopping capability at "on_epoch_end"
    Args:
      metric (str):      Parameter naming metric to use as stopping criterion
      threshold (float): Parameter of metric value at which to end training
    Returns:
      stopping flag when metric meets threshold
    Raises:
      TypeError: If metric is not a string.
      ValueError: If threshold is not a float.
    """
    # Tested: Basic
    # Last updated 19 May 2024

    def __init__(self, metric_name, threshold):
        super(customMetricCallback, self).__init__()
        self.metric_name = metric_name
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metric_value = logs.get(self.metric_name)
        
        if metric_value is not None:
            if 'acc' in self.metric_name:
                # Use > operator for accuracy metrics
                if metric_value > self.threshold:
                    print(f"\nReached {self.metric_name} < {self.threshold}. Terminating training")
                    self.model.stop_training = True
            else:
                # Use < operator for other metrics
                if metric_value < self.threshold:
                    print(f"\nReached {self.metric_name} > {self.threshold}. Terminating training")
                    self.model.stop_training = True

# Example usage:
# callback = customMetricCallback(metric_name='mae', threshold=0.02)
# history = model.fit(train_set, batch_size=BATCH_SIZE, epochs=N_EPOCHS, verbose=1, 
#                     validation_data=valid_set, callbacks=[callback])

#################################### Evaluating and Printing Data types #####################################################

# From 202404-ToolkitDev-Plot.ipynb
#  Sample usage: 
#    variable_list = [sentences, labels]
#    describe_variables("solution_model", variable_list)  where "solution_model" is the name of the current function
#                                                      i.e. from which the call is being made


# Call the data_summary_inp() function iteratively with the function name
#    and list of variables
def describe_variables(calling_function_name, variables_list):
    """
    Prints calling function name and type and sample data from variable list
    Args:
      calling_function_name (str): Parameter naming function from which call is made (manually set)
      variables_list  (list)     : Parameter of variable names in a list to be described
    Calls:
      data_summary_inp
    Raises:
      TypeError: If metric is not a string.
      ValueError: If threshold is not a float.
    """
    # Tested: Basic
    # Last updated 19 May 2024
    # Return the type of an object and value if possible 
    for var in variables_list:
        data_summary_inp(calling_function_name, var)

def find_var_name(var):
#   This function will not be able to find the variable name from a different scope
#  (like the caller's scope). So, it might not return the expected variable names.
    all_vars = {**globals(), **locals()}
    for name, value in all_vars.items():
        if var is value:
            return name
    return "Variable not found"

def data_summary_inp(func_name, data, NUM_SAMPLES=3, verbose=1):
    """
      Check for the type of the data object. States / summarises the object details.
        Prints out a brief sample as appropriate
    :param func_name: Calling function name
    :param data:      Data object to be described
    :param NUM_SAMPLES: Number of samples to be printed from object details eg numpy arr values
    :param verbose:   Unused at present. Could be used to expand output at later date
    :return:          Prints object type, description and summary details
    Returns:
       Calling function name as entered where it is called
       Variable name (often returns "var" because of scoping issues)
       Variable type
     Raises:
       TypeError: If func_name is not a string.
       ValueError: If NUM_SAMPLES not an integer.
    """

    print("\nCalled from function ", func_name)

    if isinstance(data, pd.DataFrame):
        # List columns and types
        print("Dataframe Summary of ", find_var_name(data), "is:")
        pd.DataFrame.info(data)
        # Lists summary stats for summary fields
        print("Dataframe Listing of ", find_var_name(data), "is:")
        pd.DataFrame.describe(data)
        
    elif isinstance(data, (int, float)):
        print(f"Value of {find_var_name(data)} is: {data}")

    elif isinstance(data, list):
        if len(data) > NUM_SAMPLES:
            print("List listing of ", find_var_name(data), "is:")
            print(data[:NUM_SAMPLES])  # First NUM_SAMPLES elements
        elif len(data) > NUM_SAMPLES:
            print("List listing of ", find_var_name(data), "is:")
            print(data[:])  # All elements

    elif isinstance(data, np.ndarray):
        # List columns and types
        print("Array Summary of ", find_var_name(data), "is:")
        print(data.shape)

        # Print a listing depending on the shape of the array
        if len(data.shape) == 1:  # 1-D array
            print("Array Listing of ", find_var_name(data), "is:")
            print(data[:NUM_SAMPLES])  # First NUM_SAMPLES elements
        elif len(data.shape) == 2:  # 2-D array
            print("First ", NUM_SAMPLES," rows of Array Listing of ", find_var_name(data), "is:")
            print(data[:NUM_SAMPLES, :])  # First NUM_SAMPLES rows
        else:  # Higher dimensional array
            print("Array Listing of ", find_var_name(data), "is too large to display in full.")
            print("Showing slice for the first element in each dimension:")
            print(data[(slice(0, 1),) * len(data.shape)])

    else:
    # Caters for no specific data listing case
        print("Variable listing not provided for this type: ", type(data))

# Describe batches of feature - target pairs
def describe_windowed_dataset(windows):
    # Extract a few samples from the dataset
    samples = next(iter(windows.take(1)))
    features, targets = samples

    # Describe the dataset
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}\n")

    # Display a few sample feature-target pairs
    print("Sample feature-target pairs:\n")
    for i in range(min(NUM_W_SAMPLES, features.shape[0])):
        print(f"Feature {i+1}: {features[i].numpy()}")
        print(f"Target {i+1}: {targets[i].numpy()}\n")
    
################## Programmatically returns the input expected in the first line of a model #################################

def model_input_shape(model):
    # Brendon
    """
    Queries a compiled model for the full input shape to the first layer
      Developed because I kept not understanding why my data was being
      rejected by a model
    :param model: compiled model
    :return:      full input shape to the first layer
    """

    # For input shape of the first layer
    input_shape = model.layers[0].input_shape
    
    print("Input shape to model: ", input_shape)
    return input_shape

#################################### Passing an array with random content through the model #################################
# 

def input_output_shape(model, first_dim, second_dim=0):
# Code to pass a data shape through a model and return the input and output dimensions
    """
    Generates random data in the 1 or 2D shape specified and passes it into the model
      to determine whether it is accepted by the model
    Helps diagnose data shape problems icw model.shape()
      :param model:       (str) Compiled model
      :param first_dim:   (int) First dimension
      :param second_dim:  (int) Second dimension (optional)
      :return:            (tuple?) Shape of the data that emerges from the final layer
"""
    # Generate a random numpy array of the specified dimensions
    if (second_dim == 0):
        random_array = np.random.random((first_dim))
    else:
        random_array = np.random.random((first_dim, second_dim))

    # Add a blank batch dimension
    random_array_with_batch = np.expand_dims(random_array, axis=0)

    # Pass the array through the model
    output = model.predict(random_array_with_batch)

    #debug
    print("Input and ouput shapes are: ", random_array_with_batch.shape, " and ", output.shape)

    return output.shape

################################### TS Plotting #####################################################
def plot_line(x_series, y_series1, y_series2, x_axis_label="Time", y_axis_label="Value", y_series1_label="Series 1",
              y_series2_label="Series 2", title="Time Series"):
    """
    This is a function that plots two lines on a graph, against the same x series.

    Args:
        x_series (numpy array) : Usually a time series.
        y_series1 (numpy array): The first set of time series values.
        y_series2 (numpy array): The second set of time series values.
        x_series_label:         : Label for the x axis (optional)
        y_series_label:         : Label for the y axis (optional)
        y_series1_label:        : Label for the first time series (optional).
        y_series2_label:        : Label for the second time series (optional)
        title:                  : Label for the graph (optional)

    Returns:
        None. Plots a graph.

    Raises:
        Not implemented
        ValueError: If param1 is not an integer.
        TypeError: If param2 is not a string.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(x_series, y_series1, label=y_series1_label)
    plt.plot(x_series, y_series2, label=y_series2_label)
    plt.xlabel("x_axis_label")
    plt.ylabel("y_axis_label")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()


#################################### Metric Plotting ################################################
# All plots together on one canvas

import matplotlib.pyplot as plt

def plot_train_val(history_fit):
    """
    Function iterates over all metrics in the history, and plots them for training and validation
    Does not fail if validation data is not specified in the model.fit() function since
    list of "val_" metrics is simply empty, and this condition is checked before attempting to plot

    """

    history = history_fit.history
    metrics = [key for key in history.keys() if not key.startswith('val_')]
    val_metrics = ['val_' + metric for metric in metrics]

    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    if num_metrics == 1:
        axs = [axs]  # Ensure axs is iterable when there's only one metric

    for i, metric in enumerate(metrics):
        epochs = range(len(history[metric]))

        # Plot training metric
        axs[i].plot(epochs, history[metric], 'r', label=f'Training {metric}')
        # debug
        print("Training metric ", i, "is: ", metric, "\n")

        # Plot validation metric
        if 'val_' + metric in history:
            axs[i].plot(epochs, history['val_' + metric], 'b', label=f'Validation {metric}')
        
        axs[i].set_title(f'Training and Validation {metric}')
        axs[i].legend(loc='best')

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_train_val(history_fit)



