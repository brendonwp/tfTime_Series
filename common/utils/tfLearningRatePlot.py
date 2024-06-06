# Generalised code to plot model dynamic learning rate  
# Last update: 23 May 2024

import tensorflow as tf
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

def learning_rate_history(compiled_model, training_dataset, 
              num_epochs = 10, base = 1e-4, exp_rate = 20):

    """
    Runs model with dynamic learning rate, and plots the result. 
    Args: compiled_model (tf model) : Compiled tensorflow model
          training_dataset (dataset)  : Dataset in format suitable for 
              model training
          num_epochs (int)            : Number of epochs to run the model
          base (float)                : Number used as the base value to
              be raised to a power and incremented per epoch
          exp_rate (int)              : Denominator of exponential change 
              co-efficient   
          e.g. base * 10**(epoch / exp_rate))
    
    Returns: model_history (history object): For analysis / assessment
    
    Raises: Not yet documented. This code will fail with some loss 
            functions and optimisers 
    Note: The minimum x axis limit is scaled by 1/10th of the range. 
          Given this is the lower end of a log scale this displaces 
          the lower end of the plot off to the right significantly 
    """

    def plot_learning_rate(lr_history):
        # Extract history data
        learning_rates = lr_history.history["lr"]
        losses = lr_history.history["loss"]

        # Determine the extent of the history data
        xmin = min(learning_rates)
        xmax = max(learning_rates)
        ymin = min(losses)
        ymax = max(losses)
        xrange = xmax-xmin
        yrange = ymax-ymin
        
        # Create the plot
        plt.semilogx(learning_rates, losses)

        # Set the axis limits
        plt.axis([xmin - (0.1* xrange), xmax + (0.1* xrange), 
                               ymin - (0.1* yrange), ymax + (0.1* yrange)])

        # Add labels and title
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate vs. Loss')

        # Show the plot
        plt.show()
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: \
        base * 10**(epoch / exp_rate))

    # Run the training with dynamic LR    
    lr_history = compiled_model.fit(training_dataset, epochs=num_epochs, 
        callbacks=[lr_schedule])
 
    plot_learning_rate(lr_history) 
   
    return lr_history

