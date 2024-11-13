# Training constants
TRAINING_SIZE = 50         # How many (data points, labels) examples to train on
TEST_SIZE = TRAINING_SIZE   # Equal to training size since we can generate arbitrary amounts of data
SAMPLE_SIZE = 5            # How many data points should be shown to the network
NUM_SPLITS = 2              # Less is more for large datasets, but potentially play with this   
EPOCHS = 5                 # How many times to repeat the training process per generated dataset
RUNS = 1                    # How many times to generate new data and train model on it
                            # Model currently only exports the last run
MEAN_SCALE = 10             # Data we generate will have means between -10 and 10
NUM_DIMENSIONS = 1          # How many dimensions of data we're currently working with
LEARNING_RATE = 1e-3        # Learning rate, for optimizer
# Increasing this punishes the model more for guesing the wrong distribution
# Turning this up will increase loss but might decrease other metrics
# Also, tweaking it and observing how much loss changes tells you...
#       How much the model is failing on distribution guess vs mean+stddev
#       If changing it a lot doesn't change the loss much...
#               model is good at distribution guesses
# Should be > 0. 
ALPHA = 2
DEVICE = (
        # "cuda"        # Use with large networks and good GPU; requires special torch install
        # if torch.cuda.is_available()
        # else "mps"    # Should be faster, but only works with Intel CPUs
        # if torch.backends.mps.is_available()
        # else
        "cpu"           # Fastest for small networks because moving stuff to GPU is slow
        )
