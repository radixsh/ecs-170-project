CONFIG = {
        "TRAINING_SIZE": 1000,  # How many (data points, labels) examples to train on
        "TEST_SIZE": 100,       # To speed up regression_performance.py
        "SAMPLE_SIZE": 10,      # Size of input layer
        "BATCH_SIZE": 32,       # How many examples to see before performing backpropagation
        "EPOCHS": 5,            # How many times to repeat the training process per generated dataset
        "LEARNING_RATE": 1e-3,  # Learning rate, for optimizer
        "ALPHA": 2,             # Increases how much of the loss calculation
                                # comes from classification task. Should be > 0.
        }

HYPERPARAMETER = "TRAINING_SIZE"

DEVICE = (
        # "cuda"        # Use with large networks and good GPU; requires special torch install
        # if torch.cuda.is_available()
        # else "mps"    # Should be faster, but only works with Intel CPUs
        # if torch.backends.mps.is_available()
        # else
        "cpu"           # Fastest for small networks because moving stuff to GPU is slow
        )

MEAN_SCALE = 10             # Data we generate will have means between -10 and 10
NUM_DIMENSIONS = 1          # How many dimensions of data we're currently working with
