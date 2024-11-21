CONFIG = {
        "TRAIN_SIZE": 1000,  # How many (data points, labels) examples to train on
        "TEST_SIZE": 100,       # To speed up regression_performance.py
        "SAMPLE_SIZE": 30,      # Size of input layer
        "BATCH_SIZE": 2,       # How many examples to see before performing backpropagation
        "EPOCHS": 10,            # How many times to repeat the training process per generated dataset
        "LEARNING_RATE": 1e-3,  # Learning rate, for optimizer
        }

HYPERPARAMETER = "TRAIN_SIZE"
VALUES = [1e1]#, 1e4, 1e5]
VALUES = [int(i) for i in VALUES]

DEVICE = (
        # "cuda"        # Use with large networks and good GPU; requires special torch install
        # if torch.cuda.is_available()
        # else "mps"    # Should be faster, but only works with Intel CPUs
        # if torch.backends.mps.is_available()
        # else
        "cpu"           # Fastest for small networks because moving stuff to GPU is slow
        )

NUM_DIMENSIONS = 2          # How many dimensions of data we're currently working with
