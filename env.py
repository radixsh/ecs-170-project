CONFIG = {
        "TRAIN_SIZE": int(1e6),     # How many (data points, labels) examples to train on
        "TEST_SIZE": int(1e5),      # How many (data points, labels) examples to test on, should be >=10% TRAIN_SIZE
        "SAMPLE_SIZE": 30,          # Size of input layer
        "BATCH_SIZE": 1000,         # How many examples to see before performing backpropagation
        "EPOCHS": 30,               # How many times to repeat the training process per generated dataset
        "LEARNING_RATE": 1e-3,      # Learning rate, for optimizer
        "NUM_DIMENSIONS": 1         # How many dimensions of data we're currently working with
        }

MODEL_ARCHITECTURE = {
        "SHARED_LAYER_SIZES": [60], # Sizes of the hidden layers in the network that stddev and classification share
        "STDDEV_LAYER_SIZES": [30], # Sized of the hidden layers for the stddev-specific head of the network (can be empty)
        "CLASS_LAYER_SIZES": [30,30],  # Sized of the hidden layers for the classification head of the network (can be empty)
}

HYPERPARAMETER = "LEARNING_RATE"
VALUES = [1e-2]

DEVICE = (
        # "cuda"        # Use with large networks and good GPU; requires special torch install
        # if torch.cuda.is_available()
        # else "mps"    # Should be faster, but only works with Intel CPUs
        # if torch.backends.mps.is_available()
        # else
        "cpu"           # Fastest for small networks because moving stuff to GPU is slow
        )
