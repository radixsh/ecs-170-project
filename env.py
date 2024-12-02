CONFIG = {
    "TRAIN_SIZE": int(1e6),  # How many (data points, labels) examples to train on
    "TEST_SIZE": int(1e5),  # How many examples to test on, should be >=10% TRAIN_SIZE
    "SAMPLE_SIZE": 30,  # Size of input layer
    "BATCH_SIZE": 1000,  # How many examples to see before performing backpropagation
    "EPOCHS": 30,  # How many times to repeat the training process per generated dataset
    "LEARNING_RATE": 1e-3,  # Learning rate, for optimizer
    "NUM_DIMENSIONS": 1,  # How many dimensions of data we're currently working with
    "DEVICE": "cpu",  # Use this unless the network is huge AND you have a good GPU
}

MODEL_ARCHITECTURE = {
    # Sizes of the hidden layers in the network that stddev and classification share
    "SHARED_LAYER_SIZES": [256, 256, 128],
    # Sizes of the hidden layers for the stddev-specific head of the network (can be empty)
    "STDDEV_LAYER_SIZES": [64, 32],
    # Sizes of the hidden layers for the classification head of the network (can be empty)
    "CLASS_LAYER_SIZES": [128, 64, 32],
}

HYPERPARAMETER = "LEARNING_RATE"
VALUES = [1e-2]
