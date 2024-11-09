import torch
from torch import nn
from loss_fn import *
from generate_data import *
from distributions import (
    beta_dist,
    gamma_dist,
    gumbel_dist,
    laplace_dist,
    logistic_dist,
    lognormal_dist,
    normal_dist,
    rayleigh_dist,
    wald_dist,
)

# Define a canonical ordering (from generate_data.py on main branch)
DISTRIBUTION_FUNCTIONS = {
    "beta": beta_dist,
    "gamma": gamma_dist,
    "gumbel": gumbel_dist,
    "laplace": laplace_dist,
    "logistic": logistic_dist,
    "lognormal": lognormal_dist,
    "normal": normal_dist,
    "rayleigh": rayleigh_dist,
    "wald": wald_dist,
}

# Training constants
TRAINING_SIZE = 800    # How many (data points, labels) examples to train on
TEST_SIZE = TRAINING_SIZE # Equal to training size since we can generate arbitrary amounts of data
SAMPLE_SIZE = 30        # How many data points should be shown to the network
NUM_SPLITS = 2 # Less is more for large datasets, but potentially play with this   
EPOCHS = 5              # How many times to repeat the training process per generated dataset
RUNS = 1000                # How many times to generate new data and train model on it
NUM_DISTS = len(DISTRIBUTION_FUNCTIONS)
MEAN_SCALE = 10 # Scale of means for data
NUM_DIMENSIONS = 1      # How many dimensions of data we're currently working with
INPUT_SIZE = SAMPLE_SIZE*NUM_DIMENSIONS
OUTPUT_SIZE = (NUM_DISTS+2)*NUM_DIMENSIONS
LOSS_FN = CustomLoss()  # Custom loss function defined in loss_fn.py
LEARNING_RATE = 1e-3    # Learning rate, for optimizer
DEVICE = (
        # "cuda"        # Use with large networks and good GPU; requires special torch install
        # if torch.cuda.is_available()
        # else "mps"    # Should be faster, but only works with Intel CPUs
        # if torch.backends.mps.is_available()
        # else
        "cpu"           # Fastest for small networks because moving stuff to GPU is slow
        )
