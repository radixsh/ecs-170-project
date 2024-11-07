import torch
import mpmath
from torch import nn
from loss_fn import *

# How many dimensions we're working with right now
NUM_DIMENSIONS = 1

# How many splits for crossvalidation
NUM_SPLITS = 5

# How many (data points, labels) pairs to have for training/testing
TRAINING_SIZE = 1000
TEST_SIZE = int(TRAINING_SIZE * 0.2 / 0.8) # Fixed at ~20% of training + test

# How many data points should be sampled from each distribution
SAMPLE_SIZE = 10

# How many times to generate new data and train model on it
RUNS = 3

# How many times to repeat the training process per generated dataset
EPOCHS = 5

# Define a canonical ordering (from generate_data.py on main branch)
DISTRIBUTION_TYPES = ["exponential", "normal"]
# DISTRIBUTION_TYPES = ["beta", # [0,1]
#             "gamma", # R+
#             "gumbel", # R
#             "laplace", # R
#             "logistic", # R
#             "lognormal", # R+
#             "normal", # R
#             "rayleigh", #R+
#             "wald"] #R+
NUM_DISTS = len(DISTRIBUTION_TYPES)

# Learning rate 
LEARNING_RATE = 1e-3

# Model architecture (the below indicates 1 hidden layer of 32 units)
MODEL = nn.Sequential(
        nn.Linear(in_features=SAMPLE_SIZE*NUM_DIMENSIONS, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        # out_features should be len(DISTRIBUTION_TYPES) + 2, for the one
        # hot vector for distribution type, plus mean and stddev
        nn.Linear(in_features=32, out_features=len(DISTRIBUTION_TYPES)+2))

DEVICE = (
    "cpu" # Fastest for small models! Moving stuff to GPU is costly and accel in small models is minimal
    # "mps" # Should be faster, but only works with intel CPUs
    # "cuda" # Only use with large networks + good GPU, also requires special torch install

    
    # "cuda"
    # if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available() # Possibly broken, returns 
    # else "cpu"
)

#LOSS_FN = nn.L1Loss()
LOSS_FN = CustomLoss()

OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)


# "Support fail penalty" 
# the amount to add to loss when the network TOTALLY fails the distribution type guess
SFP = 0.99

# Manually calculated: total variation distance/sqrt(2) to normalize between 0 and 1
# See: https://www.desmos.com/calculator/8h3zthas2q
# Each entry is the non-similarity of two distributions
# If the distributions have different support, use SFP
# indexed by DISTRIBUTION_TYPES
# Diagonal is always 0s
# symmetric about the diagonal
### DON'T USE THIS FOR N
# DIST_VAR_MATRIX = [
#     [0  ,SFP,SFP,SFP,SFP,SFP,SFP,SFP,SFP], # beta [0,1]
#     [SFP,0  ,SFP,SFP,SFP,0.113506763819,0.353553390593,0.217681125289,0.113036008855], # gamma R+
#     [SFP,SFP,0  ,0.122458776456,0.0933338655318,SFP,0.0968575030474,SFP,SFP], # gumbel R
#     [SFP,SFP,0.122458776456,0  ,0.065553960094,SFP,0.0999157699984,SFP,SFP], # laplace R
#     [SFP,SFP,0.0933338655318,0.065553960094,0  ,SFP,0.0384284266337,SFP,SFP], # logistic R
#     [SFP,0.113506763819,SFP,SFP,SFP,0  ,SFP,0.171801196924,0.0298717548468], # lognormal R+
#     [SFP,0.353553390593,0.0968575030474,0.0999157699984,0.0384284266337,SFP,0  ,SFP,SFP], # normal R
#     [SFP,0.217681125289,SFP,SFP,SFP,0.171801196924,SFP,0  ,0.19627978404], # rayleigh R+
#     [SFP,0.113036008855,SFP,SFP,SFP,0.0298717548468,SFP,0.19627978404,0  ]  # wald R+
# ]
DIST_VAR_MATRIX = [
    [0,0.353553390593],
    [0.353553390593,0]
]