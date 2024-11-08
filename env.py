import torch
from torch import nn
from loss_fn import *

TRAINING_SIZE = 1000    # How many (data points, labels) examples to train on
TEST_SIZE = int(TRAINING_SIZE * 0.2 / 0.8)      # 20% of (training + test)
SAMPLE_SIZE = 30        # How many data points should be shown to the network
NUM_SPLITS = 5          # How many splits for cross-validation
EPOCHS = 5              # How many times to repeat the training process per generated dataset
RUNS = 3                # How many times to generate new data and train model on it

DISTRIBUTION_TYPES = ["exponential", "normal"]
'''
DISTRIBUTION_TYPES = ["beta", # [0,1]
            "gamma", # R+
            "gumbel", # R
            "laplace", # R
            "logistic", # R
            "lognormal", # R+
            "normal", # R
            "rayleigh", #R+
            "wald"] #R+
'''
NUM_DISTS = len(DISTRIBUTION_TYPES)

NUM_DIMENSIONS = 1      # How many dimensions of data we're currently working with

# Model architecture (currently, 2 hidden layers of 64 and 32 units)
MODEL = nn.Sequential(
        nn.Linear(in_features=SAMPLE_SIZE*NUM_DIMENSIONS, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        # out_features should be len(DISTRIBUTION_TYPES) + 2, for the one
        # hot vector for distribution type, plus mean and stddev
        nn.Linear(in_features=32, out_features=len(DISTRIBUTION_TYPES)+2))

DEVICE = (
        # "cuda"        # Use with large networks and good GPU; requires special torch install
        # if torch.cuda.is_available()
        # else "mps"    # Should be faster, but only works with Intel CPUs
        # if torch.backends.mps.is_available()
        # else
        "cpu"           # Fastest for small networks because moving stuff to GPU is slow
        )

LOSS_FN = CustomLoss()  # Custom loss function defined in loss_fn.py
LEARNING_RATE = 1e-3    # Learning rate, for optimizer
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)

SFP = 0.99  # "Support fail penalty": the amount to add to loss when the network
            # TOTALLY guessues wrong for distribution type. This is going away

# Manually calculated: (total variation distance) / sqrt(2) to normalize between 0 and 1
# See: https://www.desmos.com/calculator/8h3zthas2q
DIST_VAR_MATRIX = [
    [0, 0.353553390593],
    [0.353553390593, 0]
]
'''
# Indexed by DISTRIBUTION_TYPES
# Each entry is the non-similarity of two distributions
# If the distributions have different support, use SFP
# Symmetric about the diagonal (which is all 0s)
DIST_VAR_MATRIX = [
    [0  ,SFP,SFP,SFP,SFP,SFP,SFP,SFP,SFP], # beta [0,1]
    [SFP,0  ,SFP,SFP,SFP,0.113506763819,0.353553390593,0.217681125289,0.113036008855], # gamma R+
    [SFP,SFP,0  ,0.122458776456,0.0933338655318,SFP,0.0968575030474,SFP,SFP], # gumbel R
    [SFP,SFP,0.122458776456,0  ,0.065553960094,SFP,0.0999157699984,SFP,SFP], # laplace R
    [SFP,SFP,0.0933338655318,0.065553960094,0  ,SFP,0.0384284266337,SFP,SFP], # logistic R
    [SFP,0.113506763819,SFP,SFP,SFP,0  ,SFP,0.171801196924,0.0298717548468], # lognormal R+
    [SFP,0.353553390593,0.0968575030474,0.0999157699984,0.0384284266337,SFP,0  ,SFP,SFP], # normal R
    [SFP,0.217681125289,SFP,SFP,SFP,0.171801196924,SFP,0  ,0.19627978404], # rayleigh R+
    [SFP,0.113036008855,SFP,SFP,SFP,0.0298717548468,SFP,0.19627978404,0  ]  # wald R+
]
'''
