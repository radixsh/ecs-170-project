import torch
from torch import nn

# How many (data points, labels) pairs to have for training/testing
TRAINING_SIZE = 1000
TEST_SIZE = 200

# How many data points should be sampled from each distribution
SAMPLE_SIZE = 50

# How many times to generate new data and train model on it
RUNS = 10

# How many times to repeat the training process per generated dataset
EPOCHS = 10

# Define a canonical ordering (from generate_data.py on main branch)
DISTRIBUTION_TYPES = ["exponential", "normal"]

# Model architecture (the below indicates 1 hidden layer of 32 units)
MODEL = nn.Sequential(
        nn.Linear(in_features=SAMPLE_SIZE, out_features=SAMPLE_SIZE),
        nn.ReLU(),
        # out_features should be len(DISTRIBUTION_TYPES) + 2, for the one
        # hot vector for distribution type, plus mean and stddev
        nn.Linear(in_features=SAMPLE_SIZE, out_features=len(DISTRIBUTION_TYPES)+2))

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

LOSS_FN = nn.L1Loss()

OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=1e-3)
