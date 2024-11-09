import torch.nn as nn
from env import INPUT_SIZE, OUTPUT_SIZE

def build_model():
    model = nn.Sequential(
        nn.Linear(in_features=INPUT_SIZE, out_features=INPUT_SIZE),
        nn.ReLU(),
        nn.Linear(in_features=INPUT_SIZE, out_features=30),
        nn.ReLU(),
        nn.Linear(in_features=30, out_features=30),
        nn.ReLU(),
        # for each dimension, n distribution types + mean + stddev
        nn.Linear(in_features=30, out_features=OUTPUT_SIZE))
    return model