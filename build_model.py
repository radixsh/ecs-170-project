import torch.nn as nn

def build_model(input_size, output_size):
    # Model architecture; do not chance first in_features or last out_features
    model = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=input_size),
        nn.ReLU(),
        nn.Linear(in_features=input_size, out_features=30),
        nn.ReLU(),
        nn.Linear(in_features=30, out_features=30),
        nn.ReLU(),
        # for each dimension, n distribution types + mean + stddev
        nn.Linear(in_features=30, out_features=output_size))
    return model