import torch.nn as nn
from custom_functions import Multitask

def build_model(input_size, output_size):
    model = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=input_size),
        nn.ReLU(),
        nn.Linear(in_features=input_size, out_features=30),
        nn.ReLU(),
        nn.Linear(in_features=30, out_features=30),
        nn.ReLU(),
        nn.Linear(in_features=30, out_features=30),
        nn.ReLU(),
        nn.Linear(in_features=30, out_features=30),
        nn.ReLU(),
        nn.Linear(in_features=30, out_features=30),
        nn.ReLU(),
        nn.Linear(in_features=30, out_features=30),
        nn.ReLU(),
        nn.Linear(in_features=30, out_features=output_size),
        # custom activation function, softmaxes ONLY the vector portions
        Multitask()
        )
    
    return model