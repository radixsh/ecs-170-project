import torch.nn as nn
from custom_functions import Multitask
import torch.nn.init as init

def initialize_weights(module):
    if isinstance(module, nn.Linear):  # For linear layers
        init.kaiming_normal_(module.weight, nonlinearity='relu')  # He initialization
        if module.bias is not None:
            init.constant_(module.bias, 0)  # Initialize biases to 0


def build_model(input_size, output_size):
    model = nn.Sequential(
        # Don't bias the first layer, destroys mean calculations
        nn.Linear(in_features=input_size, out_features=30,bias=False),
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
    
    model.apply(initialize_weights)
    
    return model