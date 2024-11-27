import torch.nn as nn
from custom_functions import Multitask
import torch.nn.init as init

def initialize_weights(module):
    if isinstance(module, nn.Linear):  # For linear layers
        init.kaiming_normal_(module.weight, nonlinearity='relu')  # He initialization
        if module.bias is not None:
            init.constant_(module.bias, 0)  # Initialize biases to 0


def build_model(input_size, output_size):
    hidden_layers = [90, 180, 90, 60]
    model = nn.Sequential(
        # Don't bias the first layer, destroys mean calculations
        nn.Linear(in_features=input_size, out_features=hidden_layers[0],bias=False),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layers[0], out_features=hidden_layers[1]),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layers[1], out_features=hidden_layers[2]),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layers[2], out_features=hidden_layers[3]),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layers[3], out_features=output_size),
        # Custom activation function, softmaxes ONLY the vector portions
        Multitask()
        )
    
    model.apply(initialize_weights)
    
    return model