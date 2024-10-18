import time
import logging
import numpy as np
import torch
from torch import nn

from neural_network import NeuralNetwork
from generate_data import generate_data

logger = logging.getLogger("main")
logging.basicConfig(level=logging.DEBUG)

# How many data points should be sampled from each distribution
SAMPLE_SIZE = 5

# How many different distributions for training/testing
TRAINING_SIZE = 2
TEST_SIZE = 1

# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
def train_network(sample_size):
    start = time.time()
    training_data = generate_data(count=TRAINING_SIZE, sample_size=sample_size)
    # logging.debug(f'TRAINING DATASET:\n{pformat(training_data)}')
    training_labels = np.array([elem[0] for elem in training_data])
    training_samples = np.array([elem[1] for elem in training_data])
    logging.debug(f'Elapsed time: {time.time() - start:.4f} seconds')

    start = time.time()
    test_data = generate_data(TEST_SIZE, sample_size)
    # logging.debug(f'TEST DATASET:\n{pformat(test_data)}')
    test_labels = np.array([elem[0] for elem in test_data])
    test_samples = np.array([elem[1] for elem in test_data])
    logging.debug(f'Elapsed time: {time.time() - start:.4f} seconds')

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.debug(f'device: {device}')

    model = NeuralNetwork().to(device)
    # logging.debug(f'model:\n{pformat(model)}')
    logging.debug(model)

    # Get an item from test dataset (a set of SAMPLE_SIZE points from some dist)
    item = test_samples[0]
    logging.debug(item)
    
    layer1 = nn.Linear(in_features=5, out_features=5)
    hidden1 = layer1(item)

    logits = model(item)
    logging.debug(logits)

    predicted_probability = nn.Softmax(dim=1)(logits)
    logging.debug(predicted_probability)

    y_predicted = predicted_probability.argmax(1)
    logging.debug(y_predicted)

    logging.debug(f'Predicted y:\t{y_predicted}')

def main():
    train_network(SAMPLE_SIZE)

if __name__ == "__main__":
    main()
