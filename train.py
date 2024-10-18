import time
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neural_network import NeuralNetwork
from generate_data import generate_data

logger = logging.getLogger("main")
logging.basicConfig(level=logging.DEBUG)

# How many data points should be sampled from each distribution
SAMPLE_SIZE = 5

# How many different distributions for training/testing
TRAINING_SIZE = 2
TEST_SIZE = 1

BATCH_SIZE = 3

# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    start = time.time()
    training_data = generate_data(count=TRAINING_SIZE, sample_size=SAMPLE_SIZE)
    # logging.debug(f'TRAINING DATASET:\n{pformat(training_data)}')
    training_labels = np.array([elem[0] for elem in training_data])
    training_samples = np.array([elem[1] for elem in training_data])
    logging.debug(f'Elapsed time: {time.time() - start:.4f} seconds')

    start = time.time()
    test_data = generate_data(TEST_SIZE, SAMPLE_SIZE)
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
    logging.debug(model)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    logging.debug(train_dataloader)
    train_features, train_labels = next(iter(train_dataloader))
    logging.debug(train_features)
    logging.debug(train_labels)

    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    test_features, test_labels = next(iter(test_dataloader))
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    main()
