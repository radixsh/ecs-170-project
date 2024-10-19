import time
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neural_network import NeuralNetwork
from generate_data import generate_data, MyDataset

logger = logging.getLogger("main")
logging.basicConfig(level=logging.DEBUG)

# How many data points should be sampled from each distribution
SAMPLE_SIZE = 5

# How many (input, output) pairs for training/testing
TRAINING_SIZE = 2
TEST_SIZE = 1

BATCH_SIZE = 3

def loss(predicted, actual):
    return 0 - (predicted - actual) ** 2

# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # logging.debug(f'batch: {batch}')
        X, y = X.to(device), y.to(device)
        if X.dtype != torch.float32:
            X = X.float()
        if y.dtype != torch.float32:
            y = y.float()

        # Forward pass
        # logging.debug(f'X: {X} (type: {type(X)}, shape: {X.shape})')
        pred = model(X)
        # logging.debug(f'pred: {pred} (type: {type(pred)}, shape: {pred.shape})')

        # Compute loss (prediction error)
        # logging.debug(f'y: {y} (type: {type(y)}, shape: {y.shape})')
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss after training: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, total_error = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if X.dtype != torch.float32:
                X = X.float()

            pred = model(X)
            logging.debug(f'pred: {pred}')
            logging.debug(f'actual: {y}')

            test_loss += loss_fn(pred, y).item()

            # You're not actually supposed to calculate "accuracy" of continuous
            # data.
            if y == 0:
                error = abs(pred)
            else:
                error = np.sqrt(1 - np.sqrt(pred / y))

            # sklearn.metrics.r2_score(y, pred)

            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_error += error

    test_loss /= num_batches
    accuracy = 1 - (1 / size) * total_error
    accuracy = accuracy.type(torch.float).item()

    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, " +
          f"Avg loss: {test_loss:>8f} \n")

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.debug(f'device: {device}')

    # https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

    # model = NeuralNetwork().to(device)
    # For now, this model has 3 input nodes, 0 hidden layers, and 1 output node.
    # It will look at 3 data points and predict the mean.
    model = nn.Sequential(
            nn.Linear(in_features=3, out_features=1))
    logging.debug(model)


    # raw_training_data = generate_data(count=TRAINING_SIZE, sample_size=SAMPLE_SIZE)
    raw_training_data = [([1.0, 2.0, 3.0], [2.0]),
                         ([4.0, 5.0, 6.0], [5.0])]
    training_samples = np.array([elem[0] for elem in raw_training_data])
    training_labels = np.array([elem[1] for elem in raw_training_data])
    training_dataset = MyDataset(training_samples, training_labels)
    training_dataloader = DataLoader(training_dataset)

    # raw_test_data = generate_data(TEST_SIZE, SAMPLE_SIZE)
    raw_test_data = [([3.0, 4.0, 5.0], [4.0]),
                     ([2.0, 3.0, 4.0], [3.0])]
    test_samples = np.array([elem[0] for elem in raw_test_data])
    test_labels = np.array([elem[1] for elem in raw_test_data])
    test_dataset = MyDataset(test_samples, test_labels)
    test_dataloader = DataLoader(test_dataset)

    # nn.CrossEntropyLoss() is intended for data between 0 and 1, so use MSE
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    main()
