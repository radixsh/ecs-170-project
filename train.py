import time
import logging
from pprint import pformat
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from generate_data import generate_data
from env import *

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label

# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()

        pred = model(X)             # Forward pass
        loss = loss_fn(pred, y)     # Compute loss (prediction error)
        loss.backward()             # Backpropagation
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.debug(f"Loss after training: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    guesses = []
    actuals = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).float()

            pred = model(X)
            logger.debug(f"pred: \t{pred}")
            logger.debug(f"y: \t\t{y}")

            test_loss += loss_fn(pred, y).item()
            logger.debug(f"loss: \t{loss_fn(pred,y)}")

    test_loss /= num_batches
    logger.info(f"Avg loss: {test_loss:>8f}")

def main():
    start = time.time()

    logger.debug(MODEL)

    for r in range(RUNS):
        run_start = time.time()

        # Samples: a flat array of SAMPLE_SIZE points from a distribution
        # Labels: an array whose first n entries are a one-hot array to identify
        # distribution type, and whose last 2 entries are mean and stddev
        # For example, [0 1 3.5 1.2] indicates an Exponential distribution with
        # mean=3.5 and stddev=1.2
        raw_training_data = generate_data(count=TRAINING_SIZE, sample_size=SAMPLE_SIZE)
        training_samples = np.array([elem[0] for elem in raw_training_data])
        training_labels = np.array([elem[1] for elem in raw_training_data])
        training_dataset = MyDataset(training_samples, training_labels)
        training_dataloader = DataLoader(training_dataset)

        raw_test_data = generate_data(count=TEST_SIZE, sample_size=SAMPLE_SIZE)
        test_samples = np.array([elem[0] for elem in raw_test_data])
        test_labels = np.array([elem[1] for elem in raw_test_data])
        test_dataset = MyDataset(test_samples, test_labels)
        test_dataloader = DataLoader(test_dataset)

        for t in range(EPOCHS):
            logger.debug(f"\nEpoch {t+1}\n-------------------------------")
            train(training_dataloader, MODEL, LOSS_FN, OPTIMIZER, DEVICE)
            test(test_dataloader, MODEL, LOSS_FN, DEVICE)
        run_end = time.time()
        logger.info(f"Finished run {r + 1} of {RUNS} in " +
                     f"{run_end - run_start:.2f} seconds")

    end = time.time()
    logger.info(f"Finished overall in {end - start:.2f} seconds")

    torch.save(MODEL.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    main()
