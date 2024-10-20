import time
import logging
from pprint import pformat
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score 

from generate_data import generate_data

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

# How many (data points, labels) pairs to have for training/testing
TRAINING_SIZE = 50
TEST_SIZE = 10

# How many data points should be sampled from each distribution
SAMPLE_SIZE = 10

# How many times to generate new data and train model on it
RUNS = 10
# How many times to repeat the training process per generated dataset 
EPOCHS = 5

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
            logging.debug(f"Loss after training: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
            guesses.append(pred[0])
            actuals.append(y[0])

            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    logging.debug(f'guesses: \n{pformat(guesses)}')
    logging.debug(f'actuals: \n{pformat(actuals)}')
    r2 = r2_score(actuals, guesses)

    logging.info(f"R^2: {r2:>0.8f}, Avg loss: {test_loss:>8f}")

def main():
    start = time.time()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.debug(f'device: {device}')

    model = nn.Sequential(
            # out_features should be len(DISTRIBUTION_TYPES) + 2, for mean and
            # stddev
            nn.Linear(in_features=SAMPLE_SIZE, out_features=5),
            nn.ReLU(),
            nn.ReLU())
    logging.debug(model)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for r in range(RUNS):
        run_start = time.time()

        # Samples: a flat array of SAMPLE_SIZE points from a distribution
        # Labels: an array whose first n entries are a one-hot array to identify
        # distribution type, and whose last 2 entries are mean and stddev
        # E.g., [0 1 3.5 1.2] indicates that it's an Exponential distribution with
        # mean=3.5 and stddev=1.2
        raw_training_data = generate_data(count=TRAINING_SIZE, sample_size=SAMPLE_SIZE)
        logging.debug(f'raw_training_data: \n{pformat(raw_training_data)}')
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
            logging.debug(f"\nEpoch {t+1}\n-------------------------------")
            train(training_dataloader, model, loss_fn, optimizer, device)
            test(test_dataloader, model, loss_fn, device)
        run_end = time.time()
        logging.info(f"Finished run {r + 1} of {RUNS} in " +
                     f"{run_end - run_start:.2f} seconds")
   
    end = time.time()
    logging.info(f"Finished overall in {end - start:.2f} seconds")

    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    main()
