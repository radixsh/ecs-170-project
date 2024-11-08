import time
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from build_model import build_model
from generate_data import generate_data
from env import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

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

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model = model.to(device)        # For GPU use
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

            # TODO: Do stuff with guesses and actuals
            # implement some metrics here

            test_loss += loss_fn(pred, y).item()
            logger.debug(f"loss: \t{loss_fn(pred,y)}")

    test_loss /= num_batches
    return test_loss

def main():
    # consistent initialization
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    start = time.time()

    for r in range(RUNS):
        # rebuild model and optimizer each run
        model = build_model(INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, foreach=True)
        run_start = time.time()

        # Generate the entire dataset first
        raw_data = generate_data(count=TRAINING_SIZE)
        samples = np.array([elem[0] for elem in raw_data])
        labels = np.array([elem[1] for elem in raw_data])

        # Split the data set into training and validation sets for
        # NUM_SPLITS-fold cross-validation.
        # The entire dataset is randomly divided into NUM_SPLITS equal subsets
        # (folds), each of which is used exactly once as validation while the
        # k - 1 remaining folds form the training set.
        # This reduces bias, gives our model more data to train on, and helps us
        # evaluate our model's performance.
        kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=42)
        for epoch in range(EPOCHS):
            for fold, (train_index, val_index) in enumerate(kf.split(samples)):
                training_samples, validation_samples = samples[train_index], samples[val_index]
                training_labels, validation_labels = labels[train_index], labels[val_index]

                training_dataset = MyDataset(training_samples, training_labels)
                training_dataloader = DataLoader(training_dataset)
                validation_dataset = MyDataset(validation_samples, validation_labels)
                validation_dataloader = DataLoader(validation_dataset)

                # Train the model on the training data, and cross-validate
                train(training_dataloader, model, LOSS_FN, optimizer, DEVICE)
                loss = test(validation_dataloader, model, LOSS_FN, DEVICE)
                logger.info(f"Epoch {epoch + 1}\tFold {fold + 1}\t"
                            f"Avg loss (cross-validation phase): {loss}")

        # Test the model on the test data
        raw_test_data = generate_data(count=TEST_SIZE)
        test_samples = np.array([elem[0] for elem in raw_test_data])
        test_labels = np.array([elem[1] for elem in raw_test_data])
        test_dataset = MyDataset(test_samples, test_labels)
        test_dataloader = DataLoader(test_dataset)
        loss = test(test_dataloader, model, LOSS_FN, DEVICE)
        logger.info(f"Avg loss (testing): {loss}")

        run_end = time.time()
        logger.info(f"Finished run {r + 1} of {RUNS} in " +
                     f"{run_end - run_start:.2f} seconds")

    end = time.time()
    logger.info(f"Finished {RUNS} runs in {end - start:.2f} seconds")

    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    main()
