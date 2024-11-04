import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import logging
from pprint import pformat

from train import MyDataset
from generate_data import generate_data
from env import DISTRIBUTION_TYPES, SAMPLE_SIZE, TEST_SIZE

logger = logging.getLogger("meta")
logger.setLevel(logging.DEBUG)

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
            logger.debug(f"predicted: \t{pred}")
            logger.debug(f"actual: \t{y}")
            guesses.append(pred)
            actuals.append(y)

            test_loss += loss_fn(pred, y).item()
            logger.debug(f"loss: \t{loss_fn(pred, y)}")

    test_loss /= num_batches
    logger.info(f"Avg loss: {test_loss:>8f}")

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    loss_fn = nn.MSELoss()

    # Load existing model, and run it for TEST_SIZE tests
    model = nn.Sequential(
            nn.Linear(in_features=SAMPLE_SIZE, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=len(DISTRIBUTION_TYPES)+2),
            )
    weights_path = 'model_weights.pth'
    model.load_state_dict(torch.load(weights_path))

    raw_test_data = generate_data(count=TEST_SIZE, sample_size=SAMPLE_SIZE)
    test_samples = np.array([elem[0] for elem in raw_test_data])
    test_labels = np.array([elem[1] for elem in raw_test_data])
    test_dataset = MyDataset(test_samples, test_labels)
    test_dataloader = DataLoader(test_dataset)

    # Run it on a slightly modified test() function in this file (the only
    # difference is that it gives more verbose logging)
    test(test_dataloader, model, loss_fn, device)

if __name__ == "__main__":
    main()
