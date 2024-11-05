import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import logging
from pprint import pformat

from train import MyDataset
from generate_data import generate_data
from env import DISTRIBUTION_TYPES, SAMPLE_SIZE, TEST_SIZE, MODEL, LOSS_FN, DEVICE

logger = logging.getLogger("analyze_performance")
logger.setLevel(logging.INFO)

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
    return test_loss

def main():
    weights_path = 'model_weights.pth'
    MODEL.load_state_dict(torch.load(weights_path))

    LOSS_FN = nn.L1Loss()
    avg_loss = 0
    tests = 20
    for _ in range(tests):
        raw_test_data = generate_data(count=TEST_SIZE, sample_size=SAMPLE_SIZE)
        test_samples = np.array([elem[0] for elem in raw_test_data])
        test_labels = np.array([elem[1] for elem in raw_test_data])
        test_dataset = MyDataset(test_samples, test_labels)
        test_dataloader = DataLoader(test_dataset)

        # Run it on a slightly modified test() function in this file (the only
        # difference is that it gives more verbose logging)
        avg_loss += test(test_dataloader, MODEL, LOSS_FN, DEVICE)
    avg_loss /= tests
    logger.info(f"Avg loss over {tests} tests: {avg_loss}")

if __name__ == "__main__":
    main()
