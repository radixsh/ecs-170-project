import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

from train import MyDataset, test
from generate_data import generate_data
from env import DISTRIBUTION_TYPES, SAMPLE_SIZE, TEST_SIZE, MODEL, LOSS_FN, DEVICE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def main():
    weights_path = 'model_weights.pth'
    MODEL.load_state_dict(torch.load(weights_path))

    avg_loss = 0
    tests = 20
    for _ in range(tests):
        raw_test_data = generate_data(count=TEST_SIZE)
        test_samples = np.array([elem[0] for elem in raw_test_data])
        test_labels = np.array([elem[1] for elem in raw_test_data])
        test_dataset = MyDataset(test_samples, test_labels)
        test_dataloader = DataLoader(test_dataset)

        loss = test(test_dataloader, MODEL, LOSS_FN, DEVICE)
        avg_loss += loss
        logger.info(f"Loss: {loss}")
    avg_loss /= tests
    logger.info(f"Avg loss over {tests} tests: {avg_loss}")

if __name__ == "__main__":
    main()
