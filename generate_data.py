import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from distributions import DISTRIBUTION_FUNCTIONS
from env import CONFIG
from dataset import MyDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_data(count, sample_size):
    data = []
    for _ in range(count):
        for dist_name, dist_func in DISTRIBUTION_FUNCTIONS.items():
            points, labels = dist_func(sample_size)
            data.append((points, labels))
    return data

# If running this file directly as a script, then generate some training
# examples and save them to a file for later use
if __name__ == "__main__":
    # Make sure the 'data' directory exists
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)

    # https://pytorch.org/docs/stable/generated/torch.save.html
    raw_train_data = generate_data(count=CONFIG['TRAINING_SIZE'],
                                   sample_size=CONFIG['SAMPLE_SIZE'])
    train_samples = np.array([elem[0] for elem in raw_train_data])
    train_labels = np.array([elem[1] for elem in raw_train_data])
    train_dataset = MyDataset(train_samples, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'])
    torch.save(train_dataloader, 'data/train_dataloader')

    raw_test_data = generate_data(count=CONFIG['TRAINING_SIZE'],
                                  sample_size=CONFIG['SAMPLE_SIZE'])
    test_samples = np.array([elem[0] for elem in raw_test_data])
    test_labels = np.array([elem[1] for elem in raw_test_data])
    test_dataset = MyDataset(test_samples, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'])
    torch.save(test_dataloader, 'data/test_dataloader')

    print(f"Generated {CONFIG['TRAINING_SIZE']} training examples in "
          f"'data/train_dataloader' and {CONFIG['TEST_SIZE']} test examples in "
          f"'data/test_dataloader'")
