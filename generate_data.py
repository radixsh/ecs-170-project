import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import time

from distributions import DISTRIBUTION_FUNCTIONS
from env import CONFIG

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

def generate_data(count, sample_size):
    data = []

    # Generate 'count' examples uniformly at random (before, it was 9 * 'count')
    for _ in range(count):
        items = list(DISTRIBUTION_FUNCTIONS.items())
        choice = np.random.choice(len(items))
        _, dist_func = items[choice]
        points, labels = dist_func(sample_size)
        data.append((points, labels))
    return data

def make_dataset(filename, examples_count):
    start = time.time()
    raw_data = generate_data(count=examples_count,
                             sample_size=CONFIG['SAMPLE_SIZE'])
    samples = np.array([elem[0] for elem in raw_data])
    labels = np.array([elem[1] for elem in raw_data])
    dataset = MyDataset(samples, labels)
    torch.save(dataset, filename)

    end = time.time()
    logger.info(f"Generated and saved {examples_count} examples out "
                f"to {filename} in {end - start:.2f} seconds "
                f"(BATCH_SIZE={CONFIG['BATCH_SIZE']})")

    return dataset

# If running this file directly as a script, then generate some training
# examples and save them to a file for later use
if __name__ == "__main__":
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)

    make_dataset('data/train_dataset', examples_count=CONFIG['TRAINING_SIZE'])
    make_dataset('data/test_dataset', examples_count=CONFIG['TEST_SIZE'])
