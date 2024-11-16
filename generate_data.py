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

def make_dataloader(filename, config, count):
    # https://pytorch.org/docs/stable/generated/torch.save.html
    raw_data = generate_data(count=count, sample_size=CONFIG['SAMPLE_SIZE'])
    samples = np.array([elem[0] for elem in raw_data])
    labels = np.array([elem[1] for elem in raw_data])
    dataset = MyDataset(samples, labels)
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'])
    torch.save(dataloader, filename) 

# If running this file directly as a script, then generate some training
# examples and save them to a file for later use
if __name__ == "__main__":
    # Make sure the 'data' directory exists
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)

    make_dataloader('data/train_dataloader', CONFIG, CONFIG['TRAINING_SIZE'])
    make_dataloader('data/test_dataloader', CONFIG, CONFIG['TEST_SIZE'])
