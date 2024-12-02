import time
import os
import re
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from distributions import DISTRIBUTIONS, NUM_DISTS

### Illegal imports:
# env, core, metrics, generate_data, sanity_check, performance, train_multiple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def get_dataset(config, mode):
    """
    Searches for an existing dataset which is compatible with the current config.
    If none is found, generates and saves a new dataset which is usable.

    Args:
        config (dict): Determines what dataset to search for or create.
        mode (str): "TRAIN" or "TEST", controls whether "TRAIN_SIZE" or "TEST_SIZE"
            is the desired amount of data.

    Returns:
        Dataset object: A torch Dataset object with data compatible with config.
    """
    data_size = config[f"{mode}_SIZE"]
    sample_size = config["SAMPLE_SIZE"]
    num_dims = config["NUM_DIMENSIONS"]
    os.makedirs("data", exist_ok=True)

    # Attempt to find a file whose data matches the current configuration, but allow
    # for the amount of data in the file to be larger than the current required amount.
    pattern = rf"dataset_{mode}_len_(\d+)_sample_{sample_size}_dims_{num_dims}.pth"
    for filename in os.listdir("data"):
        match = re.match(pattern, filename)
        if not match:
            continue
        length = int(match.group(1))
        if length >= data_size:
            good_filename = os.path.join("data", filename)
            logger.debug(f"Found appropriate data at {good_filename}.")
            dataset = torch.load(good_filename, weights_only=False)
            return Subset(dataset, indices=range(data_size))

    # Failed to find valid data, need to generate some.
    filename = os.path.join(
        "data",
        f"dataset_{mode}_len_{data_size}_sample_{sample_size}_dims_{num_dims}.pth",
    )
    logger.debug(
        f"Generating {data_size} pieces of {num_dims}-dimensional "
        f"{mode.lower()}ing data, each with {sample_size} samples."
    )
    start = time.time()

    # Main data generation loop. Preinstantiate arrays for speed, randomly select a
    # distribution, and use its methods get a sample set of points and a label.
    samples = np.zeros((data_size, num_dims * sample_size))
    labels = np.zeros((data_size, num_dims * (NUM_DISTS + 2)))
    for idx in range(data_size):
        curr_points = np.zeros((num_dims, sample_size))
        curr_labels = np.zeros((num_dims, (NUM_DISTS + 2)))
        for dim in range(num_dims):
            dist = np.random.choice(list(DISTRIBUTIONS.values()))()
            curr_points[dim] = dist.rng(sample_size)
            curr_labels[dim] = dist.get_label()
        samples[idx] = np.ravel(curr_points)  # Equivalent to flatten
        labels[idx] = np.ravel(curr_labels)

    logger.debug(f"Generated and saved to {filename} in {time.time() - start:.3f}s")
    dataset = MyDataset(samples, labels)
    torch.save(dataset, filename)

    return dataset


def make_weights_filename(config):
    """
    Generates a filename to save the model weights in after training.
    Ex: "models/weights_train_1000_sample_30_dims_2_batch_100_lrate_0.01.pth"

    Args:
        config (dict): Dictionary in the format of CONFIG in ENV.

    Returns:
        str: A string for a filename matching the given config.
    """
    return os.path.join(
        "models",
        f"weights_train_{config['TRAIN_SIZE']}_"
        f"sample_{config['SAMPLE_SIZE']}_"
        f"dims_{config['NUM_DIMENSIONS']}_"
        f"batch_{config['BATCH_SIZE']}_"
        f"lrate_{config['LEARNING_RATE']}"
        f".pth",
    )


class MyDataset(Dataset):
    """
    Minimalist dataset for use with torch's Dataloader.
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
