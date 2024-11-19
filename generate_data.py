import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
import math
import mpmath

from env import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

### SEED THE RNG
rng = np.random.default_rng()

### HELPERS
# Returns a uniformly random value in (-1, 1)
def generate_mean():
    return rng.normal(0, 1)

# Generates a standard deviation according lognormal with mean and stddev 1
def generate_stddev():
    return rng.lognormal(mean= -math.log(2) / math.sqrt(2), sigma=math.sqrt(math.log(2)))

# Generates a standard deviation according lognormal with mean and stddev 1
def generate_pos_mean():
    return rng.lognormal(mean= -math.log(2) / math.sqrt(2), sigma=math.sqrt(math.log(2)))

## SUPPORT = R

def normal(sample_size):
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,0,1,0,0,mean,stddev]

    sample_data = rng.normal(mean, stddev, sample_size)
    return (sample_data, labels)

def gumbel(sample_size):
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,1,0,0,0,0,0,0,mean,stddev]

    # stddev^2 = (pi^2 beta^2) / 6
    # beta = stddev * sqrt(6) / pi
    beta = stddev * math.sqrt(6) / math.pi

    # euler is the Euler-Mascheroni constant
    # mean = mu + beta * euler
    # mu = mean - beta * euler
    mu = mean - beta * float(mpmath.euler)

    sample_data = rng.gumbel(mu, beta, sample_size)
    return (sample_data, labels)

def laplace(sample_size):
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,0,1,0,0,0,0,0,mean,stddev]

    # stddev^2 = 2b^2
    # b = sqrt(2)*stddev
    b = stddev * math.sqrt(2)
    sample_data = rng.laplace(mean, b, sample_size)
    return (sample_data, labels)

def logistic(sample_size):
    mean = generate_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,1,0,0,0,0,mean,stddev]

    # stddev^2 = pi^2 s^2 * 1/3
    # s = sqrt(3)/pi * stddev
    s = stddev * math.sqrt(3) / math.pi
    sample_data = rng.logistic(mean, s, sample_size)
    return (sample_data, labels)


## SUPPORT >= 0

def lognormal(sample_size):
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,1,0,0,0,mean,stddev]

    # mean = exp(mu + sigma^2/2)
    # stddev^2 = (exp(sigma^2) - 1) exp(2 mu - sigma^2)
    sigma = math.sqrt(math.log(1 + ((stddev / mean) ** 2)))
    mu = math.log((mean ** 2) / math.sqrt((mean ** 2) + (stddev ** 2)))
    sample_data = rng.lognormal(mu, sigma, sample_size)
    return (sample_data, labels)

def rayleigh(sample_size):
    mean = generate_pos_mean()
    # mean = sigma sqrt(pi/2)
    # sigma = mean * sqrt(2/pi)
    sigma = mean * math.sqrt(2 / math.pi)
    stddev = mean * math.sqrt((4 / math.pi)  - 1)
    labels = [0,0,0,0,0,0,0,1,0,mean,stddev]

    # One parameter, which must be positive
    sample_data = rng.rayleigh(sigma, sample_size)
    return (sample_data, labels)

## SUPPORT > 0
# Non-positive mean is illegal for these distributions

def beta(sample_size):
    # Need a mean in (0, 1) 
    sign1 = rng.choice([-1, 1])
    mean1 = rng.uniform() * sign1
    mean = (mean1 + 1) / 2
    # Need a stddev in (0, mean - mean^2)
    sign2 = rng.choice([-1, 1])
    mean2 = rng.uniform() * sign2
    stddev = ((mean2 + 1) / 2) * (mean - (mean ** 2))
    labels = [1,0,0,0,0,0,0,0,0,mean,stddev]

    a = math.sqrt((((mean ** 2) - (mean ** 3)) / stddev) - mean)
    b = (a / mean) - a
    sample_data = rng.beta(a, b, sample_size)
    return (sample_data, labels)

def gamma(sample_size):
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,1,0,0,0,0,0,0,0,mean,stddev]

    k = (mean / stddev) ** 2
    theta = (stddev ** 2) / mean
    sample_data = rng.gamma(k, theta, sample_size)
    return (sample_data, labels)

def wald(sample_size):
    mean = generate_pos_mean()
    stddev = generate_stddev()
    labels = [0,0,0,0,0,0,0,0,1,mean,stddev]

    # stddev^2 = mean^3 / lam
    # lam = mean^3 / stddev^2
    lam = (mean ** 3) / (stddev ** 2)
    sample_data = rng.wald(mean, lam, sample_size)
    return (sample_data, labels)

DISTRIBUTION_FUNCTIONS = {
    "beta": beta,
    "gamma": gamma,
    "gumbel": gumbel,
    "laplace": laplace,
    "logistic": logistic,
    "lognormal": lognormal,
    "normal": normal,
    "rayleigh": rayleigh,
    "wald": wald,
}

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

def get_dataloader(config, filename=None, examples_count=-1):
    try:
        dataset = torch.load(filename)

        # If examples_count is passed in, then adhere to it. Otherwise,
        # examples_count is default value, indicating dataset can have any size
        acceptable_count = (len(dataset) == examples_count \
                            or examples_count == -1)

        dataset_sample_size = len(dataset.__getitem__(0)[0])
        acceptable_sample_size = (config['SAMPLE_SIZE'] == dataset_sample_size)

        if not isinstance(dataset, Dataset):
            raise Exception(f'Could not read dataset from {filename}')
        if not acceptable_count:
            raise Exception(f"Incorrect dataset size: {len(dataset)}")
        if not acceptable_sample_size:
            raise Exception(f"Incorrect sample size: model expects "
                            f"{config['SAMPLE_SIZE']} points as input, but "
                            f"this dataset would give {dataset_sample_size}")

        logger.info(f"Using dataset from {filename} (size: {len(dataset)})")

    except Exception as e:
        logger.info(e)
        logger.info(f'Generating fresh data...')
        dataset = make_dataset(filename, examples_count)

    # If no filename is passed in, the file does not exist, or the file's
    # contents do not represent a DataLoader as expected, then generate some
    # new data, and write it out to the given filename
    dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'])
    return dataloader

# If running this file directly as a script, then generate some training
# examples and save them to a file for later use
if __name__ == "__main__":
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)

    make_dataset('data/train_dataset', examples_count=CONFIG['TRAINING_SIZE'])
    make_dataset('data/test_dataset', examples_count=CONFIG['TEST_SIZE'])
