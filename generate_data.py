import numpy as np
import logging
from pprint import pformat

logger = logging.getLogger("generate_data")

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

# Get a tuple for our training dataset
def binomial(sample_size) -> (list, list):
    rng = np.random.default_rng()

    # Generate n from Rayleigh distribution (smooth Poisson distribution),
    # similarly to how we generated stddev for normal distributions
    n = np.ceil(rng.rayleigh(scale=100.0))

    # Generate p uniformly from [0.0, 1.0)
    p = rng.uniform(low=0.0, high=1.0)

    # Save these values as inputs to the reward function
    # (Include mean and stddev to give us a unified way to compare dists:
    # https://en.wikipedia.org/wiki/Binomial_distribution)
    specs = {"distribution_type": "binomial",
             "n": n,
             "p": p,
             "mean": n * p,
             "stddev": np.sqrt(n * p * (1 - p))}

    # Sample from the binomial distribution specified by n and p
    sample_data = rng.binomial(n, p, sample_size)

    return (sample_data, specs)


def normal(sample_size) -> (list, list):
    rng = np.random.default_rng()

    # numpy.random.Generator.uniform() gives a value in [0.0, 1.0),
    # so multiply by a sign bit to get values in (-1.0, 1.0)
    sign = rng.choice([-1, 1])
    mean = rng.uniform() * sign

    # Draw from Rayleigh distribution with sigma=1
    stddev = rng.rayleigh(scale=1.0)

    # Save these values (mean and stddev) as inputs to the reward function
    specs = {"distribution_type": "normal",
             "mean": mean,
             "stddev": stddev}

    # Get `sample_size` points from normal distribution with given specs
    sample_data = rng.normal(mean, stddev, sample_size)

    return (sample_data, specs)


def exponential(sample_size) -> (list, list):
    rng = np.random.default_rng()

    # Draw from Rayleigh distribution with sigma=1
    rate = rng.rayleigh(scale=1.0)

    # Save these values (mean and stddev) as inputs to the reward function
    specs = {"distribution_type": "normal",
             "rate": rate,
             "mean": 1 / rate,
             "stddev": 1 / rate}

    # Get `sample_size` points from normal distribution with given specs
    sample_data = rng.exponential(rate, sample_size)

    return (sample_data, specs)


'''
Generate a dataset where each piece of data is an ordered pair:
    (distribution type and parameters,
    set of random sampled points from the distribution)

Parameters:
`count`: size of dataset (number of trials)
`sample_size`: number of points we will show the AI network for each trial
'''
def generate_data(count, sample_size):

    data = []
    distributions = ['normal', 'binomial', 'exponential']

    for _ in range(count):
        for dist in distributions:
            if dist == 'normal':
                data, labels = normal(sample_size)
            elif dist == 'binomial':
                data_piece = binomial(sample_size)
            elif dist == 'exponential':
                data_piece = exponential(sample_size)
            # logger.info(f'data_piece: \n{pformat(data_piece)}\n')
            data.append(data_piece)

    return data

def main():
    training_data = generate_training_data(count=1, sample_size=5)
    print(f'Dataset generated for training:\n{pformat(training_data)}')

if __name__ == "__main__":
    main()
