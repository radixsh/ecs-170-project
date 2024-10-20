import numpy as np
import logging
from pprint import pformat

logger = logging.getLogger("generate_data")
logging.basicConfig(level=logging.INFO)

# Define a canonical ordering
DISTRIBUTION_TYPES = ["binomial", "exponential", "normal"]

# Get a tuple for our training dataset: (data points, labels)
def binomial(sample_size) -> (list, list):
    rng = np.random.default_rng()

    # Generate n from Rayleigh distribution (smooth Poisson distribution),
    # similarly to how we generated stddev for normal distributions
    n = np.ceil(rng.rayleigh(scale=100.0))

    # Generate p uniformly from [0.0, 1.0)
    p = rng.uniform(low=0.0, high=1.0)

    # The first n entries are a one-hot indicator of distribution type,
    # following the canonical ordering of DISTRIBUTION_TYPES, while the last 2
    # entries are mean and standard deviation.
    # (Include mean and stddev to give us a unified way to compare dists:
    # https://en.wikipedia.org/wiki/Binomial_distribution)
    mean = n * p
    stddev = np.sqrt(n * p * (1 - p))
    labels = [1, 0, 0, mean, stddev]

    # Sample from the binomial distribution specified by n and p
    sample_data = rng.binomial(n, p, sample_size)

    return (sample_data, labels)

def exponential(sample_size) -> (list, list):
    rng = np.random.default_rng()

    # Draw from Rayleigh distribution with sigma=1
    rate = rng.rayleigh(scale=1.0)

    mean = 1 / rate 
    stddev = 1 / rate
    labels = [0, 1, 0, mean, stddev] 

    # Get `sample_size` points from normal distribution with given specs
    sample_data = rng.exponential(rate, sample_size)

    return (sample_data, labels)

def normal(sample_size) -> (list, list):
    rng = np.random.default_rng()

    # numpy.random.Generator.uniform() gives a value in [0.0, 1.0),
    # so multiply by a sign bit to get values in (-1.0, 1.0)
    sign = rng.choice([-1, 1])
    mean = rng.uniform() * sign

    # Draw from Rayleigh distribution with sigma=1
    stddev = rng.rayleigh(scale=1.0)

    labels = [0, 0, 1, mean, stddev]

    # Get `sample_size` points from the normal distribution with given specs
    sample_data = rng.normal(mean, stddev, sample_size)

    return (sample_data, labels)


'''
Generate a dataset where each piece of data is an ordered pair:
    (set of random sampled points from the distribution,
     distribution type and parameters)

Parameters:
`count`: size of dataset (number of trials)
`sample_size`: number of points we will show the AI network for each trial
'''
def generate_data(count, sample_size):
    data = []

    for _ in range(count):
        for dist in DISTRIBUTION_TYPES:
            if dist == 'binomial':
                data_piece = binomial(sample_size)
            elif dist == 'exponential':
                data_piece = exponential(sample_size)
            elif dist == 'normal':
                data_piece = normal(sample_size)
            logger.debug(f'data_piece: \n{pformat(data_piece)}\n')
            data.append(data_piece)

    return data
