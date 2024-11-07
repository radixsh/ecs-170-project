import numpy as np
import logging
from pprint import pformat

from env import DISTRIBUTION_TYPES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def exponential(sample_size):
    rng = np.random.default_rng()

    # Draw from Rayleigh distribution with sigma=1
    rate = rng.rayleigh(scale=1.0)

    mean = 1 / rate 
    stddev = 1 / rate
    labels = [1, 0, mean, stddev] 

    # Get `sample_size` points from normal distribution with given specs
    sample_data = rng.exponential(rate, sample_size)

    return (sample_data, labels)

def normal(sample_size):
    rng = np.random.default_rng()

    # numpy.random.Generator.uniform() gives a value in [0.0, 1.0),
    # so multiply by a sign bit to get values in (-1.0, 1.0)
    sign = rng.choice([-1, 1])
    mean = rng.uniform() * sign

    # Draw from Rayleigh distribution with sigma=1
    stddev = rng.rayleigh(scale=1.0)

    labels = [0, 1, mean, stddev]

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
            if dist == 'exponential':
                data_piece = exponential(sample_size)
            elif dist == 'normal':
                data_piece = normal(sample_size)
            logger.debug(f'data_piece: \n{pformat(data_piece)}\n')
            data.append(data_piece)

    return data
