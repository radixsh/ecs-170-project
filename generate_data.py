import numpy as np
import logging
from pprint import pprint, pformat

logging.basicConfig(level=logging.INFO)

# Get a tuple for our training dataset
def binomial(sample_size) -> (dict, list):
    # Generate n from Rayleigh distribution (smooth Poisson distribution) with sigma=1
    n = np.ceil(np.random.rayleigh(scale=100.0))
    logging.debug(f'n\t\t= {n}')

    # Generate p uniformly from [0.0, 1.0)
    p = np.random.uniform(low=0.0, high=1.0)
    logging.debug(f'p\t\t= {p}')

    # Sample from the binomial distribution specified by n and p
    sample_data = np.random.binomial(n, p, sample_size)
    logging.debug(f'sample_data\t= {sample_data}')

    # Include mean and stddev so that we have a unified way of comparing dists
    # https://en.wikipedia.org/wiki/Binomial_distribution
    specs = {"distribution_type": "binomial",
             "n": n,
             "p": p,
             "mean": n * p,
             "stddev": np.sqrt(n * p * (1 - p))}
    logging.debug(f'specs: \n{pformat(specs)}')

    return (specs, sample_data) 


def normal(sample_size) -> (dict, list):
    # np.random.random_sample() gives a value in [0.0, 1.0),
    # so multiply by -1 or 1 to get values in (-1.0, 1.0)
    sign = np.random.choice([-1, 1])
    mean = np.random.random_sample() * sign
    logging.debug(f'mean\t\t= {mean}')

    # Draw from Rayleigh distribution with sigma=1
    stddev = np.random.rayleigh(scale=1.0)
    logging.debug(f'stddev\t= {stddev}')

    # Get `sample_size` points from normal distribution with given specs
    sample_data = np.random.normal(mean, stddev, sample_size)
    logging.debug(f'sample_data\t= {sample_data}')

    specs = {"distribution_type": "normal",
             "mean": mean,
             "stddev": stddev}
    logging.debug(f'specs: \n{pformat(specs)}')

    return (specs, sample_data) 


'''
Generate a training dataset where each piece of data is an ordered pair:
    (distribution type and parameters,
    set of random sampled points from the distribution)

Parameters:
`count`: size of training set (number of trials)
`sample_size`: number of points we will show the AI network for each trial
'''
def generate_training_data(count, sample_size):

    training_data = []
    distributions = ['normal', 'binomial']

    for _ in range(count):
        for dist in distributions:
            if dist == 'normal':
                data_piece = normal(sample_size)
            elif dist == 'binomial':
                data_piece = binomial(sample_size)
            
            logging.info(f'data_piece: \n{pformat(data_piece)}\n')
            training_data.append(data_piece)

    return training_data

def main():
    training_data = generate_training_data(count=1, sample_size=5)
    print(f'Dataset generated for training:\n{pformat(training_data)}')

if __name__ == "__main__":
    main()
