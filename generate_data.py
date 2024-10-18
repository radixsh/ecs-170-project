import numpy as np
import logging
from pprint import pprint, pformat

logging.basicConfig(level=logging.DEBUG)

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

    for _ in range(count):

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

        data_piece = (specs, sample_data)
        logging.debug(f'data_piece: \n{pformat(data_piece)}\n')

        training_data.append(data_piece)

    return training_data

def main():
    training_data = generate_training_data(count=2, sample_size=5)
    print(f'Dataset generated for training:\n{pformat(training_data)}')

if __name__ == "__main__":
    main()
