import numpy as np
import logging
from pprint import pformat
from distributions import *

from env import DISTRIBUTION_TYPES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

'''
Parameters:
`count`: size of dataset (number of trials)
`sample_size`: number of points we will show the AI network for each trial
'''
def generate_data(count):
    data = []

    for _ in range(count):
        for dist in DISTRIBUTION_TYPES:
            # Avoid giant if statement, almost definitely slower
            exec("data_piece = " + dist + "_dist()")
            exec("data.append(data_piece)")
            # Legacy implementation
            # data_piece = normal_dist()
            # if dist == 'exponential':
            #     data_piece = exponential(sample_size)
            # elif dist == 'normal':
            #     data_piece = normal(sample_size)
            #logger.debug(f'data_piece: \n{pformat(data_piece)}\n')
            #data.append(data_piece)

    return data
