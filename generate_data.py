# generate_data.py

import numpy as np
import logging
from env import DISTRIBUTION_FUNCTIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_data(count):
    data = []
    for _ in range(count):
        for dist_name, dist_func in DISTRIBUTION_FUNCTIONS.items():
            # Call the function directly using the dictionary value
            data_piece, labels = dist_func()
            data.append((data_piece, labels))
    return data
