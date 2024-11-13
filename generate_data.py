import logging
from distributions import DISTRIBUTION_FUNCTIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_data(count, sample_size):
    data = []
    for _ in range(count):
        for dist_name, dist_func in DISTRIBUTION_FUNCTIONS.items():
            points, labels = dist_func(sample_size)
            data.append((points, labels))
    return data
