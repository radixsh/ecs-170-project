import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
from custom_functions import *
from env import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

### Other stuff

def generate_data(count, sample_size):
    data = []
    # Generate 'count' examples uniformly at random (before, it was 9 * 'count')
    for _ in range(count):
        dist_class = np.random.choice(DISTRIBUTIONS.items())
        dist_object = dist_class()
        points = dist_object.rng(sample_size)
        label = dist_object.get_label()
        data.append((points, label))
        # items = list(DISTRIBUTIONS.items())
        # choice = np.random.choice(len(items))
        # _, dist_func = items[choice]
        # points, labels = dist_func(sample_size)
        # data.append((points, labels))
    return data

'''
# TODO: multidimensional generation
it should randomly generate N distribution types, where N is NUM_DIMENSIONS
the function should call the randomly chosen distribution functions from distributions.py, and make an array of N-tuples out of the result
it should also make an array of labels, which is the concatenation of the 1hot, mean and stddev of each distribution
return a pair of those arrays

e.g., if NUM_DIMENSIONS is 2 and SAMPLE_SIZE is 3, we randomly choose 3 dimensions (say normal and gamma)
the function should return something that looks like:
[
   [n1, n2, n3, g1, g2, g3],
   [0,0,0,0,0,0,1,0,0,normal_mean,normal_stddev,0,1,0,0,0,0,0,0,0,gamma_mean,gamma_stddev]
]
where n1,g1 are the first points generated by the normal dist and gamma dist, etc

count is the number of examples to generate
sample_size is the number of points to generate for each distribution
'''

def generate_multidim_data(dimensions, count, sample_size):
    data = []

    for _ in range(count):
        points = []
        labels = []
        items = list(DISTRIBUTIONS.items())
        for _ in range(dimensions):
            dist_name = np.random.choice(list(DISTRIBUTIONS.keys()))
            logger.debug(f'dist_name: {dist_name}')

            # dist_class = np.random.choice(DISTRIBUTIONS.items())
            dist_class = DISTRIBUTIONS[dist_name]
            # Instantiate the class!!! 
            dist_object = dist_class()

            dist_points = dist_object.rng(sample_size)
            logger.debug(f'dist_points from this dist: {dist_points}')
            dist_points = list(dist_points)
            logger.debug(f'dist_points but cast to list: {dist_points}')
            points.append(dist_points)

            labels += dist_object.get_label()
            logger.debug(f'labels from this dist: {labels}')
            # don't use .extend please, obfuscates what's going on
            #labels.extend(labels)

        # List of each dimension's points: [[1,3,5], [2,4,6]]
        logger.debug(f"Each dimension's points: {points}")

        # Flattened: [1,3,5, 2,4,6]
        # The actual points in the 2D distribution will be (1,2), (3,4), (5,6)
        #points = [item for dim in points for item in dim]
        points = np.ravel(points,order='C')
        logger.debug(f"Flattened: {points}")

        data.append((points, labels))

    return data

def get_good_filename(config,mode):
    for filename in os.listdir('data'):

        # Catches weird stuff (like .DS_store)
        if 'dataset' not in filename:
            continue

        # Check the type, that there's enough data, 
        # the sample size is right, and that num_dimensions matches
        file_info = parse_data_filename(filename)
        if file_info['TYPE'] == mode \
            and file_info['SIZE'] >= config[f'{mode}_SIZE'] \
            and file_info['SAMPLE_SIZE'] == config['SAMPLE_SIZE'] \
            and file_info['NUM_DIMS'] == NUM_DIMENSIONS:
                good_filename = os.path.join("data", filename)
                return good_filename
    return None

def make_dataset(config, mode):

    # Check for a preexisting good file
    good_filename = get_good_filename(config,mode)
    
    if good_filename:
        logger.info(f"Sufficient data already exists at {good_filename}!")
        dataset = torch.load(good_filename)
    
    else:
        start = time.time()

        # Get the correct size
        examples_count = config[f'{mode}_SIZE']

        filename = make_data_filename(mode,
                                examples_count,
                                config['SAMPLE_SIZE'],
                                NUM_DIMENSIONS)

        raw_data = generate_multidim_data(dimensions=NUM_DIMENSIONS,
                                        count=examples_count,
                                        sample_size=config['SAMPLE_SIZE'])

        samples = np.array([elem[0] for elem in raw_data])
        labels = np.array([elem[1] for elem in raw_data])
        dataset = MyDataset(samples, labels)
        
        torch.save(dataset, filename)

        end = time.time()
        logger.info(f"Generated and saved {examples_count} examples out "
                    f"to {filename} in {end - start:.2f} seconds "
                    f"(BATCH_SIZE={config['BATCH_SIZE']})")

    return dataset

def get_dataloader(config, mode='TRAIN'): #mode should be 'TRAIN' or 'TEST'
   
    good_filename = get_good_filename(config,mode)
    
    if good_filename: 
        logger.info(f"Loading data from {good_filename}...")
        dataset = torch.load(good_filename)
    else: # If no valid data is found, then generate some new data
        logger.info(f'No valid data found, generating fresh data...')
        dataset = make_dataset(config, mode)
    
    if mode == 'TRAIN':
        dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'])
    elif mode == 'TEST': # batch size should just be 1 for testing.
        dataloader = DataLoader(dataset, batch_size=1)

    return dataloader

# If running this file directly as a script, then generate some training
# examples and save them to a file for later use
if __name__ == "__main__":
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)
    make_dataset(CONFIG,mode='TRAIN')
    make_dataset(CONFIG,mode='TEST')
