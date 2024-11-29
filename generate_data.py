import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import time
from custom_functions import *
from env import *

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# console_handler = logging.StreamHandler()
# logger.addHandler(console_handler)

# def generate_multidim_data(config, mode):
#     data = []

#     for _ in range(config[f'{mode}_SIZE']):
#         points = []
#         labels = []
#         items = list(DISTRIBUTIONS.items())
#         for _ in range(config['NUM_DIMENSIONS']):
#             dist_name = np.random.choice(list(DISTRIBUTIONS.keys()))
#             logger.debug(f'dist_name: {dist_name}')

#             # dist_class = np.random.choice(DISTRIBUTIONS.items())
#             dist_class = DISTRIBUTIONS[dist_name]
#             # Instantiate the class!!!
#             dist_object = dist_class()

#             dist_points = dist_object.rng(config['SAMPLE_SIZE'])
#             logger.debug(f'dist_points from this dist: {dist_points}')
#             dist_points = list(dist_points)
#             logger.debug(f'dist_points but cast to list: {dist_points}')
#             points.append(dist_points)

#             labels += dist_object.get_label()
#             logger.debug(f'labels from this dist: {labels}')

#         # List of each dimension's points: [[1,3,5], [2,4,6]]
#         logger.debug(f"Each dimension's points: {points}")

#         # Flattened: [1,3,5, 2,4,6]
#         # The actual points in the 2D distribution will be (1,2), (3,4), (5,6)
#         #points = [item for dim in points for item in dim]
#         points = np.ravel(points,order='C')
#         logger.debug(f"Flattened: {points}")

#         data.append((points, labels))

#     return data

# def get_good_filename(config, mode):
#     for filename in os.listdir('data'):

#         # Catches weird stuff (like .DS_store)
#         if 'dataset' not in filename:
#             continue

#         # Check the type, that there's enough data,
#         # the sample size is right, and that num_dimensions matches
#         file_info = parse_data_filename(filename)
#         if file_info['MODE'] == mode \
#             and file_info[f'{mode}_SIZE'] >= config[f'{mode}_SIZE'] \
#             and file_info['SAMPLE_SIZE'] == config['SAMPLE_SIZE'] \
#             and file_info['NUM_DIMENSIONS'] == config['NUM_DIMENSIONS']:
#                 good_filename = os.path.join("data", filename)
#                 return good_filename
#     return None

# def make_dataset(config, mode):
#     # Check for a preexisting good file
#     good_filename = get_good_filename(config,mode)

#     if good_filename:
#         logger.info(f"Sufficient data already exists at {good_filename}!")
#         dataset = torch.load(good_filename)
#         return dataset

#     start = time.time()

#     # Get the correct size
#     examples_count = config[f'{mode}_SIZE']

#     filename = make_data_filename(mode, config)

#     logger.info(f"Generating {examples_count} pieces of {config['NUM_DIMENSIONS']}"
#                 f"-dimensional {mode.lower()} data, each with {config['SAMPLE_SIZE']} samples.")

#     raw_data = generate_multidim_data(config, mode)

#     samples = np.array([elem[0] for elem in raw_data])
#     labels = np.array([elem[1] for elem in raw_data])
#     dataset = MyDataset(samples, labels)

#     torch.save(dataset, filename)

#     end = time.time()
#     logger.info(f"Generated and saved {examples_count} examples out to "
#                 f"{filename} in {end - start:.3f} seconds.")
#     return dataset

# def get_dataloader(config, mode='TRAIN'): #mode should be 'TRAIN' or 'TEST'
#     good_filename = get_good_filename(config,mode)

#     if good_filename:
#         logger.info(f"Using data from {good_filename}")
#         dataset = torch.load(good_filename, weights_only=False)
#         if parse_data_filename(good_filename)[f'{mode}_SIZE'] != config[f'{mode}_SIZE']:
#             logger.debug(f"Taking the first {config[f'{mode}_SIZE']} entries")
#             dataset = Subset(dataset, indices=range(config[f'{mode}_SIZE']))

#     else:
#         logger.info(f'No valid data found, generating fresh data...')
#         dataset = make_dataset(config, mode)

#     if mode == 'TRAIN':
#         dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
#     elif mode == 'TEST': # batch size should be the entire thing for testing.
#         dataloader = DataLoader(dataset, batch_size=config['TRAIN_SIZE'], shuffle=True)

#     return dataloader

# If running this file directly as a script, then generate some training
# examples and save them to a file for later use
if __name__ == "__main__":
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)
    make_dataset(CONFIG,mode='TRAIN')
    make_dataset(CONFIG,mode='TEST')
