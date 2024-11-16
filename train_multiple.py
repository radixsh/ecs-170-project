import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import torch
import os
import importlib
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from env import NUM_DIMENSIONS, DEVICE, CONFIG
from build_model import build_model
from generate_data import generate_data
from pipeline import pipeline, MyDataset
from distributions import DISTRIBUTION_FUNCTIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Goal: Train multiple models, and save each one
def main():
    start = time.time()

    training_sizes = [100]#, 100, 500, 1000, 2000, 4000, 8000, 10000]

    # Make sure the models directory exists
    models_directory = 'models'
    os.makedirs(models_directory, exist_ok=True)

    # Filenames for various models
    dests = []
    for i in training_sizes:
        dests.append(f'{models_directory}/weights_training_size_{i}.pth')

    # Train one model per training_size
    for i, training_size in enumerate(training_sizes):
        # Update TRAINING_SIZE in the dict from env.py
        CONFIG['TRAINING_SIZE'] = training_size

        # Initialize a new neural net
        input_size = CONFIG['SAMPLE_SIZE'] * NUM_DIMENSIONS
        output_size = (len(DISTRIBUTION_FUNCTIONS) + 2) * NUM_DIMENSIONS
        model = build_model(input_size, output_size).to(DEVICE)

        # Train the model anew, and save the resulting model's weights out
        logger.debug(f"Training model with TRAINING_SIZE={training_size}...")
        train_start = time.time()

        model_weights = pipeline(model, CONFIG)
        if model_weights is None:
            logger.info("pipeline() did not return anything!!! mehhhhhh!!!!!")
        torch.save(model_weights, dests[i])

        train_end = time.time()
        logger.debug(f"Wrote out weights to {dests[i]} "
                     f"(finished in {train_end - train_start:.2f} seconds)")

    end = time.time()
    logger.info(f"Finished training {dests} models in {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
