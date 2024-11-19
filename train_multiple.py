import logging
import time
import torch
import os

from env import NUM_DIMENSIONS, DEVICE, CONFIG, HYPERPARAMETER
from build_model import build_model
from pipeline import pipeline
from distributions import DISTRIBUTION_FUNCTIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Goal: Train multiple models, and save each one
def main():
    start = time.time()

    hyperparams = [1e1, 1e2, 1e3, 1e4, 1e5]#, 1e6]
    hyperparams = [int(i) for i in hyperparams]

    # Make sure the 'data' directory exists
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)

    # Make sure the models directory exists
    models_directory = 'models'
    os.makedirs(models_directory, exist_ok=True)

    # Filenames for various models
    dests = []
    for i in hyperparams:
        label = HYPERPARAMETER.lower().replace(' ', '_')
        dests.append(f'{models_directory}/weights_{label}_{i}.pth')

    # Train one model per training_size
    for i, hyperparam in enumerate(hyperparams):
        # Update TRAINING_SIZE in the dict from env.py
        CONFIG[HYPERPARAMETER] = hyperparam

        # Initialize a new neural net
        input_size = CONFIG['SAMPLE_SIZE'] * NUM_DIMENSIONS
        output_size = (len(DISTRIBUTION_FUNCTIONS) + 2) * NUM_DIMENSIONS
        model = build_model(input_size, output_size).to(DEVICE)

        # Train the model anew, and save the resulting model's weights out
        logger.debug(f"Training model with {HYPERPARAMETER}={hyperparam}...")
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
