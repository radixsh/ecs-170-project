import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import torch
import os
import re
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from env import *
from custom_loss_function import CustomLoss
from build_model import build_model
from generate_data import generate_data
from model_pipeline import pipeline, MyDataset
from distributions import DISTRIBUTION_FUNCTIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def update_sample_size(sample_size):
    # https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file
    with open('env.py', 'r') as f:
        lines = f.readlines()

    replacement = f'SAMPLE_SIZE = {sample_size}\n'
    updated = [replacement if line.startswith('SAMPLE_SIZE') else line
               for line in lines]

    with open('env.py', 'w') as f:
        f.writelines(updated)

def get_mae(model, desired) -> float:
    if desired == "mean":
        index = -2
    elif desired == "stddev":
        index = -1

    # Test the model on the test data
    raw_test_data = generate_data(count=TEST_SIZE)
    test_samples = np.array([elem[0] for elem in raw_test_data])
    test_labels = np.array([elem[1] for elem in raw_test_data])
    test_dataset = MyDataset(test_samples, test_labels)
    test_dataloader = DataLoader(test_dataset)

    actuals = []
    guesses = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(DEVICE).float(), y.to(DEVICE).float()

            # For some reason, y looks like this:
            # tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            #     0.5348, 0.0965]])
            # So I have to do y = y[0] to access the list, and then get the
            # mean with [-2] (or stddev with [-1]) after that
            actual_value = y[0][index]
            actuals.append(actual_value)

            # Same with predictions
            pred = model(X)
            predicted_value = pred[0][index]
            guesses.append(predicted_value)

    return mean_absolute_error(actuals, guesses)

def get_r2(model, desired) -> float:
    if desired == "mean":
        index = -2
    elif desired == "stddev":
        index = -1

    # Test the model on the test data
    raw_test_data = generate_data(count=TEST_SIZE)
    test_samples = np.array([elem[0] for elem in raw_test_data])
    test_labels = np.array([elem[1] for elem in raw_test_data])
    test_dataset = MyDataset(test_samples, test_labels)
    test_dataloader = DataLoader(test_dataset)

    actuals = []
    guesses = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(DEVICE).float(), y.to(DEVICE).float()

            # For some reason, y looks like this:
            # tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            #     0.5348, 0.0965]])
            # So I have to do y = y[0] to access the list, and then get the
            # mean with [-2] (or stddev with [-1]) after that
            actual_value = y[0][index]
            actuals.append(actual_value)

            # Same with predictions
            pred = model(X)
            predicted_value = pred[0][index]
            guesses.append(predicted_value)

    return r2_score(actuals, guesses)

def plot_mae(sample_sizes, mean_maes, stddev_maes):
    plt.scatter(sample_sizes, mean_maes, color="blue", label="Means")
    plt.scatter(sample_sizes, stddev_maes, color="orange", label="Stddevs")

    plt.xlim(0, max(sample_sizes) * 1.1)
    plt.xlabel('Sample sizes')

    both = mean_maes + stddev_maes + [0]
    plt.ylim(min(both), max(both) * 1.1)
    plt.ylim(0, max(max(mean_maes), max(stddev_maes)) * 1.1)
    plt.ylabel('MAE (mean average error)')

    plt.title("MAE")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig("results/mae.png", bbox_inches="tight")
    plt.show()

def plot_r2(sample_sizes, mean_r2s, stddev_r2s):
    plt.scatter(sample_sizes, mean_r2s, color="blue",
                label="Means")
    plt.scatter(sample_sizes, stddev_r2s, color="orange",
                label="Stddevs")

    plt.xlim(0, max(sample_sizes) * 1.1)
    plt.xlabel('Sample sizes')

    both = mean_r2s + stddev_r2s + [0]
    plt.ylim(min(both) * 1.1, max(both) * 1.1)
    plt.ylabel('R2 (correlation coefficient)')

    plt.title("R2")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig("results/r2.png", bbox_inches="tight")
    plt.show()

# Goal: Graph how loss, R^2, and MAE improves with greater sample sizes
def main():
    logger.info("Measuring MAE and R2 for means and stddevs")
    start = time.time()

    # Make sure the models directory exists
    models_directory = 'models'
    os.makedirs(models_directory, exist_ok=True)

    model_files = []
    for dirpath, _, files in os.walk(models_directory):
        model_files += [os.path.join(dirpath, f) for f in files] 
    logger.debug(model_files)

    mean_maes = []
    stddev_maes = []
    mean_r2s = []
    stddev_r2s = []
    for model_filename in model_files:
        # Get sample_size from filename (super hacky but it works)
        try:
            match = re.search(r'_(\d+)\.pth$', model_filename)
            sample_size = match.group(1)
        except:
            continue 
        
        # Use that sample_size value to initialize a new neural net of the
        # proper dimensions
        input_size = int(sample_size * NUM_DIMENSIONS)
        output_size = (len(DISTRIBUTION_FUNCTIONS) + 2) * NUM_DIMENSIONS
        model = build_model(input_size, output_size).to(DEVICE)

        # Populate model with the weights from the filename
        model.load_state_dict(torch.load(model_filename))

        # Get MAE for this sample size
        mean_mae = get_mae(model, "mean")
        mean_maes.append(mean_mae)
        stddev_mae = get_mae(model, "stddev")
        stddev_maes.append(stddev_mae)

        # Get R2 correlation coefficient
        mean_r2 = get_r2(model, "mean")
        mean_r2s.append(mean_r2)
        stddev_r2 = get_r2(model, "stddev")
        stddev_r2s.append(stddev_r2)

        train_end = time.time()
        logger.info(f"SAMPLE_SIZE={sample_size}\t--> "
                    f"mean_mae={mean_mae},\t\tstddev_MAE={stddev_mae} ")
        logger.info(f"\t\t--> mean_r2={mean_r2},\tstddev_r2={stddev_r2} "
                    f"(Finished in {train_end - train_start:.2f} seconds)")

    end = time.time()
    logger.info(f"Finished collecting data in {end - start:.2f} seconds")

    # fig = plt.figure()
    plot_mae(sample_sizes, mean_maes, stddev_maes)

    # fig = plt.figure()
    plot_r2(sample_sizes, mean_r2s, stddev_r2s)

if __name__ == "__main__":
    main()
