import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import torch
import os
import re
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
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

VARIABLE = "TRAINING_SIZE"

def get_mae_mape_r2(model, desired) -> (float, float):
    if desired == "mean":
        index = -2
    elif desired == "stddev":
        index = -1

    # Test the model on the test data
    raw_test_data = generate_data(count=SETUP['TEST_SIZE'],
                                  sample_size=SETUP['SAMPLE_SIZE'])
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

    return mean_absolute_error(actuals, guesses), mean_absolute_percentage_error(actuals, guesses), r2_score(actuals, guesses)

def plot_mae(sample_sizes, mean_maes, stddev_maes):
    plt.scatter(sample_sizes, mean_maes, color="blue", label="Means")
    plt.scatter(sample_sizes, stddev_maes, color="orange", label="Stddevs")

    plt.xlim(0, max(sample_sizes) * 1.1)
    plt.xlabel(VARIABLE)

    both = mean_maes + stddev_maes + [0]
    plt.ylim(min(both), max(both) * 1.1)
    plt.ylabel('MAE (mean average error)')

    plt.title("MAE")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig("results/mae.png", bbox_inches="tight")
    plt.show()

def plot_mape(sample_sizes, mean_mapes, stddev_mapes):
    plt.scatter(sample_sizes, mean_mapes, color="blue", label="Means")
    plt.scatter(sample_sizes, stddev_mapes, color="orange", label="Stddevs")

    plt.xlim(0, max(sample_sizes) * 1.1)
    plt.xlabel(VARIABLE)

    both = mean_mapes + stddev_mapes + [0]
    plt.ylim(min(both), max(both) * 1.1)
    plt.ylabel('MAPE (mean average percentage error)')

    plt.title("MAPE")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig("results/mape.png", bbox_inches="tight")
    plt.show()

def plot_r2(sample_sizes, mean_r2s, stddev_r2s):
    plt.scatter(sample_sizes, mean_r2s, color="blue",
                label="Means")
    plt.scatter(sample_sizes, stddev_r2s, color="orange",
                label="Stddevs")

    plt.xlim(0, max(sample_sizes) * 1.1)
    plt.xlabel(VARIABLE)

    both = mean_r2s + stddev_r2s + [0]
    plt.ylim(min(both) * 1.1, max(both) * 1.1)
    plt.ylabel('R^2 (correlation coefficient)')

    plt.title("R^2")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig("results/r2.png", bbox_inches="tight")
    plt.show()

# Goal: Graph how loss, R^2, and MAE improves with greater sample sizes
def main():
    logger.info("Measuring MAE, MAPE, and R2 for means and stddevs")
    start = time.time()

    # Make sure the 'results' directory exists (for png graphs of results)
    results_directory = 'results'
    os.makedirs(results_directory, exist_ok=True)

    # Make sure the 'models' directory exists (and has model weights in it)
    models_directory = 'models'
    os.makedirs(models_directory, exist_ok=True)

    model_filenames = []
    for item in os.listdir(models_directory):
        fullpath = os.path.join(models_directory, item)
        if os.path.isfile(fullpath):
            model_filenames.append(fullpath)
    print(f'Analyzing {model_filenames}')

    mean_maes = []
    stddev_maes = []
    mean_mapes = []
    stddev_mapes = []
    mean_r2s = []
    stddev_r2s = []
    training_sizes = []       # For matplotlib graphs
    for filename in model_filenames:
        model_start = time.time()

        # Update SETUP[VARIABLE] with the value from filename
        match = re.search(r'(\d+).pth$', filename)
        if match:
            training_size = int(match.group(1))
            training_sizes.append(training_size)     # For matplotlib graphs
            SETUP[VARIABLE] = training_size
        else:
            logger.info(f'No VARIABLE detected in "{filename}", skipping')
            continue

        # For each new sample size, re-initialize
        input_size = SETUP['SAMPLE_SIZE'] * NUM_DIMENSIONS
        output_size = (len(DISTRIBUTION_FUNCTIONS) + 2) * NUM_DIMENSIONS
        model = build_model(input_size, output_size).to(DEVICE)

        # Load the model's weights
        state_dict = torch.load(filename)
        if state_dict is None:
            logger.debug("Looks like the state dict wasn't properly saved "
                        "out :( Skipping...")
            continue
        model.load_state_dict(state_dict)

        # Get MAE and R2 for this sample size
        mean_mae, mean_mape, mean_r2 = get_mae_mape_r2(model, "mean")
        mean_maes.append(mean_mae)
        mean_mapes.append(mean_mape)
        mean_r2s.append(mean_r2)
        stddev_mae, stddev_mape, stddev_r2 = get_mae_mape_r2(model, "stddev")
        stddev_maes.append(stddev_mae)
        stddev_mapes.append(stddev_mape)
        stddev_r2s.append(stddev_r2)

        model_end = time.time()
        logger.info(f"{VARIABLE}={training_size}\t--> "
                    f"mean_mae={mean_mae:.2f},\tstddev_mae={stddev_mae:.2f} "
                    f"\n\t\t\t--> mean_mape={mean_mape:.2f},\tstddev_mape={stddev_mape:.2f}"
                    f"\n\t\t\t--> mean_r2={mean_r2:.2f},\tstddev_r2={stddev_r2:.2f} "
                    f"(Finished in {model_end - model_start:.2f} seconds)")

    end = time.time()
    logger.info(f"Finished collecting data in {end - start:.2f} seconds")

    plot_mae(training_sizes, mean_maes, stddev_maes)
    plot_mape(training_sizes, mean_mapes, stddev_mapes)
    plot_r2(training_sizes, mean_r2s, stddev_r2s)

if __name__ == "__main__":
    main()