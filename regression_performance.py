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

from env import CONFIG, HYPERPARAMETER, NUM_DIMENSIONS, DEVICE, VALUES
from build_model import build_model
from train_multiple import get_dataloader
from custom_functions import DISTRIBUTIONS, make_weights_filename, get_indices

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def test_model(model):

    test_dataloader = get_dataloader(CONFIG,mode='TEST')
    model.eval()

    actual_means = []
    actual_stddevs = []
    predicted_means = []
    predicted_stddevs = []

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(DEVICE).float(), y.to(DEVICE).float()

            # Call the model
            pred = model(X)

            # Indices for means and stddevs
            mean_idxs = get_indices(mean=True, dims = range(1,NUM_DIMENSIONS+1))
            stddev_idxs = get_indices(stddev=True, dims = range(1,NUM_DIMENSIONS+1))

            # Grab appropriate values
            curr_actual_means = y[0][mean_idxs]
            curr_actual_stddevs = y[0][stddev_idxs]
            curr_predicted_means = pred[0][mean_idxs]
            curr_predicted_stddevs = pred[0][stddev_idxs]

            # Need to transpose them for processing!
            curr_actual_means = curr_actual_means.unsqueeze(1)
            curr_actual_stddevs = curr_actual_stddevs.unsqueeze(1)
            curr_predicted_means = curr_predicted_means.unsqueeze(1)
            curr_predicted_stddevs = curr_predicted_stddevs.unsqueeze(1)

            # Store them for processing
            actual_means.append(curr_actual_means)
            actual_stddevs.append(curr_actual_stddevs)
            predicted_means.append(curr_predicted_means)
            predicted_stddevs.append(curr_predicted_stddevs)

    mean_maes = []
    mean_mapes = []
    mean_r2_scores = []
    stddev_maes = []
    stddev_mapes = []
    stddev_r2_scores = []

    # If there's a better way to do this, go ahead
    # But note that averaging across dimensions first then taking the MAE
    # is NOT the same as MAE then averaging across dimensions (what this nonsense is doing)
    # Also don't ask me why all the other ranges are offset and this one isn't
    for dim in range(NUM_DIMENSIONS):

        curr_mean_mae = mean_absolute_error(actual_means[dim], predicted_means[dim])
        curr_mean_mape = mean_absolute_percentage_error(actual_means[dim], predicted_means[dim])
        curr_mean_r2_score = r2_score(actual_means[dim], predicted_means[dim])
        curr_stddev_mae = mean_absolute_error(actual_stddevs[dim], predicted_stddevs[dim])
        curr_stddev_mape = mean_absolute_percentage_error(actual_stddevs[dim], predicted_stddevs[dim])
        curr_stddev_r2_score = r2_score(actual_stddevs[dim], predicted_stddevs[dim])

        mean_maes.append(curr_mean_mae)
        mean_mapes.append(curr_mean_mape)
        mean_r2_scores.append(curr_mean_r2_score)
        stddev_maes.append(curr_stddev_mae)
        stddev_mapes.append(curr_stddev_mape)
        stddev_r2_scores.append(curr_stddev_r2_score)

    # Average across all dimensions
    # Top 10 worst variables names
    mean_mean_mae = np.mean(mean_maes)
    mean_mean_mape = np.mean(mean_mapes)
    mean_mean_r2_score = np.mean(mean_r2_scores)
    mean_stddev_mae = np.mean(stddev_maes)
    mean_stddev_mape = np.mean(stddev_mapes)
    mean_stddev_r2_score = np.mean(stddev_r2_scores)

    out = {
        "mean_mae": mean_mean_mae,
        "mean_mape": mean_mean_mape,
        "mean_r2_score": mean_mean_r2_score,
        "stddev_mae": mean_stddev_mae,
        "stddev_mape": mean_stddev_mape,
        "stddev_r2_score": mean_stddev_r2_score,
    }

    return out

def create_png(name, x_values, means, stddevs):
    both = means + stddevs + [0]
    plt.ylim(min(both) * 1.1, max(both) * 1.1)
    plt.ylabel(name)

    x_values = np.float64(x_values)
    sorted_indices = np.argsort(x_values)
    x_values = x_values[sorted_indices]
    means = np.array(means)[sorted_indices]
    stddevs = np.array(stddevs)[sorted_indices]

    plt.scatter(x_values, means, color="royalblue", label="Means")
    mean_slope, mean_intercept = np.polyfit(x_values, means, deg=1)
    # Line of best fit only looks bent because of logarithmic scaling
    mean_trend = np.polyval([mean_slope, mean_intercept], x_values)
    plt.plot(x_values, mean_trend, color="royalblue",
             label=f'Mean slope: {mean_slope}')

    plt.scatter(x_values, stddevs, color="tomato", label="Stddevs")
    stddev_slope, stddev_intercept = np.polyfit(x_values, stddevs, deg=1)
    stddev_trend = stddev_slope * x_values + stddev_intercept
    plt.plot(x_values, stddev_trend, color="tomato",
             label=f'Stddev slope: {stddev_slope}')

    plt.gca().set_xscale('log')
    plt.xlabel(HYPERPARAMETER)

    plt.title(name)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    destination = f"results/{HYPERPARAMETER.lower()}_{name.replace(' ', '_')}.png"
    plt.savefig(destination, bbox_inches="tight")
    plt.show()

# Goal: Graph how loss, R^2, and MAE varies with different sample sizes,
# training sizes, epoch counts, and/or learning rate
def main():
    start = time.time()

    # Make sure the 'results' directory exists (for png graphs of results)
    results_directory = 'results'
    os.makedirs(results_directory, exist_ok=True)

    # Make sure the 'models' directory exists (and has model weights in it)
    models_directory = 'models'
    os.makedirs(models_directory, exist_ok=True)

    mean_maes = []
    stddev_maes = []
    mean_mapes = []
    stddev_mapes = []
    mean_r2_scores = []
    stddev_r2_scores = []
    hyperparams = []       # For matplotlib graphs
    count = 0

    for hyperparam_val in VALUES:
        
        # Update CONFIG with the hyperparameter
        CONFIG[HYPERPARAMETER] = hyperparam_val
        
        # Filename to search for
        model_filename = make_weights_filename(CONFIG['TRAIN_SIZE'],
                                              CONFIG['SAMPLE_SIZE'],
                                              NUM_DIMENSIONS)
        
        logger.info(f'Analyzing model at {model_filename}...')

        # Try to load it in
        state_dict = torch.load(model_filename)
        if state_dict is None:
            # Could add more info to this debug.
            logger.debug("Model not found, skipping!")
            continue

        model_start = time.time()              # For timing
        hyperparams.append(hyperparam_val)     # For matplotlib graphs
        count += 1                             # For debug
        
        # Reinitialize
        input_size = CONFIG['SAMPLE_SIZE'] * NUM_DIMENSIONS
        output_size = (len(DISTRIBUTIONS) + 2) * NUM_DIMENSIONS
        model = build_model(input_size, output_size).to(DEVICE)

        model.load_state_dict(state_dict)

        # Get MAE and R2 for this sample size
        test_results = test_model(model)

        mean_mae = test_results['mean_mae']
        mean_mape = test_results['mean_mape']
        mean_r2_score = test_results['mean_r2_score']
        stddev_mae = test_results['stddev_mae']
        stddev_mape = test_results['stddev_mape']
        stddev_r2_score = test_results['stddev_r2_score']

        mean_maes.append(mean_mae)
        mean_mapes.append(mean_mape)
        mean_r2_scores.append(mean_r2_score)
        stddev_maes.append(stddev_mae)
        stddev_mapes.append(stddev_mape)
        stddev_r2_scores.append(stddev_r2_score)

        model_end = time.time()
        logger.info(f"{HYPERPARAMETER}={hyperparam_val}"
                    f"\n\t\t--> mean_mae={mean_mae:.2f},\tstddev_mae={stddev_mae:.2f} "
                    f"\n\t\t--> mean_mape={mean_mape:.2f},\tstddev_mape={stddev_mape:.2f}"
                    f"\n\t\t--> mean_r2={mean_r2_score:.2f},\tstddev_r2={stddev_r2_score:.2f} "
                    f"\n(Finished in {model_end - model_start:.2f} seconds)")

    end = time.time()
    logger.info(f"Analyzed {count} models in {end - start:.2f} seconds")
    if count == 0:
        return

    create_png("MAE (mean average error)",
               hyperparams, mean_maes, stddev_maes)
    create_png("MAPE (mean average percentage error)",
               hyperparams, mean_mapes, stddev_mapes)
    create_png("R^2 (correlation coefficient)",
               hyperparams, mean_r2_scores, stddev_r2_scores)

if __name__ == "__main__":
    main()
