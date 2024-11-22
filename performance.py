import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import torch
import os
import re
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, f1_score
from torch.utils.data import DataLoader

from env import CONFIG, HYPERPARAMETER, NUM_DIMENSIONS, DEVICE
from build_model import build_model
from pipeline import get_dataloader
from distributions import DISTRIBUTION_FUNCTIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def get_mae_mape_r2(model, desired):
    if desired == "mean":
        index = -2
    elif desired == "stddev":
        index = -1
    else:
        logger.warning(f"Passed unreadable string to get_mae_mape_r2(), "
                       f"do you want mean or stddev?")
        return

    test_dataloader = get_dataloader(CONFIG, 'data/test_dataset',
                                     required_size=CONFIG['TEST_SIZE'])
    model.eval()
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

    return (mean_absolute_error(actuals, guesses),
            mean_absolute_percentage_error(actuals, guesses),
            r2_score(actuals, guesses))

# My sh*tty test (W.I.P) for measuring clssification.
def get_classification_metrics(model, desired):
    

    return None
    


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
    logger.info("Measuring MAE, MAPE, and R^2 for means and stddevs")
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
        filenameified = HYPERPARAMETER.lower().replace(' ', '_')
        if os.path.isfile(fullpath) and filenameified in fullpath:
            model_filenames.append(fullpath)
    print(f'Analyzing {model_filenames}')

    mean_maes = []
    stddev_maes = []
    mean_mapes = []
    stddev_mapes = []
    mean_r2s = []
    stddev_r2s = []
    hyperparams = []       # For matplotlib graphs
    count = 0
    for filename in model_filenames:
        model_start = time.time()

        # Update CONFIG[HYPERPARAMETER] with the value from filename
        match = re.search(r'(\d+).pth$', filename)
        if match:
            hyperparam = int(match.group(1))
            hyperparams.append(hyperparam)     # For matplotlib graphs
            CONFIG[HYPERPARAMETER] = hyperparam
            count += 1
        else:
            logger.info(f'(No {HYPERPARAMETER} detected in "{filename}", skipping)')
            continue

        # For each new sample size, re-initialize
        input_size = CONFIG['SAMPLE_SIZE'] * NUM_DIMENSIONS
        output_size = (len(DISTRIBUTION_FUNCTIONS) + 2) * NUM_DIMENSIONS
        model = build_model(input_size, output_size).to(DEVICE)

        # Load the model's weights
        state_dict = torch.load(filename)
        if state_dict is None:
            logger.debug("State dict is illegible, skipping")
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
        
        # Testing function calls to measure classification performance given F1 score, recall, precision, and accuracy.
        # f1_score = 
        # precision =
        # recall = 
        # accuracy =


    

        model_end = time.time()
        logger.info(f"{HYPERPARAMETER}={hyperparam}\t--> "
                    f"mean_mae={mean_mae:.2f},\tstddev_mae={stddev_mae:.2f} "
                    f"\n\t\t\t--> mean_mape={mean_mape:.2f},\tstddev_mape={stddev_mape:.2f}"
                    f"\n\t\t\t--> mean_r2={mean_r2:.2f},\tstddev_r2={stddev_r2:.2f} "
                    f"(Finished in {model_end - model_start:.2f} seconds)")

    end = time.time()
    logger.info(f"Analyzed {count} models in {end - start:.2f} seconds")
    if count == 0:
        return

    create_png("MAE (mean average error)",
               hyperparams, mean_maes, stddev_maes)
    create_png("MAPE (mean average percentage error)",
               hyperparams, mean_mapes, stddev_mapes)
    create_png("R^2 (correlation coefficient)",
               hyperparams, mean_r2s, stddev_r2s)

if __name__ == "__main__":
    main()
