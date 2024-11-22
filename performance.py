import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import torch
import os
import re
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, accuracy_score, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader

from env import CONFIG, HYPERPARAMETER, NUM_DIMENSIONS, DEVICE
from build_model import build_model
from train_multiple import get_dataloader
from generate_data import DISTRIBUTION_FUNCTIONS
from custom_functions import get_indices

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

distribution_indices = list(DISTRIBUTION_FUNCTIONS.keys())

colors = ["royalblue", "tomato", "forestgreen", "darkorange",
        "purple", "cyan", "magenta", "yellow", "black"]

# Regression: Graph how loss, R^2, and MAE varies with different sample sizes,
# training sizes, epoch counts, and/or learning rate

# Classification: Graph how accuracy, precision, recall, and F1 score
# varies with different sample sizes, training sizes, epoch counts, and/or
# learning rate
# - there's overall accuracy, and then there's accuracy per class
# - f1, precision, recall are only per class

# regression performance
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
                                     examples_count=CONFIG['TEST_SIZE'])
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

def get_classification_metrics(model):
    index = get_indices(dists=True)
    test_dataloader = get_dataloader(CONFIG, 'data/test_dataset',
                                     examples_count=CONFIG['TEST_SIZE'])
    model.eval()
    actuals = []
    guesses = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(DEVICE).float(), y.to(DEVICE).float()
            
            # Extract actual distribution
            actual_dist = y[0][index]
            actuals.append(actual_dist)

            # Get predicted distribution
            pred = model(X)
            predicted_dist = pred[0][index]
            guesses.append(predicted_dist)

            print(f'Actual: {actual_dist}, Predicted: {predicted_dist}')

    # Convert lists of tensors to a single tensor
    actuals = torch.stack(actuals)
    guesses = torch.stack(guesses)

    # Convert distributions to class labels
    actual_labels = torch.argmax(actuals, dim=-1)
    predictions = torch.argmax(guesses, dim=-1)

    # Ensure compatibility with sklearn (convert to NumPy)
    return (accuracy_score(actual_labels.cpu().numpy(), predictions.cpu().numpy()),
        f1_score(actual_labels.cpu().numpy(), predictions.cpu().numpy(),
            average=None),
        recall_score(actual_labels.cpu().numpy(),
            predictions.cpu().numpy(), average=None),
        precision_score(actual_labels.cpu().numpy(),
            predictions.cpu().numpy(), average=None))

# regression pngs
def regression_png(name, x_values, means, stddevs):
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

def classification_png(name, x_values, metrics):
    plt.ylim(0, 1)
    plt.ylabel(name)

    x_values = np.float64(x_values)
    sorted_indices = np.argsort(x_values)
    x_values = x_values[sorted_indices]


    # Handle overall accuracy differently
    if name == "Accuracy":
        metrics = np.array(metrics)[sorted_indices]  # Sort 1D metrics array
        plt.scatter(x_values, metrics, color="royalblue", label="Accuracy")
        slope, intercept = np.polyfit(x_values, metrics, deg=1)
        trend = slope * x_values + intercept
        plt.plot(x_values, trend, color="royalblue", label=f"Slope: {slope:.2f}")
    else:
        # Sort 2D metrics array for other metrics (e.g., F1, Recall, Precision)
        metrics = np.array(metrics)[sorted_indices, :]
        for i in range(metrics.shape[1]):  # Loop over distributions (columns)
            plt.scatter(x_values, metrics[:, i], color=colors[i], label=distribution_indices[i])
            slope, intercept = np.polyfit(x_values, metrics[:, i], deg=1)
            trend = slope * x_values + intercept
            plt.plot(x_values, trend, color=colors[i])

    plt.gca().set_xscale('log')
    plt.xlabel(HYPERPARAMETER)

    plt.title(name)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    destination = f"results/{HYPERPARAMETER.lower()}_{name.replace(' ', '_')}.png"
    plt.savefig(destination, bbox_inches="tight")
    plt.show()

def main():
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
    accuracies = []
    f1s = []
    recalls = []
    precisions = []
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

        accuracy, f1, recall, precision = get_classification_metrics(model)
        accuracies.append(accuracy)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)

        model_end = time.time()
        logger.info(f"{HYPERPARAMETER}={hyperparam}\t--> "
                    f"mean_mae={mean_mae:.2f},\tstddev_mae={stddev_mae:.2f} "
                    f"\n\t\t\t--> mean_mape={mean_mape:.2f},\tstddev_mape={stddev_mape:.2f}"
                    f"\n\t\t\t--> mean_r2={mean_r2:.2f},\tstddev_r2={stddev_r2:.2f} "
                    f"\n\t\t\t--> accuracy={accuracy:.2f},\tf1={f1},\trecall={recall},\tprecision={precision}"
                    f"(Finished in {model_end - model_start:.2f} seconds)")

    end = time.time()
    logger.info(f"Analyzed {count} models in {end - start:.2f} seconds")
    if count == 0:
        return

    regression_png("MAE (mean average error)",
               hyperparams, mean_maes, stddev_maes)
    regression_png("MAPE (mean average percentage error)",
               hyperparams, mean_mapes, stddev_mapes)
    regression_png("R^2 (correlation coefficient)",
               hyperparams, mean_r2s, stddev_r2s)
    classification_png("Accuracy", hyperparams, accuracies)
    classification_png("F1", hyperparams, f1s)
    classification_png("Recall", hyperparams, recalls)
    classification_png("Precision", hyperparams, precisions)

if __name__ == "__main__":
    main()
