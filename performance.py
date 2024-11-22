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

from env import CONFIG, HYPERPARAMETER, NUM_DIMENSIONS, DEVICE, VALUES
from build_model import build_model
from train_multiple import get_dataloader
from custom_functions import DISTRIBUTIONS, make_weights_filename, get_indices

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

distribution_indices = list(DISTRIBUTIONS.keys())

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

            # Curr_ refers to current data point
            curr_actual_means = []
            curr_actual_stddevs = []
            curr_predicted_means = []
            curr_predicted_stddevs = []

            # Go through each dimension
            for dim in range(1, NUM_DIMENSIONS+1):
                dim_mean_idxs = get_indices(mean=True, dims = [dim])
                dim_stddev_idxs = get_indices(stddev=True, dims = [dim])

                # Get the specified values; dim_ refers to current dimensions
                dim_actual_means = y[0][dim_mean_idxs]
                dim_actual_stddevs = y[0][dim_stddev_idxs]
                dim_predicted_means = pred[0][dim_mean_idxs]
                dim_predicted_stddevs = pred[0][dim_stddev_idxs]

                curr_actual_means.append(dim_actual_means)
                curr_actual_stddevs.append(dim_actual_stddevs)
                curr_predicted_means.append(dim_predicted_means)
                curr_predicted_stddevs.append(dim_predicted_stddevs)

            # After looping through dims, our curr_actual_stuff looks like
            # [dim1_stuff, dim2_stuff, ...]
            # Store them for further processing
            actual_means.append(curr_actual_means)
            actual_stddevs.append(curr_actual_stddevs)
            predicted_means.append(curr_predicted_means)
            predicted_stddevs.append(curr_predicted_stddevs)

    # actual_stuff now looks like:
    # [[x1_dim1_stuff, x1_dim2_stuff, ...],
    #  [x2_dim2_stuff, x2_dim2_stuff, ...],
    #  ...]
    # We want something for the form:
    # [[x1_dim1_stuff, x2_dim1_stuff, ...],
    #  [x1_dim2_stuff, x2_dim2_stuff, ...],
    #  ...]
    # ChatGPT is cracked at pytorch
    actual_means = torch.tensor(actual_means)
    actual_stddevs = torch.tensor(actual_stddevs)
    predicted_means = torch.tensor(predicted_means)
    predicted_stddevs = torch.tensor(predicted_stddevs)
    actual_means = actual_means.transpose(0, 1)
    actual_stddevs = actual_stddevs.transpose(0, 1)
    predicted_means = predicted_means.transpose(0, 1)
    predicted_stddevs = predicted_stddevs.transpose(0, 1)

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

def get_classification_metrics(model):
    index = get_indices(dists=True)
    test_dataloader = get_dataloader(CONFIG, mode='TEST')
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

            #print(f'Actual: {actual_dist}, Predicted: {predicted_dist}')

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

    mean_maes = []
    stddev_maes = []
    mean_mapes = []
    stddev_mapes = []
    mean_r2_scores = []
    stddev_r2_scores = []
    hyperparams = []       # For matplotlib graphs
    accuracies = []
    f1s = []
    recalls = []
    precisions = []
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

        accuracy, f1, recall, precision = get_classification_metrics(model)
        accuracies.append(accuracy)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)

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

    regression_png("MAE (mean average error)",
               hyperparams, mean_maes, stddev_maes)
    regression_png("MAPE (mean average percentage error)",
               hyperparams, mean_mapes, stddev_mapes)
    regression_png("R^2 (correlation coefficient)",
               hyperparams, mean_r2_scores, stddev_r2_scores)
    classification_png("Accuracy", hyperparams, accuracies)
    classification_png("F1", hyperparams, f1s)
    classification_png("Recall", hyperparams, recalls)
    classification_png("Precision", hyperparams, precisions)

if __name__ == "__main__":
    main()
