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
import warnings

from env import CONFIG, HYPERPARAMETER, NUM_DIMENSIONS, DEVICE, VALUES
from build_model import build_model
from train_multiple import get_dataloader
from custom_functions import DISTRIBUTIONS, make_weights_filename, get_indices

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Disable "Polyfit may be poorly conditioned" warning for testing
warnings.simplefilter('ignore', np.RankWarning)

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

    # For storage purposes
    actual_means = []
    actual_stddevs = []
    actual_dists = []
    pred_means = []
    pred_stddevs = []
    pred_dists = []

    with torch.no_grad():
        for data_point, label in test_dataloader:
            data_point, label = data_point.to(DEVICE).float(), label.to(DEVICE).float()[0]

            # Call the model
            pred = model(data_point)[0]

            # Curr_ refers to current data point
            curr_actual_means = []
            curr_actual_stddevs = []
            curr_actual_dists = []
            curr_pred_means = []
            curr_pred_stddevs = []
            curr_pred_dists = []

            # Go through each dimension
            for dim in range(1, NUM_DIMENSIONS+1):
                # Acquire appropiate indices for features
                dim_mean_idxs = get_indices(mean=True, dims=dim)
                dim_stddev_idxs = get_indices(stddev=True,dims=dim)
                dim_dist_idxs = get_indices(dists=True, dims=dim)

                # Get the specified values; dim_ refers to current dimensions
                dim_actual_means = label[dim_mean_idxs]
                dim_actual_stddevs = label[dim_stddev_idxs]
                #dim_actual_dists = label[dim_dist_idxs]
                dim_actual_dists = np.argmax(label[dim_dist_idxs],keepdims=True)
                dim_pred_means = pred[dim_mean_idxs]
                dim_pred_stddevs = pred[dim_stddev_idxs]
                #dim_pred_dists = pred[dim_dist_idxs]
                dim_pred_dists = np.argmax(pred[dim_dist_idxs],keepdims=True)

                curr_actual_means.append(dim_actual_means)
                curr_actual_stddevs.append(dim_actual_stddevs)
                curr_actual_dists.append(dim_actual_dists)
                curr_pred_means.append(dim_pred_means)
                curr_pred_stddevs.append(dim_pred_stddevs)
                curr_pred_dists.append(dim_pred_dists)
            
            # Store them for processing
            actual_means.append(curr_actual_means)
            actual_stddevs.append(curr_actual_stddevs)
            actual_dists.append(curr_actual_dists)
            pred_means.append(curr_pred_means)
            pred_stddevs.append(curr_pred_stddevs)
            pred_dists.append(curr_pred_dists)


    # actual_stuff now looks like:
    # [[x1_dim1_stuff, x1_dim2_stuff, ...],
    #  [x2_dim2_stuff, x2_dim2_stuff, ...],
    #  ...]
    # We want something for the form:
    # [[x1_dim1_stuff, x2_dim1_stuff, ...],
    #  [x1_dim2_stuff, x2_dim2_stuff, ...],
    #  ...]
    actual_means = np.transpose(actual_means)[0]
    actual_stddevs = np.transpose(actual_stddevs)[0]
    actual_dists = np.transpose(actual_dists)[0]
    pred_means = np.transpose(pred_means)[0]
    pred_stddevs = np.transpose(pred_stddevs)[0]
    pred_dists = np.transpose(pred_dists)[0]

    mean_maes = []
    mean_mapes = []
    mean_r2_scores = []
    stddev_maes = []
    stddev_mapes = []
    stddev_r2_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # If there's a better way to do this, go ahead
    # But note that averaging across dimensions first then taking the MAE
    # is NOT the same as MAE then averaging across dimensions (what this nonsense is doing)
    # Also don't ask me why all the other ranges are offset and this one isn't
    for dim in range(NUM_DIMENSIONS):
        curr_mean_mae = mean_absolute_error(actual_means[dim], pred_means[dim])
        curr_mean_mape = mean_absolute_percentage_error(actual_means[dim], pred_means[dim])
        curr_mean_r2_score = r2_score(actual_means[dim], pred_means[dim])
        curr_stddev_mae = mean_absolute_error(actual_stddevs[dim], pred_stddevs[dim])
        curr_stddev_mape = mean_absolute_percentage_error(actual_stddevs[dim], pred_stddevs[dim])
        curr_stddev_r2_score = r2_score(actual_stddevs[dim], pred_stddevs[dim])
        curr_accuracy = accuracy_score(actual_dists[dim],pred_dists[dim])
        # More complicated...
        curr_precision = precision_score(actual_dists[dim],
                                         pred_dists[dim],
                                         labels=range(len(DISTRIBUTIONS)),
                                         average=None,
                                         zero_division=0.0)
        curr_recall = recall_score(actual_dists[dim],
                                   pred_dists[dim],
                                   labels=range(len(DISTRIBUTIONS)),
                                   average=None,
                                   zero_division=0.0)
        curr_f1 = f1_score(actual_dists[dim],
                           pred_dists[dim],
                           labels=range(len(DISTRIBUTIONS)),
                           average=None,
                           zero_division=0.0)

        mean_maes.append(curr_mean_mae)
        mean_mapes.append(curr_mean_mape)
        mean_r2_scores.append(curr_mean_r2_score)
        stddev_maes.append(curr_stddev_mae)
        stddev_mapes.append(curr_stddev_mape)
        stddev_r2_scores.append(curr_stddev_r2_score)
        accuracy_scores.append(curr_accuracy)
        precision_scores.append(curr_precision)
        recall_scores.append(curr_recall)
        f1_scores.append(curr_f1)

    # Take means across all dimensions
    out = {
        "mean_mae": np.mean(mean_maes),
        "mean_mape": np.mean(mean_mapes),
        "mean_r2_score": np.mean(mean_r2_scores),
        "stddev_mae": np.mean(stddev_maes),
        "stddev_mape": np.mean(stddev_mapes),
        "stddev_r2_score": np.mean(stddev_r2_scores),
        "accuracy": np.mean(accuracy_scores),
        "mean_precision" : np.mean(precision_scores),
        "mean_recall": np.mean(recall_scores),
        "mean_f1": np.mean(f1_scores),
        "precision_scores" : np.mean(precision_scores,axis=0),
        "recall_scores": np.mean(recall_scores,axis=0),
        "f1_scores": np.mean(f1_scores,axis=0),
    }
    return out

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

        # Test the model, round to 3 decimal places
        
        test_results = test_model(model)

        decimals = 10 ** 3
        for key, value in test_results.items():
            test_results[key] = np.trunc(decimals * value) / decimals

        mean_mae = test_results['mean_mae']
        mean_mape = test_results['mean_mape']
        mean_r2_score = test_results['mean_r2_score']
        stddev_mae = test_results['stddev_mae']
        stddev_mape = test_results['stddev_mape']
        stddev_r2_score = test_results['stddev_r2_score']
        accuracy = test_results['accuracy']
        mean_precision = test_results['mean_precision']
        mean_recall = test_results['mean_recall']
        mean_f1 = test_results['mean_f1']
        precision = test_results['precision_scores']
        recall = test_results['recall_scores']
        f1 = test_results['f1_scores']

        mean_maes.append(mean_mae)
        mean_mapes.append(mean_mape)
        mean_r2_scores.append(mean_r2_score)
        stddev_maes.append(stddev_mae)
        stddev_mapes.append(stddev_mape)
        stddev_r2_scores.append(stddev_r2_score)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        model_end = time.time()
        logger.info(f"{HYPERPARAMETER}={hyperparam_val}"
                    f"\n Regression:"
                    f"\n\t\t-->  Mean MAE = {mean_mae:.2f}\t  Stddev MAE = {stddev_mae:.2f} "
                    f"\n\t\t--> Mean MAPE = {mean_mape:.2f}\t Stddev MAPE = {stddev_mape:.2f}"
                    f"\n\t\t-->   Mean R2 = {mean_r2_score:.2f}\t   Stddev R2 = {stddev_r2_score:.2f} "
                    f"\n Classification:"
                    f"\n\t\t-->         Accuracy = {accuracy}"
                    f"\n\t\t-->   Mean precision = {mean_precision}"
                    f"\n\t\t-->      Mean recall = {mean_recall}"
                    f"\n\t\t-->    Mean F1 score = {mean_f1}"
                    f"\n\t\t--> Precision scores = {precision}"
                    f"\n\t\t-->    Recall scores = {recall}"
                    f"\n\t\t-->        F1 scores = {f1}"
                    f"\n(Finished in {model_end - model_start:.2f} seconds)"
        )

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
