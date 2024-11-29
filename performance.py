import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import torch
import os
from torch.utils.data import DataLoader
import warnings
from collections import defaultdict

from env import *
from train_multiple import get_dataloader
from custom_functions import *

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

    loss_function = CustomLoss(num_dimensions=CONFIG['NUM_DIMENSIONS'])
    metrics = defaultdict(list)

    losses = []

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(DEVICE).float(), y.to(DEVICE).float()

            pred = model(X)
            loss = loss_function(pred, y)
            losses.append(loss.item())

            test_metrics = calculate_metrics(pred, y, CONFIG['NUM_DIMENSIONS'], mode='TEST')
            
            # Aggregate batch metrics
            # This is why we kept losses separate
            for key, value in test_metrics.items():
                metrics[key].append(value)

    # Add on the loss
    metrics['loss'] = losses

    # Average across the dataset, but need to handle extra classification
    # Definitely a little kludgey but fast enough
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']

    metrics = {key: np.mean(value) for key, value in metrics.items()}

    metrics['precision'] = np.mean(precision,axis=0)
    metrics['recall'] = np.mean(recall,axis=0)
    metrics['f1'] = np.mean(f1,axis=0)

    return metrics

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
        plt.plot(x_values, trend, color="royalblue", label=f"Slope: {slope:.3f}")
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

    metrics = defaultdict(list)

    hyperparams = []       # For matplotlib graphs
    count = 0

    for hyperparam_val in VALUES:
        
        # Update CONFIG with the hyperparameter
        CONFIG[HYPERPARAMETER] = hyperparam_val
        
        # Filename to search for
        model_filename = make_weights_filename(CONFIG)
        
        logger.info(f'Analyzing model at {model_filename}...')

        # Try to load it in
        state_dict = torch.load(model_filename)
        if state_dict is None:
            # Could add more info to this debug.
            logger.debug("Model not found, skipping!")
            continue

        model_start = time.time()              # For timing
        hyperparams.append(hyperparam_val)     # For matplotlib graphs

        # Reinitialize
        model = MultiTaskModel(CONFIG, MODEL_ARCHITECTURE, len(DISTRIBUTIONS)).to(DEVICE)
        model.load_state_dict(state_dict)

        # Test the model
        test_results = test_model(model)

        # Round to 3 decimal place 
        # Useful for precision, recall, and f1
        decimals = 10 ** 3
        for key, value in test_results.items():
            test_results[key] = np.trunc(decimals * value) / decimals

        # Save the test results in metrics
        for key,value in test_results.items():
            metrics[key].append(value)
        
        model_end = time.time()
        logger.info(f"{HYPERPARAMETER}={hyperparam_val}"
                    f"\n Regression:"
                    f"\n\t\t-->  Mean MAE = {metrics['mean_mae'][count]:.3f}\t  Stddev MAE = {metrics['stddev_mae'][count]:.3f} "
                    f"\n\t\t--> Mean MAPE = {metrics['mean_mape'][count]:.3f}\t Stddev MAPE = {metrics['stddev_mape'][count]:.3f}"
                    f"\n\t\t--> Mean RMSE = {metrics['mean_rmse'][count]:.3f}\t Stddev RMSE = {metrics['stddev_rmse'][count]:.3f}"
                    f"\n\t\t-->   Mean R2 = {metrics['mean_r2'][count]:.3f}\t   Stddev R2 = {metrics['stddev_r2'][count]:.3f} "
                    f"\n Classification:"
                    f"\n\t\t-->         Accuracy = {metrics['accuracy'][count]:.3f}"
                    f"\n\t\t-->   Mean precision = {metrics['avg_precision'][count]:.3f}"
                    f"\n\t\t-->      Mean recall = {metrics['avg_recall'][count]:.3f}"
                    f"\n\t\t-->    Mean F1 score = {metrics['avg_f1'][count]:.3f}"
                    f"\n\t\t--> Precision scores = {metrics['precision'][count]:}"
                    f"\n\t\t-->    Recall scores = {metrics['recall'][count]:}"
                    f"\n\t\t-->        F1 scores = {metrics['f1'][count]:}"
                    f"\n Loss:"
                    f"\n\t\t-->          Loss = {metrics['loss'][count]:.3f}"
                    f"\n(Finished in {model_end - model_start:.3f} seconds)"
        )
        count += 1

    end = time.time()
    logger.info(f"Analyzed {count} models in {end - start:.3f} seconds")
    if count == 0:
        return

    regression_png("MAE (mean average error)",
               hyperparams, metrics['mean_mae'], metrics['stddev_mae'])
    regression_png("MAPE (mean average percentage error)",
               hyperparams, metrics['mean_mape'], metrics['stddev_mape']),
    regression_png("RMSE (root mean squared error)",
                hyperparams, metrics['mean_rmse'], metrics['stddev_rmse'])
    regression_png("R^2 (correlation coefficient)",
               hyperparams, metrics['mean_r2'], metrics['stddev_r2'])
    classification_png("Accuracy", hyperparams, metrics['accuracy'])
    classification_png("F1", hyperparams, metrics['f1'])
    classification_png("Recall", hyperparams, metrics['recall'])
    classification_png("Precision", hyperparams, metrics['precision'])

if __name__ == "__main__":
    main()
