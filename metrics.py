from collections import defaultdict
import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from distributions import NUM_DISTS
from model import get_indices

### Illegal imports:
# env, core, data_handling, generate_data, sanity_check, performance, train_multiple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def calculate_metrics(pred, y, num_dimensions, mode):
    """
    Calculates the model's performance on a battery of tests and averages them over the
    dimensionality of the data. For classification tasks, these are accuracy, recall,
    precision, and f1. For regression tasks, these are r2, MAE, MAPE, and RMSE.
    No loss calculations are performed.

    Args:
        pred (tensor): Predicted values.
        y (tensor): Ground-truth values.
        num_dimensions (int): The number of dimensions in the data.
        mode (str): 'TEST' or 'TRAIN', controls the complexity of desired metrics.

    Returns:
        dict: A dict containing floats and lists of floats, each value is
              a particular measurement of model performance.
    """
    metrics = defaultdict(list)
    confusion_matrices = []  # Store confusion matrix data for later display

    for dim in range(num_dimensions):
        # Loop through the dimensions and take the average over each prediction-target
        # pair, then average over dimensions. The opposite order is not the same.
        dists_idx = get_indices(dim+1, NUM_DISTS, dists=True)
        mean_idx = get_indices(dim+1, NUM_DISTS, mean=True)
        stddev_idx = get_indices(dim+1, NUM_DISTS, stddev=True)
        support_idx = get_indices(dim+1, NUM_DISTS, support=True)

        # Classification metrics need class indices instead of onehot
        class_targets = torch.argmax(y[:, dists_idx], dim=1).numpy()
        class_preds = torch.argmax(pred["classification"][:, dim, :], dim=1).numpy()

        support_targets = torch.argmax(y[:, support_idx], dim=1).numpy()
        support_preds = torch.argmax(pred["support"][:, dim, :], dim=1).numpy()

        mean_targets = y[:, mean_idx].numpy()
        mean_preds = pred["mean"][:, dim].detach().numpy()

        stddev_targets = y[:, stddev_idx].numpy()
        stddev_preds = pred["stddev"][:, dim].detach().numpy()

        # Standard metrics to be calculated every epoch.
        metrics["accuracy"].append(accuracy_score(class_targets, class_preds))
        metrics["support_accuracy"].append(accuracy_score(support_targets, support_preds))
        metrics["mean_r2"].append(r2_score(mean_targets, mean_preds))
        metrics["stddev_r2"].append(r2_score(stddev_targets, stddev_preds))

        #TODO: AttibuteError, wrong object attribution when running model?
        # Collect confusion matrix data for later visualization in `display_metrics()`.
        if mode == "TEST":
            cm = confusion_matrix(class_targets, class_preds, labels=range(NUM_DISTS))
            confusion_matrices.append((dim, cm))

        # Test-only metrics.
        if mode == "TEST":
            metrics["mean_mae"].append(mean_absolute_error(mean_targets, mean_preds))
            metrics["mean_rmse"].append(
                np.sqrt(mean_squared_error(mean_targets, mean_preds))
            )
            metrics["stddev_mae"].append(
                mean_absolute_error(stddev_targets, stddev_preds)
            )
            metrics["stddev_rmse"].append(
                np.sqrt(mean_squared_error(stddev_targets, stddev_preds))
            )
            # Each of these outputs a list containing class-specific metrics.
            # The nomenclature is for convenience, each of these will be copied
            # and the original will be averaged across classes.
            metrics["avg_precision"].append(
                precision_score(
                    class_targets,
                    class_preds,
                    labels=range(NUM_DISTS),
                    average=None,
                    zero_division=0.0,
                )
            )
            metrics["avg_recall"].append(
                recall_score(
                    class_targets,
                    class_preds,
                    labels=range(NUM_DISTS),
                    average=None,
                    zero_division=0.0,
                )
            )
            metrics["avg_f1"].append(
                f1_score(
                    class_targets,
                    class_preds,
                    labels=range(NUM_DISTS),
                    average=None,
                    zero_division=0.0,
                )
            )
            metrics["avg_support_precision"].append(
                precision_score(
                    support_targets,
                    support_preds,
                    labels=range(3),
                    average=None,
                    zero_division=0.0,
                )
            )
            metrics["avg_support_recall"].append(
                recall_score(
                    support_targets,
                    support_preds,
                    labels=range(3),
                    average=None,
                    zero_division=0.0,
                )
            )
            metrics["avg_support_f1"].append(
                f1_score(
                    support_targets,
                    support_preds,
                    labels=range(3),
                    average=None,
                    zero_division=0.0,
                )
            )

    # Preserve class-specific metrics, calling np.mean without axis=0 yields one float.
    if mode == "TEST":
        precision = np.mean(metrics["avg_precision"], axis=0)
        recall = np.mean(metrics["avg_recall"], axis=0)
        f1 = np.mean(metrics["avg_f1"], axis=0)
        support_precision = np.mean(metrics["avg_support_precision"], axis=0)
        support_recall = np.mean(metrics["avg_support_recall"], axis=0)
        support_f1 = np.mean(metrics["avg_support_f1"], axis=0)

    # Take the average over the dimensionality of the data.
    metrics = {key: np.mean(value) for key, value in metrics.items()}

    if mode == "TEST":
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["support_precision"] = support_precision
        metrics["support_recall"] = support_recall
        metrics["support_f1"] = support_f1

    # Return metrics and confusion matrix data
    return metrics, confusion_matrices


def display_metrics(metrics, mode, epoch=-1, confusion_matrices=None):
    """
    Displays the loss, classification metrics (accuracy, precision, recall, f1), and
    regression metrics (r2, MAE, MAPE, RMSE) averaged over the model's full run.
    Precision, recall, and f1 are all initially 2D arrays, and need to have np.mean
    called differently to preserve individual class data.

    Args:
        metrics (dict): The performance data of the model over all epochs.
        mode (str): "TRAIN" or "TEST", controls the detail of printed metrics.
        confusion_matrices (list): Optional confusion matrix data to display.

    Returns:
        dict: A dict containing averaged performance metrics.
    """
    if mode == "TEST":
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        support_precision = metrics["support_precision"]
        support_recall = metrics["support_recall"]
        support_f1 = metrics["support_f1"]

    metrics = {key: np.mean(value) for key, value in metrics.items()}

    if mode == "TRAIN":
        logger.debug(
            f"-------------------------------------"
            f" EPOCH {epoch} "
            f"-------------------------------------"
            f"\nMetrics:"
            f"\n\t-->        Loss: {metrics['loss']:.6f}"
            f"\n\t-->     Mean R2: {metrics['mean_r2']:.6f}"
            f"\n\t-->   Stddev R2: {metrics['stddev_r2']:.6f}"
            f"\n\t-->    Accuracy: {metrics['accuracy']:.6f}"
            f"\n\t--> Support Acc: {metrics['support_accuracy']:.6f}"
            
        )
    elif mode == "TEST":
        metrics["precision"] = np.mean(precision, axis=0)
        metrics["recall"] = np.mean(recall, axis=0)
        metrics["f1"] = np.mean(f1, axis=0)
        metrics["support_precision"] = np.mean(support_precision, axis=0)
        metrics["support_recall"] = np.mean(support_recall, axis=0)
        metrics["support_f1"] = np.mean(support_f1, axis=0)
        for key, value in metrics.items():
            metrics[key] = np.round(value, decimals=3)
        logger.info(
            f"Regression:     \tMean\tStddev"
            f"\n\t-->        R2: {metrics['mean_r2']:.3f}"
            f"\t{metrics['stddev_r2']:.3f}"
            f"\n\t-->       MAE: {metrics['mean_mae']:.3f}"
            f"\t{metrics['stddev_mae']:.3f}"
            f"\n\t-->      RMSE: {metrics['mean_rmse']:.3f}"
            f"\t{metrics['stddev_rmse']:.3f}"
            f"\nClassification:      Average\tIndividual Classes"
            f"\n\t-->  Accuracy: {metrics['accuracy']:.3f}"
            f"\n\t--> Precision: {metrics['avg_precision']:.3f}"
            f"\t{metrics['precision']}"
            f"\n\t-->    Recall: {metrics['avg_recall']:.3f}"
            f"\t{metrics['recall']}"
            f"\n\t-->  F1 Score: {metrics['avg_f1']:.3f}"
            f"\t{metrics['f1']}"
            f"\nSupport:             Average\tIndividual Classes"
            f"\n\t-->  Accuracy: {metrics['support_accuracy']:.3f}"
            f"\n\t--> Precision: {metrics['avg_support_precision']:.3f}"
            f"\t{metrics['support_precision']}"
            f"\n\t-->    Recall: {metrics['avg_support_recall']:.3f}"
            f"\t{metrics['support_recall']}"
            f"\n\t-->  F1 Score: {metrics['avg_support_f1']:.3f}"
            f"\t{metrics['support_f1']}"
            f"\nLoss:\t\t(Non-performance, diagnostic only)"
            f"\n\t-->  Avg Loss: {metrics['loss']:.3f}"
        )

        # TODO: (Debug) Display confusion matrices if provided
        if confusion_matrices:
            for dim, cm in confusion_matrices:
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=[f"Class {i}" for i in range(NUM_DISTS)],
                )
                disp.plot(cmap="viridis")
                disp.ax_.set_title(f"Confusion Matrix (Dimension {dim + 1})")

    return metrics
