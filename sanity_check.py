import sys
import matplotlib.pyplot as plt
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from pprint import pformat

from env import *
from generate_data import DISTRIBUTIONS, MyDataset
from custom_functions import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

GRAPH_FIDELITY = 1000

def generate_1d():
    """
    Returns a DataLoader with 9 entries, one for each distribution family.
    """
    raw_data = []
    for dist_name, dist_class in DISTRIBUTIONS.items():
        instance = dist_class()
        points = instance.rng(CONFIG['SAMPLE_SIZE'])
        labels = instance.get_label()
        raw_data.append((points, labels))

    samples = np.array([elem[0] for elem in raw_data])
    labels = np.array([elem[1] for elem in raw_data])
    dataset = MyDataset(samples, labels)

    # Set batch_size=1 to allow us to iterate through the dataloader one query
    # at a time
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader

def generate_2d():
    """
    Returns a DataLoader with 81 entries, representing every possible
    permutation of distribution families.
    """
    raw_data = []

    for outer_name, outer_class in DISTRIBUTIONS.items():
        points, labels = [], []

        outer_class = DISTRIBUTIONS[outer_name]
        outer_object = outer_class()

        outer_points = outer_object.rng(CONFIG['SAMPLE_SIZE'])
        points.extend(outer_points)

        outer_labels = outer_object.get_label()
        labels += outer_labels

        for inner_name, inner_class in DISTRIBUTIONS.items():
            temp_points, temp_labels = list(points), list(labels)

            inner_class = DISTRIBUTIONS[inner_name]
            inner_object = inner_class()

            inner_points = inner_object.rng(CONFIG['SAMPLE_SIZE'])
            temp_points.extend(inner_points)

            inner_labels = inner_object.get_label()
            temp_labels.extend(inner_labels)

            raw_data.append((temp_points, temp_labels))

    samples = np.array([elem[0] for elem in raw_data])
    labels = np.array([elem[1] for elem in raw_data])
    dataset = MyDataset(samples, labels)

    # Set batch_size=1 to allow us to iterate through the dataloader one query
    # at a time
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader

def get_domain(dist):
    """
    Turns a support string ("R", "R+", or "I") into a linspace object for pyplot
    to graph.
    """
    if dist.support == 'R':
        x_min = dist.mean - 3 * dist.stddev
        x_max = dist.mean + 3 * dist.stddev
        return np.linspace(x_min, x_max, GRAPH_FIDELITY)
    elif dist.support == 'R+':
        x_max = dist.mean + 3 * dist.stddev
        return np.linspace(0, x_max, GRAPH_FIDELITY)
    elif dist.support == 'I':
        return np.linspace(0, 1, GRAPH_FIDELITY)

def list_to_tuple(points):
    """
    Splits a list of 2D points into separate tuples by dimension.
    Example:
        Input: [3, 2, 5, 9]
        Output: ([3, 5], [2, 9])
    """
    dim1, dim2 = [], []
    for i, value in enumerate(points):
        if i % 2 == 0:
            dim1.append(value)
        else:
            dim2.append(value)
    return (dim1, dim2)

def get_dist_objects_dict(predictions):
    dist_objs = []
    for i in range(CONFIG['NUM_DIMENSIONS']):
        onehot = predictions['classification'][i]
        dist_idx = torch.argmax(onehot).item()
        dist_class = list(DISTRIBUTIONS.values())[dist_idx]

        mean = predictions['mean'][i].item()
        stddev = predictions['stddev'][i].item()

        myobj = dist_class(mean, stddev)
        dist_objs.append(myobj)

    return dist_objs

def get_dist_objects(labels):
    """
    Converts a list of labels into a list of Distribution instances.
    Example:
        Input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 3.14, 0.42,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 2.71, 0.62]
        Output: [Gamma(mean=3.14, stddev=0.42),
                 Gumbel(mean=2.71, stddev=0.62)]
    """
    if isinstance(labels, dict):
        return get_dist_objects_dict(labels)
    
    dist_objs = []
    for i in range(1, CONFIG['NUM_DIMENSIONS'] + 1):
        dist_indices = get_indices(dists=True, dim=i)
        onehot = labels[dist_indices]
        dist_idx = torch.argmax(onehot).item()
        dist_class = list(DISTRIBUTIONS.values())[dist_idx]

        mean_idx = get_indices(mean=True, dim=i)
        mean = labels[mean_idx].item()

        stddev_idx = get_indices(stddev=True, dim=i)
        stddev = labels[stddev_idx].item()

        myobj = dist_class(mean, stddev)
        dist_objs.append(myobj)

    return dist_objs


def test_1d(model, dataloader):
    """
    Plots the sample points provided to the model on a scatter plot, and
    overlays both the model's guess and the true distribution family.
    """
    guesses = []
    with torch.no_grad():
        for points, labels in dataloader:
            points, labels = points.to(DEVICE).float(), labels.to(DEVICE).float()

            # The first item is the only item because batch_size=1 for testing
            guesses = model(points)
            print(guesses)
            points = points[0]
            labels = labels[0]

            fig, ax = plt.subplots()

            actual_color = 'royalblue'
            # The 'alpha' parameter sets points to be slightly transparent,
            # showing us where points might be highly concentrated
            ax.scatter(points, [0 for i in points], color=actual_color, s=70,
                       alpha=0.3)
            logger.debug(f"\npoints: {points}")

            actual_dist = get_dist_objects(labels)[0]
            actual_domain = get_domain(actual_dist)
            actual_label = (f"actual: {actual_dist.name} "
                           f"(µ={actual_dist.mean:.2f}, "
                           f"σ={actual_dist.stddev:.2f})")
            logger.debug(actual_label)
            plt.plot(actual_domain, actual_dist.pdf(actual_domain),
                     color=actual_color, label=actual_label)

            guess_dist = get_dist_objects(guesses)[0]
            guess_domain = get_domain(guess_dist)
            guess_label = (f"guess: {guess_dist.name} "
                           f"(µ={guess_dist.mean:.2f}, "
                           f"σ={guess_dist.stddev:.2f})")
            logger.debug(guess_label)
            plt.plot(actual_domain, guess_dist.pdf(actual_domain),
                     color='orangered', label=guess_label)

            plt.xlabel('Domain')
            plt.ylabel('Probability distribution functions')
            plt.title("Actual (blue) vs predicted (orange) distributions")
            plt.legend()

            plt.show()

def test_2d(model, dataloader):
    """
    Plots the sample points provided to the model on a scatter plot, and
    overlays both the model's guess and the true distribution family.
    """
    guesses = []
    with torch.no_grad():
        for points, labels in dataloader:
            points, labels = points.to(DEVICE).float(), labels.to(DEVICE).float()

            # The first item is the only item because batch_size=1 for testing
            guesses = model(points)[0]
            labels = labels[0]

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_title("Actual (blue) vs predicted (orange) distributions")

            points_dim1, points_dim2 = list_to_tuple(points[0])
            actual_color = "blue"
            ax.scatter(points_dim1, points_dim2, [0 for i in points],
                       color=actual_color)

            # ====================
            # ACTUAL DISTRIBUTIONS
            # ====================

            actual_dim1, actual_dim2 = get_dist_objects(labels)

            actual_dim1_domain = get_domain(actual_dim1)
            actual_dim2_domain = get_domain(actual_dim2)
            ax.set_xlim(min(actual_dim1_domain), max(actual_dim1_domain))
            ax.set_ylim(min(actual_dim2_domain), max(actual_dim2_domain))

            X, Y = np.meshgrid(actual_dim1_domain, actual_dim2_domain)
            actual = actual_dim1.pdf(X) * actual_dim2.pdf(Y)
            ax.plot_surface(X, Y, actual,
                            color=actual_color, # cmap="Blues_r",
                            alpha=0.8,
                            edgecolor=None)
            logger.debug(f"actual: {actual_dim1.name} "
                         f"(µ={actual_dim1.mean:.2f}, "
                         f"σ={actual_dim1.stddev:.2f}), "
                         f"{actual_dim2.name} "
                         f"(µ={actual_dim2.mean:.2f}, "
                         f"σ={actual_dim2.stddev:.2f})")

            # =======================
            # PREDICTED DISTRIBUTIONS
            # =======================

            guess_dim1, guess_dim2 = get_dist_objects(guesses)

            guess = guess_dim1.pdf(X) * guess_dim2.pdf(Y)
            guess = guess_dim1.pdf(X) * guess_dim2.pdf(Y)
            ax.plot_surface(X, Y, guess,
                            color="orange", # cmap="Oranges_r",
                            alpha=0.8,
                            edgecolor=None)
            logger.debug(f"guess: {guess_dim1.name} "
                         f"(µ={guess_dim1.mean:.2f}, "
                         f"σ={guess_dim1.stddev:.2f}), "
                         f"{guess_dim2.name} "
                         f"(µ={guess_dim2.mean:.2f}, "
                         f"σ={guess_dim2.stddev:.2f})")

            x_label = (f"Actual: {actual_dim1.name} "
                       f"(µ={actual_dim1.mean:.2f}, "
                       f"σ={actual_dim1.stddev:.2f})\n"
                       f"Guess: {guess_dim1.name} "
                       f"(µ={guess_dim1.mean:.2f}, "
                       f"σ={guess_dim1.stddev:.2f})")
            ax.set_xlabel(x_label)
            y_label = (f"Actual: {actual_dim2.name} "
                       f"(µ={actual_dim2.mean:.2f}, "
                       f"σ={actual_dim2.stddev:.2f})\n"
                       f"Guess: {guess_dim2.name} "
                       f"(µ={guess_dim2.mean:.2f}, "
                       f"σ={guess_dim2.stddev:.2f})")
            ax.set_ylabel(y_label)

            plt.show()

def model_setup():
    filename = sys.argv[1]
    config = parse_weights_filename(filename)

    # Load the model's weights
    model = MultiTaskModel(config, MODEL_ARCHITECTURE, len(DISTRIBUTIONS)).to(DEVICE)
    state_dict = torch.load(filename)
    if state_dict is None:
        logger.debug("State dict is illegible! Quitting")
        sys.exit(0)
    model.load_state_dict(state_dict)

    logger.debug(f'Analyzing model from {filename}')
    return model

# Sanity-check the model's performance (good for presentation)
def main():
    if len(sys.argv) != 2:
        print(f'Usage: sanity_check.py models/your_model_weights.pth')
        sys.exit(0)

    model = model_setup()
    model.eval()

    if CONFIG['NUM_DIMENSIONS'] == 1:
        dataloader = generate_1d()
        test_1d(model, dataloader)
    elif CONFIG['NUM_DIMENSIONS'] == 2:
        dataloader = generate_2d()
        test_2d(model, dataloader)
    else:
        logger.warning(f"{CONFIG['NUM_DIMENSIONS']} dimensions is not supported, exiting")
        sys.exit(0)

if __name__ == "__main__":
    main()
