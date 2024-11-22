import sys
import matplotlib.pyplot as plt
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from pprint import pformat

from env import CONFIG, NUM_DIMENSIONS, DEVICE
from generate_data import DISTRIBUTIONS, MyDataset
from custom_functions import parse_weights_filename
from build_model import build_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Returns a DataLoader with 9 entries: one per distribution family
def generate_1d():
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
    raw_data = []

    for outer_name, outer_class in DISTRIBUTIONS.items():
        # logger.debug(f'outer_name: {outer_name}')
        points, labels = [], []

        outer_class = DISTRIBUTIONS[outer_name]
        outer_object = outer_class()

        outer_points = outer_object.rng(CONFIG['SAMPLE_SIZE'])
        # outer_points_formatted = [f'{pt:.2f}' for pt in outer_points]
        # logger.debug(f'outer_points: {outer_points_formatted}')
        points.extend(outer_points)

        outer_labels = outer_object.get_label()
        labels += outer_labels
        # logger.debug(f'outer labels: {outer_labels}')

        for inner_name, inner_class in DISTRIBUTIONS.items():
            # logger.debug(f'\tinner_name: {inner_name}')
            temp_points, temp_labels = list(points), list(labels)

            inner_class = DISTRIBUTIONS[inner_name]
            inner_object = inner_class()

            inner_points = inner_object.rng(CONFIG['SAMPLE_SIZE'])
            # inner_points_formatted = [f'{pt:.2f}' for pt in inner_points]
            # logger.debug(f'\tinner_points: {inner_points_formatted}')
            # inner_points = list(inner_points)
            # logger.debug(f'\tinner_points, as a list: {inner_points}')
            temp_points.extend(inner_points)

            inner_labels = inner_object.get_label()
            temp_labels.extend(inner_labels)
            # logger.debug(f'\tinner labels: {inner_labels}')

            raw_data.append((temp_points, temp_labels))

    '''
    for _ in range(NUM_DIMENSIONS):
        points, labels = [], []
        for dist_name, dist_class in DISTRIBUTIONS.items():
            logger.debug(f'dist_name: {dist_name}')

            dist_class = DISTRIBUTIONS[dist_name]
            dist_object = dist_class()

            dist_points = dist_object.rng(CONFIG['SAMPLE_SIZE'])
            logger.debug(f'dist_points: {dist_points}')
            dist_points = list(dist_points)
            logger.debug(f'dist_points, as a list: {dist_points}')
            points.append(dist_points)

            dist_labels = dist_object.get_label()
            labels += dist_labels
            logger.debug(f'labels from this dist: {dist_labels}')

        logger.debug(len(points))
        logger.debug(f"All points: {points}")
        logger.debug(f"All points: {points}")
        raw_data.append((points, labels))

        # Flattened: [1,3,5, 2,4,6]
        # The actual points in the 2D distribution will be (1,2), (3,4), (5,6)
        points = np.ravel(points, order='C')
        sorted_and_formatted = [f'{point:.2f}' for point in points].sort()
        logger.debug(f"Flattened: {sorted_and_formatted}")
    '''

    # logger.debug(f"raw_data: {pformat(raw_data)}")
    logger.debug(f"len(raw_data)={len(raw_data)}")

    samples = np.array([elem[0] for elem in raw_data])
    labels = np.array([elem[1] for elem in raw_data])
    dataset = MyDataset(samples, labels)

    # Set batch_size=1 to allow us to iterate through the dataloader one query
    # at a time
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader

# Takes a one hot + mean + stddev vector
# Returns a function object (one of ours, in custom_functions.py)
def get_function(labels):
    # Get the one-hot part of the labels vector
    dists = labels[:len(DISTRIBUTIONS)]

    # Identify the distribution family from the one-hot vector
    one_hot_index = np.argmax(dists)

    # Get the distribution class
    items_list = list(DISTRIBUTIONS.items())
    dist_name, dist_class = items_list[one_hot_index]

    # Initialize and return an instance of this class
    logger.debug(f"Initializing member of {dist_name} with "
                 f"mean={labels[-2]}, stddev={labels[-1]}")
    dist_object = dist_class(mean=labels[-2], stddev=labels[-1])
    return dist_object

# Takes a string which is 'R', 'R+', or 'I'
# Returns a linspace object for pyplot to graph
def get_domain(support):
    if support == 'R':
        return np.linspace(-10, 10, 100)
    elif support == 'R+':
        return np.linspace(0, 10, 100)
    elif support == 'I':
        return np.linspace(0, 1, 100)
    else:
        print(f"Can't recognize domain {support}")
        return np.linspace(-10, 10, 100)

def model_setup():
    filename = sys.argv[1]
    weights_info = parse_weights_filename(filename)

    # Build the model
    input_size = weights_info['SAMPLE_SIZE'] * weights_info['NUM_DIMS'] 
    output_size = (len(DISTRIBUTIONS) + 2) * NUM_DIMENSIONS
    model = build_model(input_size, output_size).to(DEVICE)

    # Load the model's weights
    state_dict = torch.load(filename)
    if state_dict is None:
        logger.debug("State dict is illegible! Quitting")
        sys.exit(0)
    model.load_state_dict(state_dict)

    logger.debug(f'Analyzing model from {filename}')
    return model

def test_1d(model, dataloader):
    model.eval()
    guesses = []
    # Graph the sample points provided to the model on a scatter plot, and
    # overlay both the model's guess and the true distribution family
    with torch.no_grad():
        for points, labels in dataloader:
            points, labels = points.to(DEVICE).float(), labels.to(DEVICE).float()

            # The first item in this batch is the only item because batch_size=1
            guesses = model(points)[0]
            points = points[0]
            labels = labels[0]

            fig, ax = plt.subplots()

            actual_color = 'royalblue'
            ax.scatter(points, [0 for i in points], color=actual_color, s=70,
                       alpha=0.3)
            logger.debug(f"\npoints: {points}")

            actual_dist = get_function(labels)
            logger.debug(f"actual: {actual_dist}")
            actual_domain = get_domain(actual_dist.support)
            actual_label = (f"actual: {actual_dist.name} "
                            f"(mean={actual_dist.mean:.2f})")
            plt.plot(actual_domain, actual_dist.pdf(actual_domain),
                     color=actual_color, label=actual_label)

            guess_dist = get_function(guesses)
            logger.debug(f"guess: {guess_dist}")
            guess_domain = get_domain(guess_dist.support)
            guess_label = (f"guess: {guess_dist.name} "
                           f"(mean={actual_dist.mean:.2f})")
            plt.plot(actual_domain, guess_dist.pdf(actual_domain),
                     color='orangered', label=guess_label)

            plt.xlabel('Domain')
            plt.ylabel('Probability distribution functions')
            plt.title('Predicted vs actual pdfs')
            plt.legend()

            plt.show()

# Takes a list of 2d points: [3,2, 5,9], representing the points (3,2) and (5,9)
# Returns a tuple of points by each dimension: ([3,5], [2,9])
def list_to_tuple(points):
    logger.debug(f'list_to_tuple received: {points}')
    dim1, dim2 = [], []
    for i, value in enumerate(points):
        if i % 2 == 0:
            dim1.append(value)
        else:
            dim2.append(value)
    logger.debug(f'list_to_tuple returns: {(dim1, dim2)}')
    return (dim1, dim2)

def test_2d(model, dataloader):
    model.eval()
    guesses = []
    # Graph the sample points provided to the model on a scatter plot, and
    # overlay both the model's guess and the true distribution family
    with torch.no_grad():
        for points, labels in dataloader:
            points, labels = points.to(DEVICE).float(), labels.to(DEVICE).float()

            # The first item in this batch is the only item because batch_size=1
            guesses = model(points)[0]
            points_dim1, points_dim2 = list_to_tuple(points[0])
            labels = labels[0]

            # fig, ax = plt.subplots()
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            actual_color = 'royalblue'
            ax.scatter(points_dim1, points_dim2, [0 for i in points],
                       color=actual_color, s=70, alpha=0.3)
            logger.debug(f"\npoints: {points}")

            actual_dist = get_function(labels)
            logger.debug(f"actual: {actual_dist}")
            actual_domain = get_domain(actual_dist.support)
            actual_label = (f"actual: {actual_dist.name} "
                            f"(mean={actual_dist.mean:.2f})")
            plt.plot(actual_domain, actual_dist.pdf(actual_domain),
                     color=actual_color, label=actual_label)

            guess_dist = get_function(guesses)
            logger.debug(f"guess: {guess_dist}")
            guess_domain = get_domain(guess_dist.support)
            guess_label = (f"guess: {guess_dist.name} "
                           f"(mean={actual_dist.mean:.2f})")
            plt.plot(actual_domain, guess_dist.pdf(actual_domain),
                     color='orangered', label=guess_label)

            plt.xlabel('Domain')
            plt.ylabel('Probability distribution functions')
            plt.title('Predicted vs actual pdfs')
            plt.legend()

            plt.show()

# Sanity-check the model's performance (good for presentation)
def main():
    if len(sys.argv) != 2:
        print(f'Usage: sanity_check.py models/your_model_weights.pth')
        sys.exit(0)

    model = model_setup()

    if NUM_DIMENSIONS == 1:
        dataloader = generate_1d()
        test_1d(model, dataloader)
    elif NUM_DIMENSIONS == 2:
        dataloader = generate_2d()
        test_2d(model, dataloader)
    else:
        logger.warning(f"{NUM_DIMENSIONS} dimensions is not supported, exiting")
        sys.exit(0)


if __name__ == "__main__":
    main()
