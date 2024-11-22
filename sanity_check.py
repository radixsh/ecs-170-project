import sys
import matplotlib.pyplot as plt
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader

from env import CONFIG, NUM_DIMENSIONS, DEVICE
from generate_data import generate_data, DISTRIBUTIONS, MyDataset
from build_model import build_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def generate_all9():
    data = []
    for dist_name, dist_class in DISTRIBUTIONS.items():
        instance = dist_class()
        points = instance.rng(CONFIG['SAMPLE_SIZE'])
        labels = instance.get_label()
        data.append((points, labels))
    return data

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

# Sanity-check the model's performance (good for presentation)
def main():
    if len(sys.argv) != 2:
        print(f'Usage: sanity_check.py models/your_model_weights.pth')
        sys.exit(0)

    # Build the model
    input_size = CONFIG['SAMPLE_SIZE'] * NUM_DIMENSIONS
    output_size = (len(DISTRIBUTIONS) + 2) * NUM_DIMENSIONS
    model = build_model(input_size, output_size).to(DEVICE)

    # Load the model's weights
    weights_file = sys.argv[1]
    state_dict = torch.load(weights_file)
    if state_dict is None:
        logger.debug("State dict is illegible! Quitting")
        sys.exit(0)
    model.load_state_dict(state_dict)
    logger.debug(f'Analyzing model from {weights_file}')

    # Generate a DataLoader with 9 entries: one per distribution family
    raw_data = generate_all9()
    samples = np.array([elem[0] for elem in raw_data])
    labels = np.array([elem[1] for elem in raw_data])
    dataset = MyDataset(samples, labels)
    # Set batch_size=1 to allow us to iterate through the dataloader one query
    # at a time
    dataloader = DataLoader(dataset, batch_size=1)

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
            actual_domain = get_domain(actual_dist.support)
            actual_label = (f"actual: {actual_dist.name} "
                            f"(mean={actual_dist.mean:.2f})")
            plt.plot(actual_domain, actual_dist.pdf(actual_domain),
                     color=actual_color, label=actual_label)
            logger.debug(f"actual: {actual_dist}")

            guess_dist = get_function(guesses)
            guess_domain = get_domain(guess_dist.support)
            guess_label = (f"guess: {guess_dist.name} "
                           f"(mean={actual_dist.mean:.2f})")
            plt.plot(actual_domain, guess_dist.pdf(actual_domain),
                     color='orangered', label=guess_label)
            logger.debug(f"guess: {guess_dist}")

            plt.xlabel('Domain')
            plt.ylabel('Probability distribution functions')
            plt.title('Predicted vs actual pdfs')
            plt.legend()

            plt.show()

if __name__ == "__main__":
    main()
