import sys
import matplotlib.pyplot as plt
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader

from env import CONFIG, NUM_DIMENSIONS, DEVICE
from generate_data import generate_data, DISTRIBUTION_FUNCTIONS, MyDataset
from build_model import build_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# defunct, ignore
'''
def plot_both(actuals, guesses):
    # actuals = []
    # First plot the actual continuous function (from labels)

    # Then make scatter plot of what was provided

    # Then plot the continuous function guessed by model
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
'''

def generate_all9():
    data = []
    for name, function in DISTRIBUTION_FUNCTIONS.items():
        points, labels = function(CONFIG['SAMPLE_SIZE'])
        data.append((points, labels))
    return data

# Input: x_values array and a `labels` vector. The labels vector might look like
# [0, 0, 0, 0, 0, 0, 1, 0, 0, 3.14, 2.18].
# Output: the appropriate numpy/scikit function, called on the given params.
# For the above example, it would be scipy.stats.norm.pdf(x_values, 3.14, 2.18).
def magic(x_values, labels):
    # TODO
    pass

# Sanity-check the model's performance (good for presentation)
def main():
    if len(sys.argv) != 2:
        print(f'Usage: sanity_check.py models/your_model_weights.pth')
        sys.exit(0)

    # Build the model
    input_size = CONFIG['SAMPLE_SIZE'] * NUM_DIMENSIONS
    output_size = (len(DISTRIBUTION_FUNCTIONS) + 2) * NUM_DIMENSIONS
    model = build_model(input_size, output_size).to(DEVICE)

    # Load the model's weights
    weights_file = sys.argv[1]
    state_dict = torch.load(weights_file)
    if state_dict is None:
        logger.debug("State dict is illegible! Quitting")
        sys.exit(0)
    model.load_state_dict(state_dict)
    logger.debug(f'Analyzing model from {weights_file}')

    # Get a DataLoader with 9 entries: one per distribution family
    raw_data = generate_all9()
    samples = np.array([elem[0] for elem in raw_data])
    labels = np.array([elem[1] for elem in raw_data])
    dataset = MyDataset(samples, labels)
    # Set batch_size=1 because we will want to iterate through the dataloader, one
    # query at a time
    dataloader = DataLoader(dataset, batch_size=1)

    model.eval()
    guesses = []
    # Graph the sample points provided to the model on a scatter plot, and
    # overlay both the model's guesses and the true distribution family, mean,
    # and standard deviation
    with torch.no_grad():
        for points, labels in dataloader:
            points, labels = points.to(DEVICE).float(), labels.to(DEVICE).float()

            # The first item is the only item because batch_size=1
            guesses = model(points)[0]

            fig, ax = plt.subplots()

            # Again, get first item only
            points = points[0]
            ax.scatter(points, [0 for i in points])

            x = np.arange(-100, 100)    # Maybe replace with np.linspace()?

            # Possibly helpful, possibly not...
            # dists = labels[:len(DISTRIBUTION_FUNCTIONS)]
            # one_hot_index = np.argmax(dists)
            # items_list = list(DISTRIBUTION_FUNCTIONS.items())
            # function_name, _ = items_list[one_hot_index]

            plt.plot(x, magic(x, labels))
            plt.plot(x, magic(x, guesses))

            plt.show()

if __name__ == "__main__":
    main()
