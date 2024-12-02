import os

import matplotlib.pyplot as plt
import numpy as np

### No importing custom files.


def regression_png(name, x_values, means, stddevs, hyperparameter):
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
    plt.plot(x_values, mean_trend, color="royalblue", label=f"Mean slope: {mean_slope}")

    plt.scatter(x_values, stddevs, color="tomato", label="Stddevs")
    stddev_slope, stddev_intercept = np.polyfit(x_values, stddevs, deg=1)
    stddev_trend = stddev_slope * x_values + stddev_intercept
    plt.plot(
        x_values, stddev_trend, color="tomato", label=f"Stddev slope: {stddev_slope}"
    )

    plt.gca().set_xscale("log")
    plt.xlabel(hyperparameter)

    plt.title(name)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    destination = f"{hyperparameter.lower()}_{name.replace(' ', '_')}.png"
    plt.savefig(os.path.join("results", destination), bbox_inches="tight")
    plt.show()


def classification_png(name, x_values, metrics, hyperparameter):
    # Hardcoded to avoid import issues.
    distribution_indices = [
        "beta",
        "gamma",
        "gumbel",
        "laplace",
        "logistic",
        "lognormal",
        "normal",
        "rayleigh",
        "wald",
    ]
    colors = [
        "royalblue",
        "tomato",
        "forestgreen",
        "darkorange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "black",
    ]
    plt.ylim(0, 1)
    plt.ylabel(name)

    x_values = np.float64(x_values)
    sorted_indices = np.argsort(x_values)
    x_values = x_values[sorted_indices]

    if isinstance(metrics[0], float):  # Handle accuracy differently
        metrics = np.array(metrics)[sorted_indices]  # Sort 1D metrics array
        plt.scatter(x_values, metrics, color="royalblue", label=name)
        slope, intercept = np.polyfit(x_values, metrics, deg=1)
        trend = slope * x_values + intercept
        plt.plot(x_values, trend, color="royalblue", label=f"Slope: {slope:.3f}")
    else:  # Sort 2D metrics array for other metrics (F1, Recall, Precision)
        metrics = np.array(metrics)[sorted_indices, :]
        for i in range(metrics.shape[1]):  # Loop over distributions (columns)
            plt.scatter(
                x_values, metrics[:, i], color=colors[i], label=distribution_indices[i]
            )
            slope, intercept = np.polyfit(x_values, metrics[:, i], deg=1)
            trend = slope * x_values + intercept
            plt.plot(x_values, trend, color=colors[i])

    plt.gca().set_xscale("log")
    plt.xlabel(hyperparameter)

    plt.title(name)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    destination = f"{hyperparameter.lower()}_{name.replace(' ', '_')}.png"
    plt.savefig(os.path.join("data", destination), bbox_inches="tight")
    plt.show()
