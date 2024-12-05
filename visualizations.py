import os
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

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

def visualize_weights(model, layer_names=None):
    """
    Visualizes the weights of a given model layer-by-layer with symmetric smoothstep normalization.

    Args:
        model (nn.Module): The PyTorch model.
        layer_names (list, optional): List of layer names to visualize. Defaults to None, visualizing all layers.
    """
    def smoothstep(x):
        """Smoothstep normalization: x / (1 + x)."""
        return x / (1 + x)

    for name, layer in model.named_modules():
        # Skip layers without weights
        if not hasattr(layer, 'weight') or layer.weight is None:
            continue
        
        # Filter specific layers if names are provided
        if layer_names and name not in layer_names:
            continue

        # Get the weights as a numpy array
        weights = np.abs(layer.weight.detach().cpu().numpy())
        
        # Apply symmetric smoothstep normalization
        normalized_weights = smoothstep(weights)
        
        # Plot the normalized weights
        plt.figure(figsize=(12, 8))
        sns.heatmap(normalized_weights, cmap='viridis', cbar=True, vmin=0, vmax=1)
        plt.title(f'Normalized Weights of Layer: {name}')
        plt.xlabel("Input Features")
        plt.ylabel("Output Features")
        plt.show()


def visualize_activations_avg(model, dataloader, device='cpu'):
    """
    Visualizes the normalized average activations (softmax over neurons) of a model over a batch of data,
    considering task-specific outputs.
    
    Args:
        model (nn.Module): The model to inspect.
        dataloader (DataLoader): DataLoader for input data.
        device (torch.device): Device to run the model on.
    """
    # Store activations from hooks
    activations = {}
    
    def smoothstep_normalize(tensor):
        return tensor / (1 + tensor)
    
    # Hook function
    def hook_fn(name):

        def hook(module, input, output):
            if isinstance(output, dict):
                activations[name] = {key: smoothstep_normalize(value.detach().cpu().mean(dim=0))
                                    for key, value in output.items()}
            else:
                avg_activation = output.detach().cpu().mean(dim=0)
                activations[name] = smoothstep_normalize(avg_activation)

        return hook

    # Register hooks for ReLU activations
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.ReLU):  # Only target ReLU layers
            hooks.append(layer.register_forward_hook(hook_fn(name)))

    # Pass a single batch of data through the model
    model.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device).float()
            model(X)
            break  # Process only one batch

    # Remove hooks after use
    for hook in hooks:
        hook.remove()

    # Plot activations
    for name, activation in activations.items():
        if isinstance(activation, dict):  # Task-specific activations
            for task, task_activation in activation.items():
                task_activation = task_activation.squeeze()  # Remove singleton dimensions
                if task_activation.ndimension() > 1:
                    # Handle multi-dimensional activations (e.g., for classification tasks)
                    task_activation = task_activation.view(-1)
                plt.figure(figsize=(12, 8))
                sns.heatmap(task_activation[np.newaxis, :], cmap='magma', cbar=True, vmin=0, vmax=1)
                plt.title(f"Normalized Average Activation - Layer: {name}, Task: {task}")
                plt.show()

                # Cumulative distribution plot
                plot_cumulative_distribution(task_activation, name, task)
        else:
            avg_activation = activation.squeeze()  # Remove singleton dimensions
            if avg_activation.ndimension() > 1:
                # Handle multi-dimensional activations
                avg_activation = avg_activation.view(-1)
            plt.figure(figsize=(12, 8))
            sns.heatmap(avg_activation[np.newaxis, :], cmap='magma', cbar=True, vmin=0, vmax=1)
            plt.title(f"Normalized Average Activation - Layer: {name}")
            plt.show()

            # Cumulative distribution plot
            plot_cumulative_distribution(avg_activation, name)

def plot_cumulative_distribution(activations, layer_name, task_name=None):
    """
    Plots the cumulative distribution of activations for a given layer.
    
    Args:
        activations (Tensor): Activations to plot.
        layer_name (str): Name of the layer.
        task_name (str, optional): Task name (if applicable).
    """
    # Flatten and sort activations
    activations = activations.view(-1).numpy()
    sorted_activations = np.sort(activations)
    cdf = np.cumsum(sorted_activations) / np.sum(sorted_activations)
    x = np.linspace(-1,1,100)
    y = x

    plt.figure(figsize=(8, 8))
    plt.plot(sorted_activations, cdf, x, y)

    ticks = np.arange(0, 1, 0.1)
    plt.xticks(ticks)
    plt.yticks(ticks)


    plt.xlabel('Normalized Activation Value')
    plt.ylabel('Cumulative Probability')
    title = f"Cumulative Distribution - Layer: {layer_name}"
    if task_name:
        title += f", Task: {task_name}"
    plt.title(title)
    plt.grid(True)
    plt.xlim(xmin=0,xmax=1)
    plt.ylim(ymin=0,ymax=1)
    #plt.legend()
    plt.show()