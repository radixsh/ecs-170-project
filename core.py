import warnings
from collections import defaultdict, deque

import torch
from torch.optim import Adam, Adamax, AdamW, NAdam
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
import numpy as np

from data_handling import get_dataset, logger, make_weights_filename
from distributions import NUM_DISTS
from metrics import calculate_metrics, display_metrics
from model import CustomLoss
from visualizations import visualize_activations_avg

### Illegal imports: env, generate_data, sanity_check, performance, train_multiple


def run_model(model, config, mode):
    """
    Main call to train or test the model. If in training mode, this sets up an
    optimizer, creates a loss function object, then loops through epochs and adjusts
    model weights, displaying validation metrics after each epoch. If in testing
    mode, just go through one epoch and print some more detailed metrics.

    Args:
        config (dict): In the format of CONFIG in ENV, how the model is set up.
        mode (str): "TRAIN" or "TEST", controls what data the model looks for,
            whether the model updates its weights, and what metrics it ouputs.

    Returns:
        dict: A dict containing the model weights and performance metrics.
    """
    dataset = get_dataset(config, mode)

    if mode == "TRAIN":
        # Make an additional dataset here for validation.
        val_dataset = get_dataset(config, 'TEST')
        model.train()
        torch.set_grad_enabled(True)
        # Use one of the following optimizers: Adam, AdamW, Adamax, NAdam, RMSprop.
        # These use second-order gradient; first-order optimizers like SGD never 
        # overcome an initial local minima in the gradient.
        optimizer = Adamax(
            model.parameters(),
            lr=config["LEARNING_RATE"],
        )

        # Learning rate scheduler. The rate starts slow to prevent erratic jumps,
        # peaks at 5 epochs, then gradually decreases to flatten out at the last epoch.
        epochs, warmup_len = config["EPOCHS"], 5
        warmup_lr = LambdaLR(
            optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_len
        )
        decay_lr = CosineAnnealingLR(
            optimizer, T_max=(config["EPOCHS"] - warmup_len), eta_min=1e-4
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_lr, decay_lr], milestones=[warmup_len]
        )

        dataloader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config["TRAIN_SIZE"], shuffle=True)
        warnings.simplefilter("ignore", UserWarning)

    # Only one epoch and only one batch for testing.
    elif mode == "TEST":
        model.eval()
        torch.set_grad_enabled(False)
        epochs = 1
        dataloader = DataLoader(
            dataset,
            batch_size=config["TRAIN_SIZE"],
            shuffle=True,
        )

    loss_function = CustomLoss(num_dims=config["NUM_DIMENSIONS"], num_dists=NUM_DISTS)

    # Need to track the best loss to output the best model. 
    best_loss = float('inf')
    best_weights = None # Unclear what type this should be.
    best_metrics = defaultdict(list)

    # Stop the model early if the losses if last PATIENCE epochs are less than EPSILON apart.
    counter = 0
    PATIENCE = 5


    # Validation metrics are calculated and displayed at the end of every epoch
    # in training mode. In testing mode, this just runs once.
    for epoch in range(epochs):

        epoch_losses = []

        for X, y in dataloader:
            X, y = X.to(config["DEVICE"]).float(), y.to(config["DEVICE"]).float()
            pred = model(X)
            loss = loss_function(pred, y)
            epoch_losses.append(loss.item())

            if mode == "TRAIN":
                # Gradient descent happens here
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if mode == "TEST":
                # Only one batch during testing, so we're basically done.
                best_metrics = calculate_metrics(
                    pred, y, config["NUM_DIMENSIONS"], mode=mode
                )

        # Training loop done, average the loss across all batches
        epoch_loss = np.mean(epoch_losses)

        if mode == "TEST":
            # We've only seen one epoch and one batch, so everything is 'best'
            visualize_activations_avg(model, dataloader)
            best_weights = model.state_dict()
            best_metrics['loss'] = epoch_loss
            best_metrics = display_metrics(best_metrics, mode, epoch + 1)

        if mode == "TRAIN":
            scheduler.step()  # This is being called correctly but still throws warnings
            # Validation happens here.
            val_metrics = defaultdict(list)
            with torch.no_grad():
                for X, y in val_dataloader:
                    X, y = X.to(config["DEVICE"]).float(), y.to(config["DEVICE"]).float()
                    pred = model(X)
                    val_loss = loss_function(pred, y).item()
                    val_metrics = calculate_metrics(
                        pred, y, config["NUM_DIMENSIONS"], mode=mode
                    )
                    val_metrics['loss'] = val_loss

            val_metrics = display_metrics(val_metrics, mode, epoch + 1)

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_weights = model.state_dict()
                best_metrics = val_metrics
                counter = 0
            else:
                counter += 1

            if counter >= PATIENCE:
                filename = make_weights_filename(config, model.architecture)
                logger.info(f"Model stopped training early at epoch {epoch + 1}, "
                            f"saved to {filename}")
                break

    warnings.simplefilter("once", UserWarning)
    return {"weights": best_weights, "metrics": best_metrics}
