import warnings
from collections import defaultdict

import torch
from torch.optim import Adam, Adamax, AdamW, NAdam
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from data_handling import get_dataset
from metrics import calculate_metrics, display_metrics
from model import CustomLoss

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
        model.train()
        torch.set_grad_enabled(True)
        # Use one of the following optimizers: Adam, AdamW, Adamax, NAdam, RMSprop.
        # These use second-order gradient; first-order optimizers like SGD are
        # extremely lethargic for this task and barely do better than chance.
        optimizer = Adam(
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

    loss_function = CustomLoss(num_dims=config["NUM_DIMENSIONS"])

    # Metrics are calculated and displayed at the end of every epoch in training mode.
    # 10% of the data in every epoch is used for to calculate performance. Questionable
    # whether this can be called validation. In testing mode, this just runs once.
    for epoch in range(epochs):

        metrics = defaultdict(list)

        for X, y in dataloader:
            X, y = X.to(config["DEVICE"]).float(), y.to(config["DEVICE"]).float()
            pred = model(X)
            loss = loss_function(pred, y)
            metrics["loss"].append(loss.item())

            if mode == "TRAIN":
                with torch.no_grad():
                    pred = model(X[: config["BATCH_SIZE"] // 10])
                    y = y[: config["BATCH_SIZE"] // 10]

                # Gradient descent happens here
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            curr_metrics = calculate_metrics(
                pred, y, config["NUM_DIMENSIONS"], mode=mode
            )

            for key, value in curr_metrics.items():
                metrics[key].append(value)

        metrics = display_metrics(metrics, mode, epoch + 1)
        if mode == "TRAIN":
            scheduler.step()  # This is being called correctly but still throws warnings
    warnings.simplefilter("once", UserWarning)
    return {"weights": model.state_dict(), "metrics": metrics}
