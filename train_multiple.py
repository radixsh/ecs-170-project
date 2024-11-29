import logging
import time
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR
from collections import defaultdict
import warnings

from env import *
from custom_functions import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

warnings.simplefilter('ignore', UserWarning) # Silence bugged pytorch lr scheduler warning

def train_model(dataloader, model, loss_function, optimizer, device):
    model.train()
    size = len(dataloader.dataset)  # For debug logs

    # Keep these separate for now
    losses = []
    metrics = defaultdict(list)

    for X, y in dataloader:
        X, y = X.to(device).float(), y.to(device).float()

        pred = model(X)                     # Forward pass
        loss = loss_function(pred, y)       # Compute loss (prediction error)
        losses.append(loss.item())     # Save for metrics

        optimizer.zero_grad()
        loss.backward()                     # Backpropagation
        optimizer.step()

        ## Validation
        # Use 10% of the data to generate validation metrics
        # Doesn't add much training time since we're not calling .backward()
        validation_size = X.size(0) // 10
        
        with torch.no_grad(): # Already trained on these
            # Data is randomized, so we can just take the first slice
            X_subset = X[:validation_size]
            y_subset = y[:validation_size]
            pred_subset = model(X_subset)
            batch_metrics = calculate_metrics(pred_subset, y_subset, CONFIG['NUM_DIMENSIONS'], mode='TRAIN')

            # Aggregate batch metrics
            # This is why we kept losses separate
            for key, value in batch_metrics.items():
                metrics[key].append(value)
            
    # Average metrics over all batches
    # Remember to add in the loss
    metrics['loss'] = losses
    metrics = {key: np.mean(value) for key, value in metrics.items()}

    return metrics

def pipeline(model, config):
    start = time.time()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Optimizer can't be in the config dict because it depends on model params
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=config['LEARNING_RATE'], 
                                foreach=True,
                                #momentum=0.5
                                )
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, foreach=True)
    # optimizer = torch.optim.Adamw(model.parameters(), lr=LEARNING_RATE, foreach=True)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE, foreach=True)

    
    # Learning rate scheduler: linearly increase for first 5 epochs, then slowly decay
    warmup_epochs = 5
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(CONFIG['EPOCHS'] - warmup_epochs), eta_min=1e-4)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Loss function can't be in config dict because custom_loss_function.py
    # depends on config dict to build the loss function
    loss_function = CustomLoss(num_dimensions=CONFIG['NUM_DIMENSIONS'])

    train_dataloader = get_dataloader(config, 'TRAIN')

    logger.info(f"Training with {HYPERPARAMETER} = "
                f"{config[HYPERPARAMETER]}...")

    for epoch in range(config['EPOCHS']):
        logger.info(f"\nEpoch {epoch + 1}\n-------------------------------")
        test_results = train_model(train_dataloader, model, loss_function, optimizer, DEVICE)
        logger.info(f"Learning rate = {optimizer.param_groups[0]['lr']:.6f}"
                    f"\nMetrics:"
                    f"\n\t-->      Loss: {test_results['loss']:.6f}"
                    f"\n\t-->   Mean R2: {test_results['mean_r2']:.6f}"
                    f"\n\t--> Stddev R2: {test_results['stddev_r2']:.6f}"
                    f"\n\t-->  Accuracy: {test_results['accuracy']:.6f}"
                    )
        scheduler.step()

    end = time.time()
    logger.info(f"Finished training in {end - start:.3f} seconds")

    return model.state_dict()

# Goal: Train multiple models, and save each one
def main():
    start = time.time()

    # Make sure the 'data' directory exists
    os.makedirs('data', exist_ok=True)

    # Make sure the 'models' directory exists
    models_directory = 'models'
    os.makedirs(models_directory, exist_ok=True)

    # Train one model per training_size
    for i, value in enumerate(VALUES):
        # Update hyperparam in the dict from env.py
        CONFIG[HYPERPARAMETER] = value

        # Filenames for various models
        dest_filename = make_weights_filename(CONFIG)

        # Initialize a new neural net
        model = MultiTaskModel(CONFIG, MODEL_ARCHITECTURE, len(DISTRIBUTIONS)).to(DEVICE)

        # Train a new model, and save its weights out
        trainable_params = sum(param.numel() for param in model.parameters())
        formatted_config = "\n".join(f"\t{thing}: {CONFIG[thing]}" \
                for thing in CONFIG)
        logger.info(f"Training a new {CONFIG['NUM_DIMENSIONS']}-dimensional model "
                    f"with these hyperparameters:\n{formatted_config}\n"
                     f"Variable hyperparameter is {HYPERPARAMETER}. "
                     f"Model has {trainable_params} trainable parameters.")
        train_start = time.time()

        model_weights = pipeline(model, CONFIG)
        torch.save(model_weights, dest_filename)

        train_end = time.time()
        logger.debug(f"Wrote out weights to {dest_filename} "
                     f"(finished in {train_end - train_start:.3f} seconds)")

    end = time.time()
    logger.info(f"Finished training {len(VALUES)} models in "
                f"{end - start:.3f} seconds")

if __name__ == "__main__":
    main()
