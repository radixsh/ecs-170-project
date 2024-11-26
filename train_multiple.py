import logging
import time
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

from env import DEVICE, CONFIG, HYPERPARAMETER, VALUES, NUM_DIMENSIONS
from build_model import build_model
from custom_functions import CustomLoss, make_weights_filename, DISTRIBUTIONS, get_indices
from generate_data import make_dataset, get_dataloader, MyDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def train_model(dataloader, model, loss_function, optimizer, device):
    model.train()
    size = len(dataloader.dataset)  # For debug logs
    losses = []
    actual_means = []
    actual_stddevs = []
    actual_dists = []
    pred_means = []
    pred_stddevs = []
    pred_dists = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()

        pred = model(X)             # Forward pass
        loss = loss_function(pred, y)     # Compute loss (prediction error)

        pred = (pred[0]).detach().numpy()
        label = (y[0]).detach().numpy()

        # Curr_ refers to current data point
        curr_actual_means = []
        curr_actual_stddevs = []
        curr_actual_dists = []
        curr_pred_means = []
        curr_pred_stddevs = []
        curr_pred_dists = []

        # Go through each dimension
        for dim in range(1, NUM_DIMENSIONS+1):
            # Acquire appropiate indices for features
            dim_mean_idxs = get_indices(mean=True, dims=dim)
            dim_stddev_idxs = get_indices(stddev=True,dims=dim)
            dim_dist_idxs = get_indices(dists=True, dims=dim)

            # Get the specified values; dim_ refers to current dimensions
            dim_actual_means = label[dim_mean_idxs]
            dim_actual_stddevs = label[dim_stddev_idxs]
            dim_actual_dists = np.argmax(label[dim_dist_idxs],keepdims=True)
            dim_pred_means = pred[dim_mean_idxs]
            dim_pred_stddevs = pred[dim_stddev_idxs]
            dim_pred_dists = np.argmax(pred[dim_dist_idxs],keepdims=True)

            curr_actual_means.append(dim_actual_means)
            curr_actual_stddevs.append(dim_actual_stddevs)
            curr_actual_dists.append(dim_actual_dists)
            curr_pred_means.append(dim_pred_means)                
            curr_pred_stddevs.append(dim_pred_stddevs)
            curr_pred_dists.append(dim_pred_dists)
            
            # Store them for processing
            actual_means.append(curr_actual_means)
            actual_stddevs.append(curr_actual_stddevs)
            actual_dists.append(curr_actual_dists)
            pred_means.append(curr_pred_means)
            pred_stddevs.append(curr_pred_stddevs)
            pred_dists.append(curr_pred_dists)
        
        loss_value = loss.item()
        losses.append(loss_value)

        current = batch * dataloader.batch_size + len(X)#(batch + 1) * len(X)
        logger.debug(f"Loss after batch {batch}:\t"
                     f"{loss_value:>7f}  [{current:>5d}/{size:>5d}]")
        
        optimizer.zero_grad()
        loss.backward()             # Backpropagation
        optimizer.step()
    
    # Done with the epoch
    actual_means = np.transpose(actual_means)[0]
    actual_stddevs = np.transpose(actual_stddevs)[0]
    actual_dists = np.transpose(actual_dists)[0]
    pred_means = np.transpose(pred_means)[0]
    pred_stddevs = np.transpose(pred_stddevs)[0]
    pred_dists = np.transpose(pred_dists)[0]

    mean_r2_scores = []
    stddev_r2_scores = []
    accuracy_scores = []

    for dim in range(NUM_DIMENSIONS):
        curr_mean_r2_score = r2_score(actual_means[dim], pred_means[dim])
        curr_stddev_r2_score = r2_score(actual_stddevs[dim], pred_stddevs[dim])
        curr_accuracy = accuracy_score(actual_dists[dim],pred_dists[dim])

        mean_r2_scores.append(curr_mean_r2_score)
        stddev_r2_scores.append(curr_stddev_r2_score)
        accuracy_scores.append(curr_accuracy)

    # Take means across all dimensions
    out = {
        "mean_r2_score": np.mean(mean_r2_scores),
        "stddev_r2_score": np.mean(stddev_r2_scores),
        "accuracy": np.mean(accuracy_scores),
        "loss": np.mean(losses)
    }
    return out

def pipeline(model, config):
    start = time.time()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Optimizer can't be in the config dict because it depends on model params
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config['LEARNING_RATE'], foreach=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, foreach=True)
    # optimizer = torch.optim.Adamw(model.parameters(), lr=LEARNING_RATE, foreach=True)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE, foreach=True)

    # Loss function can't be in config dict because custom_loss_function.py
    # depends on config dict to build the loss function
    loss_function = CustomLoss()

    train_dataloader = get_dataloader(config, 'TRAIN')

    logger.info(f"Training with {HYPERPARAMETER} = "
                f"{config[HYPERPARAMETER]}...")

    for epoch in range(config['EPOCHS']):
        logger.debug(f"\nEpoch {epoch + 1}\n-------------------------------")
        test_results = train_model(train_dataloader, model, loss_function, optimizer, DEVICE)
        logger.info(f"Metrics for epoch {epoch + 1}:"
                    f"\n\t-->      Loss: {test_results['loss']:.6f}"
                    f"\n\t-->   Mean R2: {test_results['mean_r2_score']:.6f}"
                    f"\n\t--> Stddev R2: {test_results['stddev_r2_score']:.6f}"
                    f"\n\t-->  Accuracy: {test_results['accuracy']:.6f}"
                    )

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
        # Update TRAINING_SIZE in the dict from env.py
        CONFIG[HYPERPARAMETER] = value

        # Filenames for various models
        dest_filename = make_weights_filename(CONFIG['TRAIN_SIZE'],
                                              CONFIG['SAMPLE_SIZE'],
                                              NUM_DIMENSIONS,
                                              CONFIG['BATCH_SIZE'],
                                              CONFIG['LEARNING_RATE'])

        # Initialize a new neural net
        input_size = CONFIG['SAMPLE_SIZE'] * NUM_DIMENSIONS
        output_size = (len(DISTRIBUTIONS) + 2) * NUM_DIMENSIONS
        model = build_model(input_size, output_size).to(DEVICE)

        # Train a new model, and save its weights out
        trainable_params = sum(param.numel() for param in model.parameters())
        formatted_config = "\n".join(f"\t{thing}: {CONFIG[thing]}" \
                for thing in CONFIG)
        logger.info(f"Training a new {NUM_DIMENSIONS}-dimensional model "
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
