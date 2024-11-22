import logging
import time
import torch
import os
from torch.utils.data import Dataset

from env import DEVICE, CONFIG, HYPERPARAMETER, VALUES, NUM_DIMENSIONS
from build_model import build_model
from custom_functions import CustomLoss, make_weights_filename, DISTRIBUTIONS
from generate_data import make_dataset, get_dataloader, MyDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# TODO: Increase batch size (I think it's currently 1)
# Batch size 1: gradient calculations might be noisy :(
# Big batch size: potentially not enough compute :(
# Medium batch size: good for implicit regularization :)
def train_model(dataloader, model, loss_function, optimizer, device):
    model.train()
    size = len(dataloader.dataset)  # For debug logs

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()

        pred = model(X)             # Forward pass
        loss = loss_function(pred, y)     # Compute loss (prediction error)
        loss.backward()             # Backpropagation
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        loss_value = loss.item()
        current = batch * dataloader.batch_size + len(X)#(batch + 1) * len(X)
        logger.debug(f"Loss after batch {batch}:\t"
                     f"{loss_value:>7f}  [{current:>5d}/{size:>5d}]")

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
        train_model(train_dataloader, model, loss_function, optimizer, DEVICE)

    end = time.time()
    logger.info(f"Finished training in {end - start:.2f} seconds")

    return model.state_dict()

# Goal: Train multiple models, and save each one
def main():
    start = time.time()

    # Make sure the 'data' directory exists
    os.makedirs('data', exist_ok=True)

    # Make sure the models directory exists
    models_directory = 'models'
    os.makedirs(models_directory, exist_ok=True)

    

    # Train one model per training_size
    for i, value in enumerate(VALUES):
        # Update TRAINING_SIZE in the dict from env.py
        CONFIG[HYPERPARAMETER] = value 

        # Filenames for various models
        dest_filename = make_weights_filename(CONFIG['TRAIN_SIZE'],
                                              CONFIG['SAMPLE_SIZE'],
                                              NUM_DIMENSIONS)

        # Initialize a new neural net
        input_size = CONFIG['SAMPLE_SIZE'] * NUM_DIMENSIONS
        output_size = (len(DISTRIBUTIONS) + 2) * NUM_DIMENSIONS
        model = build_model(input_size, output_size).to(DEVICE)

        num_trainable_params = sum(param.numel() for param in model.parameters())

        # Train the model anew, and save the resulting model's weights out
        logger.info(f"Training model with hyperparameters:\n"
                     f"TRAIN_SIZE = {CONFIG['TRAIN_SIZE']}\n"
                     f"SAMPLE_SIZE = {CONFIG['SAMPLE_SIZE']}\n"
                     f"BATCH_SIZE = {CONFIG['BATCH_SIZE']}\n"
                     f"EPOCHS = {CONFIG['EPOCHS']}\n"
                     f"NUM_DIMENSIONS = {NUM_DIMENSIONS}\n"
                     f"Variable hyperparameter is {HYPERPARAMETER}.\n"
                     f"Model has {num_trainable_params} trainable parameters.")
        train_start = time.time()

        model_weights = pipeline(model, CONFIG)
        if model_weights is None:
            logger.info("pipeline() did not return anything!!! mehhhhhh!!!!!")
        torch.save(model_weights, dest_filename)

        train_end = time.time()
        logger.debug(f"Wrote out weights to {dest_filename} "
                     f"(finished in {train_end - train_start:.2f} seconds)")

    end = time.time()
    #logger.info(f"Finished training {dests} models in {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
