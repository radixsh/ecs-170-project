import time
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from custom_functions import CustomLoss
from generate_data import make_dataset
from env import DEVICE, HYPERPARAMETER

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

def get_dataloader(config, filename=None, required_size=-1):
    try:
        dataset = torch.load(filename)

        # If required_size is passed in, then adhere to it. Otherwise,
        # required_size is default value, indicating dataset can have any size
        acceptable_count = (len(dataset) == required_size \
                            or required_size == -1)

        dataset_sample_size = len(dataset.__getitem__(0)[0])
        acceptable_sample_size = (config['SAMPLE_SIZE'] == dataset_sample_size)

        if not isinstance(dataset, Dataset):
            raise Exception(f'Could not read dataset from {filename}')
        if not acceptable_count:
            raise Exception(f"Incorrect dataset size: {len(dataset)}")
        if not acceptable_sample_size:
            raise Exception(f"Incorrect sample size: model expects "
                            f"{config['SAMPLE_SIZE']} points as input, but "
                            f"this dataset would give {dataset_sample_size}")

        logger.info(f"Using dataset from {filename} (size: {len(dataset)})")

    except Exception as e:
        logger.info(e)
        logger.info(f'Generating fresh data...')
        dataset = make_dataset(filename)

    # If no filename is passed in, the file does not exist, or the file's
    # contents do not represent a DataLoader as expected, then generate some
    # new data, and write it out to the given filename
    dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'])
    return dataloader

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

    train_dataloader = get_dataloader(config, 'data/train_dataset',
                                      required_size=config['TRAINING_SIZE'])

    logger.info(f"Training with {HYPERPARAMETER} = "
                f"{config[HYPERPARAMETER]}...")

    for epoch in range(config['EPOCHS']):
        logger.debug(f"\nEpoch {epoch + 1}\n-------------------------------")
        train_model(train_dataloader, model, loss_function, optimizer, DEVICE)

    end = time.time()
    logger.info(f"Finished training in {end - start:.2f} seconds")

    return model.state_dict()
