import time
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from custom_loss_function import CustomLoss
from build_model import build_model
from generate_data import generate_data, make_dataset
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

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * dataloader.batch_size + len(X)#(batch + 1) * len(X)
            logger.debug(f"Loss after batch {batch}:\t"
                         f"{loss_value:>7f}  [{current:>5d}/{size:>5d}]")

def test_model(dataloader, model, loss_function, device):
    model.eval()

    test_loss = 0
    guesses = []
    actuals = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).float()

            pred = model(X)
            # logger.debug(f"pred: \t{pred}")
            # logger.debug(f"y: \t\t{y}")

            # TODO: Implement metrics with guesses and actuals?

            test_loss += loss_function(pred, y).item()
            # logger.debug(f"loss: \t{loss_function(pred,y)}")

    test_loss /= len(dataloader)
    return test_loss

def get_dataloader(config, filename=None, require_match=False):
    try:
        dataset = torch.load(filename)
        is_acceptable_size = (len(dataset) == config['TRAINING_SIZE'] \
                or not require_match)

        if isinstance(dataset, Dataset) and is_acceptable_size:
            logger.info(f"Using dataset from {filename} (size: {len(dataset)})")
        else:
            raise Exception(f"Incorrect dataset size: {len(dataset)}")

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

    # Consistent initialization
    torch.manual_seed(42)
    np.random.seed(42)
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

    for r in range(config['RUNS']):
        run_start = time.time()

        train_dataloader = get_dataloader(config, 'data/train_dataset',
                                          require_match=True)
        test_dataloader = get_dataloader(config, 'data/test_dataset',
                                         require_match=True)

        logger.info(f"Training with {HYPERPARAMETER} = "
                    f"{config[HYPERPARAMETER]}...")
        for epoch in range(config['EPOCHS']):
            logger.debug(f"\nEpoch {epoch + 1}\n-------------------------------")
            train_model(train_dataloader, model, loss_function, optimizer, DEVICE)
            test_model(test_dataloader, model, loss_function, DEVICE)

        run_end = time.time()
        logger.debug(f"Finished run {r + 1} of {config['RUNS']} in " +
                     f"{run_end - run_start:.2f} seconds")

    end = time.time()
    logger.info(f"Finished training pipeline in {end - start:.2f} seconds")

    return model.state_dict()

    '''
        # Test the model on the test data
        raw_test_data = generate_data(count=TEST_SIZE)
        test_samples = np.array([elem[0] for elem in raw_test_data])
        test_labels = np.array([elem[1] for elem in raw_test_data])
        test_dataset = MyDataset(test_samples, test_labels)
        test_dataloader = DataLoader(test_dataset)
        loss = test_model(test_dataloader, model, loss_function, DEVICE)
        logger.info(f"Avg loss (testing): {loss}")

        run_end = time.time()
        logger.info(f"Finished run {run + 1} of {RUNS} in " +
                     f"{run_end - run_start:.2f} seconds")

    end = time.time()
    logger.info(f"Finished {RUNS} runs in {end - start:.2f} seconds")

    torch.save(model.state_dict(), 'model_weights.pth')

    # Performance metrics: accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(test_labels, test_samples)
    precision = precision_score(test_labels, test_samples)
    recall = recall_score(test_labels, test_samples)
    f1 = f1_score(test_labels, test_samples)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    '''
