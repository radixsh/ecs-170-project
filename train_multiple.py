import logging
import time
import torch
import os

from model import MultiTaskModel
from distributions import NUM_DISTS
from data_handling import make_weights_filename
from core import run_model
from env import CONFIG, MODEL_ARCHITECTURE, HYPERPARAMETER, VALUES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def main():
    """
    Trains and saves a model for each hyperparameter value in VALUES. Fully controlled
    by env.py. HYPERPARAMETER controls which hyperparameter of the model is varied.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=os.path.join("logs", "Training.log"),
        level=logging.INFO,
        format="%(message)s",
    )
    if len(VALUES) > 1:
        logger.debug(f"Variable hyperparameter is {HYPERPARAMETER}.")

    start = time.time()

    # Update CONFIG the the value, instantiate and train a model. Imports are an issue,
    # this method ensures that only files with a main are allowed to import env.
    for value in VALUES:
        CONFIG[HYPERPARAMETER] = value

        model = MultiTaskModel(CONFIG, MODEL_ARCHITECTURE, NUM_DISTS).to(
            CONFIG["DEVICE"]
        )

        trainable_params = sum(param.numel() for param in model.parameters())
        formatted_cfg = "\n".join(f"\t{val}: {CONFIG[val]}" for val in CONFIG)
        logger.info(
            f"Training a {CONFIG['NUM_DIMENSIONS']}-dimensional "
            f"model with hyperparameters:\n{formatted_cfg}\n"
            f"Model has {trainable_params} trainable parameters."
        )

        train_start = time.time()

        output = run_model(model, CONFIG, mode="TRAIN")

        torch.save(output["weights"], make_weights_filename(CONFIG, MODEL_ARCHITECTURE))
        logger.info(f"Finished in {time.time() - train_start:.3f} seconds")

    logger.debug(
        f"Finished training {len(VALUES)} models in "
        f"{time.time() - start:.3f} seconds"
    )


if __name__ == "__main__":
    main()
