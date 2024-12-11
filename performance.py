import logging
import time
import torch
import os
from collections import defaultdict

from model import MultiTaskModel
from visualizations import regression_png, classification_png, visualize_weights
from data_handling import make_weights_filename
from distributions import NUM_DISTS
from core import run_model
from env import CONFIG, MODEL_ARCHITECTURE, HYPERPARAMETER, VALUES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def main():
    """
    Tests a series of models determined by VALUES and HYPERPARAMETER in env.py.
    Generates and displays performance metrics for each, including graphs which
    compare how changing the specified hyperparameter affects performance.
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        filename=os.path.join("logs", f"Testing.log"),
        level=logging.INFO,
        format="%(message)s",
    )

    metrics = defaultdict(list)
    hyperparams = []

    start = time.time()
    # Update CONFIG the the value, instantiate and train a model. Imports are an issue,
    # this method ensures that only files with a main are allowed to import env.
    for hyperparam_val in VALUES:
        CONFIG[HYPERPARAMETER] = hyperparam_val

        model_filename = make_weights_filename(CONFIG, MODEL_ARCHITECTURE)
        logger.debug(f"Analyzing model at {model_filename}...")
        logger.debug(MODEL_ARCHITECTURE)
        state_dict = torch.load(model_filename)

        model_start = time.time()
        hyperparams.append(hyperparam_val)

        model = MultiTaskModel(CONFIG, MODEL_ARCHITECTURE, NUM_DISTS).to(
            CONFIG["DEVICE"]
        )
        model.load_state_dict(state_dict)

        if len(VALUES) > 1:
            logger.debug(
                f"-------------------------------------"
                f" {HYPERPARAMETER} = {hyperparam_val} "
                f"-------------------------------------"
            )

        output = run_model(model, CONFIG, mode="TEST")

        for key, value in output["metrics"].items():
            metrics[key].append(value)

        logger.debug(f"(Finished in {time.time() - model_start:.3f} seconds)")

    logger.debug(f"Analyzed {len(VALUES)} models in {time.time() - start:.3f} seconds")

    visualize_weights(model)

    if len(VALUES) <= 1:
        return

    regression_metrics = ["r2", "mae", "rmse"]
    for metric_name in regression_metrics:
        regression_png(
            metric_name.upper(),
            hyperparams,
            metrics[f"mean_{metric_name}"],
            metrics[f"stddev_{metric_name}"],
            HYPERPARAMETER,
        )

    classification_metrics = ["accuracy", "f1", "precision", "recall"]
    for metric_name in classification_metrics:
        classification_png(
            metric_name.upper(), hyperparams, metrics[metric_name], HYPERPARAMETER
        )


if __name__ == "__main__":
    main()
