import argparse
import logging

import numpy as np
import mlflow

import sys

sys.path.append("../src")
from train_combo import main as run_experiment
from utils import load_config, get_abs_dirname


def main(config):

    classification_weight_list = config["classification_weight_search"]["classification_weight_list"]

    for weight in classification_weight_list:

        config["train_combo"]["classification_weight"] = weight

        run_experiment(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()

    filename_config = args.config

    numeric_level = getattr(logging, args.logging_level.upper(), None)
    logging.basicConfig(level=numeric_level)

    config = load_config(filename_config)

    project_root = get_abs_dirname(filename_config)
    config["shared"]["project_root"] = project_root

    np.random.seed(config["shared"]["seed"])

    mlflow_tracking_uri = config["shared"]["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow_experiment = config["classification_weight_search"]["mlflow_experiment"]
    mlflow.set_experiment(mlflow_experiment)

    main(config)
