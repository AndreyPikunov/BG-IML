import argparse
import logging

import numpy as np
import mlflow

import sys

sys.path.append("../src")
from metric_learning_resnet import main as run_experiment
from utils import load_config, get_abs_dirname


def main(config):

    embedding_size_list = config["embedding_size_search"]["embedding_size_list"]
    model_name = config["embedding_size_search"]["model_name"]
    retrain_type = config["embedding_size_search"]["retrain_type"]

    for embedding_size in embedding_size_list:

        config["metric_learning_resnet"]["embedding_size"] = embedding_size
        config["metric_learning_resnet"]["model_name"] = model_name
        config["metric_learning_resnet"]["retrain_type"] = retrain_type

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

    mlflow_experiment = config["embedding_size_search"].get(
        "mlflow_experiment", "embedding_size_search"
    )
    mlflow.set_experiment(mlflow_experiment)

    main(config)
