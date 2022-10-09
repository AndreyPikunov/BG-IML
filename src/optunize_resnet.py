import argparse
import logging
from copy import deepcopy

import numpy as np
import mlflow
import optuna

import sys

sys.path.append("../src")
from cv_resnet import main as run_experiment
from utils import load_config, get_abs_dirname


class Objective:
    def __init__(self, config):
        self.config = deepcopy(config)

    def __call__(self, trial):

        config = self.config

        base_model_list = config["optunize_resnet"]["base_model_list"]
        base_model = trial.suggest_categorical("base_model", base_model_list)
        config["cv_resnet"]["model_name"] = base_model

        retrain_type_list = config["optunize_resnet"]["retrain_type_list"]
        retrain_type = trial.suggest_categorical("retrain_type", retrain_type_list)
        config["cv_resnet"]["retrain_type"] = retrain_type

        lr_min = config["optunize_resnet"]["lr"]["min"]
        lr_max = config["optunize_resnet"]["lr"]["max"]
        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        config["cv_resnet"]["optimizer"]["lr"] = lr

        score_best = run_experiment(config)
        return score_best


def main(config):
    objective = Objective(config)
    study = optuna.create_study(direction="maximize")
    n_trials = config["optunize_resnet"]["n_trials"]
    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    parser.add_argument(
        "--mlflow-experiment", type=str, default="optuna-cv-resnet"
    )  # useful for debugging
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

    mlflow_experiment = args.mlflow_experiment
    mlflow.set_experiment(mlflow_experiment)

    main(config)
