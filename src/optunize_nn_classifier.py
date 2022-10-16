import argparse
import logging
from copy import deepcopy

import numpy as np
import mlflow
import optuna

from train_nn_classifier import main as run_experiment
from utils import load_config, get_abs_dirname


class Objective:
    def __init__(self, config):
        self.config = deepcopy(config)

    def __call__(self, trial):

        config = self.config

        resnet_name = trial.suggest_categorical(
            "resnet_name", config["optunize_nn_classifier"]["resnet_name_list"]
        )
        config["train_nn_classifier"]["resnet_name"] = resnet_name

        freeze_resnet_cnn = trial.suggest_categorical(
            "freeze_resnet_cnn",
            config["optunize_nn_classifier"]["freeze_resnet_cnn_list"],
        )
        config["train_nn_classifier"]["params_embedder"][
            "freeze_resnet_cnn"
        ] = freeze_resnet_cnn

        if freeze_resnet_cnn:
            freeze_resnet_fc = trial.suggest_categorical(
                "freeze_resnet_fc",
                config["optunize_nn_classifier"]["freeze_resnet_fc_list"],
            )
        else:
            freeze_resnet_fc = True
        config["train_nn_classifier"]["params_embedder"][
            "freeze_resnet_fc"
        ] = freeze_resnet_fc

        embedding_size = trial.suggest_int(
            "embedding_size",
            config["optunize_nn_classifier"]["embedding_size"]["min"],
            config["optunize_nn_classifier"]["embedding_size"]["max"],
            log=True,
        )
        config["train_nn_classifier"]["params_embedder"][
            "embedding_size"
        ] = embedding_size

        lr = trial.suggest_float(
            "lr",
            config["optunize_nn_classifier"]["lr"]["min"],
            config["optunize_nn_classifier"]["lr"]["max"],
            log=True,
        )
        config["train_nn_classifier"]["optimizer"]["lr"] = lr

        score_best = run_experiment(config)
        return score_best


def main(config):
    objective = Objective(config)
    study = optuna.create_study(direction="maximize")
    n_trials = config["optunize_nn_classifier"]["n_trials"]
    study.optimize(objective, n_trials=n_trials)


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

    mlflow_experiment = config["train_nn_classifier"].get(
        "mlflow_experiment", "train_nn_classifier"
    )
    mlflow.set_experiment(mlflow_experiment)

    main(config)
