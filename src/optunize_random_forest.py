import os
import argparse
import logging
from copy import deepcopy

import pandas as pd
import numpy as np
import mlflow
import optuna

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from sklearn.metrics import top_k_accuracy_score

from joblib import dump

import sys
sys.path.append("../src")
from utils import load_config, get_abs_dirname


class Objective:
    def __init__(self, config):
        self.config = deepcopy(config)

    def __call__(self, trial):

        config = self.config

        filename_input = os.path.join(
            config["shared"]["project_root"],
            config["optunize_random_forest"]["filename_input"],
        )

        df = pd.read_csv(filename_input)

        columns_features = [c for c in df.columns if c.startswith("emb")]

        X = df[columns_features].copy()
        y = df["label_code"]

        column_fold = config["optunize_random_forest"]["column_fold"]
        folds_use = config["optunize_random_forest"].get("folds_use")
        if folds_use is None:
            folds_use = df[column_fold].unique()

        params = self.suggest_forest_params(trial)
        
        params["class_weight"] = "balanced"

        criterion = trial.suggest_categorical("criterion", ["gini", "log_loss"])
        params["criterion"] = criterion

        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        params["bootstrap"] = bootstrap

        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        params["max_features"] = max_features

        # use_pca = trial.suggest_categorical("use_pca", [True, False])
        # steps = [StandardScaler()]
        # if use_pca:
        #     steps.append(PCA())
        # preprocessor = make_pipeline(*steps)

        fold_scores = []

        for fold_index in folds_use:

            if fold_index == "val":
                continue

            mlflow.start_run()
            mlflow.log_dict(config, "config-runtime.yaml")
            mlflow.log_param("fold_index", fold_index)

            fold_index_str = str(fold_index)

            mask_test = df.fold_author == fold_index_str
            mask_val = df.fold_author == "val"
            mask_train = ~(mask_test | mask_val)

            X_train, X_test = X[mask_train], X[mask_test]
            y_train, y_test = y[mask_train], y[mask_test]

            clf = RandomForestClassifier(**params)
            mlflow.log_params(params)

            # mlflow.log_param("use_pca", use_pca)

            # X_train = preprocessor.fit_transform(X_train)
            # X_test = preprocessor.transform(X_test)

            clf.fit(X_train, y_train)

            folder_output = config["optunize_random_forest"]["folder_output"]
            os.makedirs(folder_output, exist_ok=True)

            filename_save = os.path.join(folder_output, "model.joblib")
            dump(clf, filename_save)

            mlflow.log_artifact(filename_save)

            for name, X_name, y_name in (
                ("train", X_train, y_train),
                ("test", X_test, y_test),
            ):

                proba_pred = clf.predict_proba(X_name)

                label_weight = 1 / np.unique(y_name, return_counts=True)[-1]
                weight = [label_weight[i] for i in y_name]

                score = top_k_accuracy_score(y_name, proba_pred, k=2, sample_weight=weight)

                if name == "test":
                    fold_scores.append(score)

                mlflow.log_metric(f"score_{name}", score)

            mlflow.end_run()

        return np.mean(fold_scores)

    def suggest_forest_params(self, trial):

        config = self.config

        random_state = config["shared"]["seed"]
        params = {"random_state": random_state}

        hps = config["optunize_random_forest"]["hyperparams"]

        for name, bounds in hps.items():
            p = trial.suggest_int(
                name,
                bounds["min"],
                bounds["max"],
                log=True
            )
            params[name] = p

        return params


def main(config):
    objective = Objective(config)
    study = optuna.create_study(direction="maximize")
    n_trials = config["optunize_random_forest"]["n_trials"]
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

    mlflow_experiment = config["optunize_random_forest"].get("mlflow_experiment", "optunize_random_forest")
    mlflow.set_experiment(mlflow_experiment)

    main(config)
