import argparse
import logging

import numpy as np
import pandas as pd
import os
import torch
import mlflow

from tqdm import tqdm

from datasets import PaintingDataset
from utils import load_config, get_abs_dirname, load_resnet
from utils.mlflow import get_artifact_storage, download_artifact_yaml
from models import NNClassifier


def predict(model, dataset, device):

    Y_true = []
    Y_pred = []
    Z = []

    with torch.no_grad():

        for i in tqdm(range(len(dataset))):

            x, y = dataset[i]

            x = x.unsqueeze(0).to(device)
            true = y.unsqueeze(0)
            pred, z = model(x)

            Y_pred.append(pred)
            Y_true.append(true)
            Z.append(z)

    Z = torch.concat(Z).cpu()
    Y_true = torch.concat(Y_true).cpu()
    Y_pred = torch.concat(Y_pred).cpu()

    return Y_pred, Y_true, Z


def main(config):

    artifact_storage = get_artifact_storage(
        config["shared"]["mlflow_tracking_uri"],
        config["predict_nn_classifier"]["mlflow"]["experiment_name"],
        config["predict_nn_classifier"]["mlflow"]["run_id"],
    )

    artifact_uri = os.path.join(artifact_storage, "config-runtime.yaml")
    config_restored = download_artifact_yaml(artifact_uri)

    resnet_name = config_restored["train_nn_classifier"]["params_embedder"][
        "resnet_name"
    ]
    _, resnet_weights = load_resnet(resnet_name)
    transform_resnet = resnet_weights.DEFAULT.transforms()

    filename_ann = os.path.join(
        config_restored["shared"]["project_root"],
        config_restored["train_nn_classifier"]["filename_design"],
    )
    ann = pd.read_csv(filename_ann)

    folder_images = os.path.join(
        config_restored["shared"]["project_root"],
        config_restored["train_nn_classifier"]["folder_images"],
    )

    dataset = PaintingDataset(ann, folder_images, transform_preprocess=transform_resnet)

    artifact_uri = os.path.join(artifact_storage, "code2label.yml")
    code2label = pd.Series(download_artifact_yaml(artifact_uri))

    artifact_uri = os.path.join(artifact_storage, "model.st")
    filename_model_st = mlflow.artifacts.download_artifacts(artifact_uri)

    n_classes = len(code2label)

    model = NNClassifier(
        n_classes, config_restored["train_nn_classifier"]["params_embedder"]
    )
    model.load_state_dict(torch.load(filename_model_st))
    model.eval()

    device = torch.device(config["predict_nn_classifier"]["device"])
    model = model.to(device)

    Y_pred, Y_true, Z = predict(model, dataset, device)

    features = pd.DataFrame(Z, columns=[f"emb_{i:02d}" for i in range(Z.shape[1])])

    df = pd.concat([features, ann], axis=1)

    proba_pred = torch.nn.functional.softmax(Y_pred, dim=1)
    proba_true = Y_true

    for i, label in code2label.iteritems():
        key = f"proba_pred_{label}"
        df[key] = proba_pred[:, i]
        key = f"proba_true_{label}"
        df[key] = proba_true[:, i]

    df["label_code_pred"] = Y_pred.argmax(dim=1)
    df["label_code_true"] = Y_true.argmax(dim=1)

    df["label_pred"] = df["label_code_pred"].replace(code2label)
    df["label_true"] = df["label_code_true"].replace(code2label)

    folder_output = os.path.join(
        config["predict_nn_classifier"]["folder_output"],
        config["predict_nn_classifier"]["mlflow"]["experiment_name"],
        config["predict_nn_classifier"]["mlflow"]["run_id"],
    )
    os.makedirs(folder_output, exist_ok=True)

    filename_output = os.path.join(folder_output, "predictions.csv")

    df.to_csv(filename_output, index=False)


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

    main(config)
