import logging
import argparse
from itertools import chain

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

import mlflow

import sys

sys.path.append("../src")

from PaintingDataset import PaintingDataset
from Scorer import Scorer
from utils import load_config, get_abs_dirname
from utils.training import train


def create_model(n_classes):

    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False

    resnet_fc = resnet.fc
    resnet_fc.requires_grad = True

    logging.info("reset resnet fc")
    for name, layer in resnet_fc.named_modules():
        # https://discuss.pytorch.org/t/how-to-reset-parameters-of-layer/120782/2

        logging.info(name)
        if hasattr(layer, "reset_parameters"):
            logging.info(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()

    head = nn.Linear(resnet_fc.out_features, n_classes)

    model = nn.Sequential(resnet, head)

    parameters = chain(resnet_fc.parameters(), head.parameters())

    return model, parameters


def main(config):

    mps_is_ok = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    device = torch.device("mps" if mps_is_ok else "cpu")
    logging.info(f"device: {device}")

    filename_design = config["cv_resnet18"]["filename_design"]
    ann = pd.read_csv(filename_design)

    code2label = (
        ann[["label", "label_code"]].drop_duplicates().set_index("label_code")["label"]
    )
    code2label.sort_index(inplace=True)

    n_classes = len(ann.label.unique())
    logging.info(f"n_classes: {n_classes}")

    transform_resnet = ResNet18_Weights.DEFAULT.transforms()
    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0
            ),
            transforms.RandomErasing(p=0.5),
            transform_resnet,
        ]
    )

    column_fold = config["cv_resnet18"]["column_fold"]

    folds_use = config["cv_resnet18"].get("folds_use")
    if folds_use is None:
        folds_use = ann[column_fold].unique()

    batch_size = config["cv_resnet18"]["batch_size"]
    folder_images = config["cv_resnet18"]["folder_images"]

    for fold_index in folds_use:

        if fold_index == "val":
            continue

        mlflow.start_run()

        mlflow.log_param("fold_index", fold_index)

        fold_index_str = str(fold_index)

        mask_test = ann.fold_author == fold_index_str
        mask_val = ann.fold_author == "val"
        mask_train = ~(mask_test | mask_val)

        dataset_train = PaintingDataset(
            ann[mask_train], folder_images, transform_train=transform_train
        )

        dataset_test = PaintingDataset(
            ann[mask_test], folder_images, transform_preprocess=transform_resnet
        )

        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        model, parameters = create_model(n_classes)
        model = model.to(device)

        label_weight = 1 / dataset_train.ann.label_code.value_counts().sort_index()
        mlflow.log_dict(label_weight.to_dict(), "label_weight.yml")

        label_smoothing = 0.1
        criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=torch.tensor(label_weight).float().to(device),
        )

        lr = config["cv_resnet18"]["optimizer"]["lr"]
        optimizer_name = config["cv_resnet18"]["optimizer"]["name"]

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(parameters, lr=lr)
        else:
            raise NotImplementedError()

        mlflow.log_params({"optimizer": optimizer_name, "lr": lr})

        scorer = Scorer(code2label=code2label)

        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scorer=scorer,
            train_dataloader=dataloader_train,
            test_dataloader=dataloader_test,
            params=config["cv_resnet18"],
            device=device,
        )

        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    parser.add_argument(
        "--mlflow-experiment", type=str, default="cv_resnet18"
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
