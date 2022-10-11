import os
import logging
import argparse
from itertools import chain

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms

import mlflow

import sys

sys.path.append("../src")

from datasets.PaintingDataset import PaintingDatasetTriplet as PaintingDataset
from trainers import TrainerTriplet as Trainer
from scorers import ScorerClustering as Scorer
from utils import load_config, get_abs_dirname, load_resnet


def create_model(embedding_size, model_name, retrain_type="none"):

    resnet_base, resnet_weights = load_resnet(model_name)

    resnet = resnet_base(weights=resnet_weights.DEFAULT)
    resnet_fc = resnet.fc

    if retrain_type == "none":
        for param in resnet.parameters():
            param.requires_grad = False

    if retrain_type == "fc":
        for param in resnet.parameters():
            param.requires_grad = False
        resnet_fc.requires_grad = True

    head = nn.Linear(resnet_fc.out_features, embedding_size)
    model = nn.Sequential(resnet, head)

    if retrain_type == "full":
        parameters = model.parameters()
    elif retrain_type == "fc":
        parameters = chain(resnet_fc.parameters(), head.parameters())
    else:
        parameters = head.parameters()

    model.eval()
    logging.info(f"new {model_name} model was created")

    return model, parameters


def main(config):

    mps_is_ok = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    device = torch.device("mps" if mps_is_ok else "cpu")
    logging.info(f"device: {device}")

    filename_design = os.path.join(
        config["shared"]["project_root"],
        config["metric_learning_resnet"]["filename_design"],
    )
    ann = pd.read_csv(filename_design)

    model_name = config["metric_learning_resnet"]["model_name"]
    _, resnet_weights = load_resnet(model_name)

    transform_resnet = resnet_weights.DEFAULT.transforms()
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

    column_fold = config["metric_learning_resnet"]["column_fold"]

    folds_use = config["metric_learning_resnet"].get("folds_use")
    if folds_use is None:
        folds_use = ann[column_fold].unique()

    batch_size = config["metric_learning_resnet"]["batch_size"]
    folder_images = os.path.join(
        config["shared"]["project_root"],
        config["metric_learning_resnet"]["folder_images"],
    )


    keys = "folder_output_model", "folder_output_report"
    for key in keys:
        value = os.path.join(
            config["shared"]["project_root"], config["metric_learning_resnet"][key]
        )
        config["metric_learning_resnet"][key] = value

    score_best_folds = []

    for fold_index in folds_use:

        if fold_index == "val":
            continue

        mlflow.start_run()

        mlflow.log_dict(config, "config-runtime.yaml")

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

        retrain_type = config["metric_learning_resnet"]["retrain_type"]
        embedding_size = config["metric_learning_resnet"]["embedding_size"]
        model, parameters = create_model(
            embedding_size, model_name, retrain_type=retrain_type
        )
        model = model.to(device)

        mlflow.log_params(
            {
                "model_name": model_name,
                "retrain_type": retrain_type,
                "embedding_size": embedding_size,
                "batch_size": batch_size
            }
        )

        label_weight = 1 / dataset_train.ann.label_code.value_counts().sort_index()
        mlflow.log_dict(label_weight.to_dict(), "label_weight.yml")

        criterion = nn.TripletMarginLoss()

        lr = config["metric_learning_resnet"]["optimizer"]["lr"]
        optimizer_name = config["metric_learning_resnet"]["optimizer"]["name"]

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(parameters, lr=lr)
        else:
            raise NotImplementedError()

        gamma = config["metric_learning_resnet"]["scheduler"]["gamma"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        mlflow.log_params(
            {
                "optimizer": optimizer_name,
                "lr": lr,
                "scheduler": "ExponentialLR",
                "gamma": gamma,
            }
        )

        ann_fold = dataset_train.ann
        code2label = (
            ann_fold[["label", "label_code"]]
            .drop_duplicates()
            .set_index("label_code")["label"]
        )
        code2label.sort_index(inplace=True)

        scorer = Scorer()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader_train=dataloader_train,
            dataloader_test=dataloader_test,
            params=config["metric_learning_resnet"],
            device=device,
            code2label=code2label,
            scheduler=scheduler,
            scorer=scorer
        )

        score_best = trainer.train()

        score_best_folds.append(score_best)

        mlflow.end_run()

    score_mean = np.mean(score_best_folds)

    return score_mean


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

    mlflow_experiment = config["metric_learning_resnet"].get(
        "mlflow_experiment", "metric_learning_resnet"
    )
    mlflow.set_experiment(mlflow_experiment)

    main(config)