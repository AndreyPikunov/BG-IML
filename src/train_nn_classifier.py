import os
import logging
import argparse

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import mlflow

from datasets import PaintingDataset
from trainers import Trainer
from scorers import ScorerCombo as Scorer
from models import NNClassifier

from utils import load_config, get_abs_dirname, load_resnet


def main(config):

    device = torch.device(config["train_nn_classifier"].get("device", "mps"))
    logging.info(f"device: {device}")

    filename_design = os.path.join(
        config["shared"]["project_root"],
        config["train_nn_classifier"]["filename_design"],
    )

    ann = pd.read_csv(filename_design)

    labels_use = config["train_nn_classifier"].get("labels_use")
    if labels_use is not None:
        ann = ann[ann.label.isin(labels_use)]
    else:
        labels_use = "all"

    ann["label_code"] = ann.label.astype("category").cat.codes

    resnet_name = config["train_nn_classifier"]["params_embedder"]["resnet_name"]
    _, resnet_weights = load_resnet(resnet_name)

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

    column_fold = config["train_nn_classifier"]["column_fold"]

    folds_use = config["train_nn_classifier"].get("folds_use")
    if folds_use is None:
        folds_use = ann[column_fold].unique()

    batch_size = config["train_nn_classifier"]["batch_size"]
    folder_images = os.path.join(
        config["shared"]["project_root"],
        config["train_nn_classifier"]["folder_images"],
    )

    keys = "folder_output_model", "folder_output_report"
    for key in keys:
        value = os.path.join(
            config["shared"]["project_root"], config["train_nn_classifier"][key]
        )
        config["train_nn_classifier"][key] = value

    score_best_folds = []

    for fold_index in folds_use:

        if fold_index == "val":
            continue

        mlflow.start_run()

        mlflow.log_dict(config, "config-runtime.yaml")

        mlflow.log_param("fold_index", fold_index)
        mlflow.log_param("labels_use", labels_use)

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

        ann_fold = dataset_train.ann
        code2label = (
            ann_fold[["label", "label_code"]]
            .drop_duplicates()
            .set_index("label_code")["label"]
        )
        code2label.sort_index(inplace=True)
        n_classes = len(ann_fold.label.unique())
        logging.info(f"n_classes: {n_classes}")
        mlflow.log_dict(code2label.to_dict(), "code2label.yml")

        params_embedder = config["train_nn_classifier"]["params_embedder"]
        mlflow.log_params(params_embedder)

        model = NNClassifier(n_classes, params_embedder=params_embedder)
        parameters = model.parameters()
        model = model.to(device)

        mlflow.log_param("n_classes", n_classes)

        lr = config["train_nn_classifier"]["optimizer"]["lr"]
        weight_decay = config["train_nn_classifier"]["optimizer"].get("weight_decay", 0)
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

        lr_final = config["train_nn_classifier"]["scheduler"]["lr_final"]
        n_epochs = config["train_nn_classifier"]["n_epochs"]
        gamma = np.power(lr_final / lr, 1 / n_epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        mlflow.log_params(
            {
                "optimizer": "Adam",
                "lr": lr,
                "weight_decay": weight_decay,
                "scheduler": "ExponentialLR",
                "gamma": gamma,
            }
        )

        label_weight = 1 / dataset_train.ann.label_code.value_counts().sort_index()
        mlflow.log_dict(label_weight.to_dict(), "label_weight.yml")

        kw_criterion = {
            "weight": torch.tensor(label_weight.values).float().to(device),
            "label_smoothing": config["train_nn_classifier"].get("label_smoothing"),
        }

        criterion = torch.nn.CrossEntropyLoss(**kw_criterion)

        mlflow.log_params(kw_criterion)

        top_k_list = config["train_nn_classifier"]["top_k_list"]
        scorer = Scorer(top_k_list, code2label=code2label)
        mlflow.log_param("top_k", top_k_list)

        score_target = config["train_nn_classifier"]["score_target"]

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader_train=dataloader_train,
            dataloader_test=dataloader_test,
            params=config["train_nn_classifier"],
            device=device,
            code2label=code2label,
            scheduler=scheduler,
            scorer=scorer,
            score_target=score_target,
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

    mlflow_experiment = config["train_nn_classifier"].get(
        "mlflow_experiment", "train_nn_classifier"
    )
    mlflow.set_experiment(mlflow_experiment)

    main(config)
