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
from trainers import TrainerEmbedder as Trainer
from scorers import ScorerClustering as Scorer
from models import Embedder

from pytorch_metric_learning import losses, reducers

from utils import load_config, get_abs_dirname, load_resnet


def main(config):

    device = torch.device(config["train_embedder"].get("device", "mps"))
    logging.info(f"device: {device}")

    filename_design = os.path.join(
        config["shared"]["project_root"],
        config["train_embedder"]["filename_design"],
    )

    ann = pd.read_csv(filename_design)

    labels_use = config["train_embedder"].get("labels_use")
    if labels_use is not None:
        ann = ann[ann.label.isin(labels_use)]
    else:
        labels_use = "all"

    ann["label_code"] = ann.label.astype("category").cat.codes

    resnet_name = config["train_embedder"]["params_embedder"]["resnet_name"]
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

    labels_test = config["train_embedder"]["labels_test"]
    batch_size = config["train_embedder"]["batch_size"]
    folder_images = os.path.join(
        config["shared"]["project_root"],
        config["train_embedder"]["folder_images"],
    )

    keys = "folder_output_model", "folder_output_report"
    for key in keys:
        value = os.path.join(
            config["shared"]["project_root"], config["train_embedder"][key]
        )
        config["train_embedder"][key] = value

    score_best_folds = []

    mlflow.start_run()

    mlflow.log_dict(config, "config-runtime.yaml")

    mlflow.log_param("labels_use", labels_use)

    mask_val = ann.fold_author == "val"
    mask_test = ann.label.isin(labels_test) & (~mask_val)
    mask_train = ~(mask_test | mask_val)

    dataset_train = PaintingDataset(
        ann[mask_train],
        folder_images,
        transform_train=transform_train,
        apply_one_hot=False,
        remake_label_code=True
    )

    dataset_test = PaintingDataset(
        ann[mask_test],
        folder_images,
        transform_preprocess=transform_resnet,
        apply_one_hot=False,
        remake_label_code=True
    )

    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True
    )
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    n_classes = len(dataset_train.ann.label.unique())
    logging.info(f"n_classes: {n_classes}")
    mlflow.log_param("n_classes", n_classes)

    code2label_dict = {}

    for name, dataset in (("train", dataset_train), ("test", dataset_test)):
        ann_dataset = dataset.ann
        code2label = (
            ann_dataset[["label", "label_code"]]
            .drop_duplicates()
            .set_index("label_code")["label"]
        )
        code2label.sort_index(inplace=True)
        code2label_dict[name] = code2label
        mlflow.log_dict(code2label.to_dict(), f"code2label-{name}.yml")

    params_embedder = config["train_embedder"]["params_embedder"]
    model = Embedder(**params_embedder)

    parameters = model.parameters()
    model = model.to(device)

    lr = config["train_embedder"]["optimizer"]["lr"]
    weight_decay = config["train_embedder"]["optimizer"].get("weight_decay", 0)
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    lr_final = config["train_embedder"]["scheduler"]["lr_final"]
    n_epochs = config["train_embedder"]["n_epochs"]
    gamma = np.power(lr_final / lr, 1 / n_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    mlflow.log_params(
        {
            "optimizer": "Adam",
            "lr": lr,
            "lr_final": lr_final,
            "weight_decay": weight_decay,
            "scheduler": "ExponentialLR",
            "gamma": gamma,
        }
    )

    label_weight = 1 / dataset_train.ann.label_code.value_counts().sort_index()
    label_weight = (label_weight / label_weight.sum()) * 100
    mlflow.log_dict(label_weight.to_dict(), f"label-weight.yml")

    criterion = losses.ArcFaceLoss(
        num_classes=n_classes,
        embedding_size=params_embedder["embedding_size"],
        reducer=reducers.ClassWeightedReducer(
            weights=torch.tensor(label_weight.values).float()
        ),
    )

    lr = config["train_embedder"]["optimizer_loss"]["lr"]
    optimizer_loss = torch.optim.Adam(criterion.parameters(), lr=lr)

    mlflow.log_params({
        "criterion": "ArcFaceLoss",
        "use_class_weight": True,
        "lr_loss": lr,
    })

    scorer = Scorer()
    score_target = config["train_embedder"]["score_target"]

    optimizers = optimizer, optimizer_loss

    trainer = Trainer(
        model=model,
        optimizers=optimizers,
        criterion=criterion,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        params=config["train_embedder"],
        device=device,
        code2label_train=code2label_dict["train"],
        code2label_test=code2label_dict["test"],
        scheduler=scheduler,
        scorer=scorer,
        score_target=score_target
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

    mlflow_experiment = config["train_embedder"].get("mlflow_experiment", "train_embedder")
    mlflow.set_experiment(mlflow_experiment)

    main(config)
