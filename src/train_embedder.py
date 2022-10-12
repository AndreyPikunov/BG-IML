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

from pytorch_metric_learning import losses as pml_losses

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

    model_name = config["train_embedder"]["model_name"]
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

    column_fold = config["train_embedder"]["column_fold"]

    folds_use = config["train_embedder"].get("folds_use")
    if folds_use is None:
        folds_use = ann[column_fold].unique()

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
            ann[mask_train], folder_images, transform_train=transform_train, apply_one_hot=False
        )

        dataset_test = PaintingDataset(
            ann[mask_test], folder_images, transform_preprocess=transform_resnet, apply_one_hot=False
        )

        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        embedding_size = config["train_embedder"]["embedding_size"]

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

        model = Embedder(model_name, embedding_size)

        for param in model.resnet.parameters():
            param.requires_grad = False
        
        for param in model.resnet.fc.parameters():
            param.requires_grad = True

        parameters = model.parameters()
        model = model.to(device)

        mlflow.log_params(
            {
                "model_name": model_name,
                "embedding_size": embedding_size,
                "batch_size": batch_size,
                "n_classes": n_classes,
            }
        )

        pairs = ("test", dataset_test), ("train", dataset_train)
        # important! "train" MUST be the last
        for name, dataset in pairs:
            label_weight = 1 / dataset.ann.label_code.value_counts().sort_index()
            label_weight = (label_weight / label_weight.sum()) * 100
            mlflow.log_dict(label_weight.to_dict(), f"label-weight-{name}.yml")

        if False:
            kw_criterion = {
                "margin": config["train_embedder"]["loss"]["margin"],
            }

            loss_name = config["train_embedder"]["loss"]["name"]
            if loss_name == "TripletMarginLoss":
                ...
                # criterion = pml_losses.TripletMarginLoss(**kw_criterion)
                # criterion = pml_losses.ContrastiveLoss(**kw_criterion)

                # from pytorch_metric_learning import miners
                # miner = miners.TripletMarginMiner()
                # criterion = lambda x, y: loss(x, y, miner(x, y))

            else:
                raise RuntimeError()

                mlflow.log_params({"criterion": loss_name, **kw_criterion})

        # criterion = pml_losses.ContrastiveLoss()
        criterion = pml_losses.TripletMarginLoss(margin=1)

        lr = config["train_embedder"]["optimizer"]["lr"]
        optimizer_name = config["train_embedder"]["optimizer"]["name"]

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(parameters, lr=lr)
        else:
            raise NotImplementedError()

        gamma = config["train_embedder"]["scheduler"]["gamma"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        mlflow.log_params(
            {
                "optimizer": optimizer_name,
                "lr": lr,
                "scheduler": "ExponentialLR",
                "gamma": gamma,
            }
        )

        scorer = Scorer()
        score_target = config["train_embedder"]["score_target"]

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader_train=dataloader_train,
            dataloader_test=dataloader_test,
            params=config["train_embedder"],
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

    mlflow_experiment = config["train_embedder"].get("mlflow_experiment", "train_embedder")
    mlflow.set_experiment(mlflow_experiment)

    main(config)
