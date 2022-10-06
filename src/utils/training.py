import os

import torch
from tqdm import trange
import mlflow


def get_device(device="cpu"):
    if device == "cpu":
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    return device


def predict_model(model, dataloader, device="cpu"):

    device = get_device(device)

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for X, y_true in dataloader:
            X = X.to(device)
            y_true = y_true.to(device)
            y_pred = model(X)
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)

    y_true = torch.vstack(y_true_list)
    y_pred = torch.vstack(y_pred_list)

    return y_pred, y_true


def test_model(model, dataloader, criterion, scorer, device="cpu"):

    y_pred, y_true = predict_model(model, dataloader, device)

    loss = criterion(y_pred, y_true).item()
    score = scorer(y_pred, y_true)

    return loss, score


def train_model(model, dataloader, criterion, scorer, optimizer, device="cpu"):

    y_true_list = []
    y_pred_list = []

    device = get_device(device)

    model.train()

    for X, y_true in dataloader:
        X = X.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()
        y_pred = model(X)

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    model.eval()

    y_true = torch.vstack(y_true_list)
    y_pred = torch.vstack(y_pred_list)

    loss = criterion(y_pred, y_true).item()
    score = scorer(y_pred, y_true)

    return loss, score


def train_epoch(
    model, optimizer, criterion, scorer, train_dataloader, test_dataloader, device="cpu"
):

    device = get_device(device)

    objectives = {}

    loss, score = train_model(
        model, train_dataloader, criterion, scorer, optimizer, device
    )
    objectives["loss_train"] = loss
    objectives["score_train"] = score

    loss, score = test_model(model, test_dataloader, criterion, scorer, device)
    objectives["loss_test"] = loss
    objectives["score_test"] = score

    return objectives


def train(
    model,
    optimizer,
    criterion,
    scorer,
    train_dataloader,
    test_dataloader,
    params,
    device="cpu",
):

    folder_output_model = params["folder_output_model"]
    os.makedirs(folder_output_model, exist_ok=True)

    filename_save_model_st = os.path.join(folder_output_model, "model.st")
    filename_save_model_pt = os.path.join(folder_output_model, "model.pt")

    folder_output_progress = params["folder_output_report"]
    os.makedirs(folder_output_progress, exist_ok=True)

    score_test_best = 0.0

    rows = []
    n_epochs = params["n_epochs"]

    mlflow.log_param("n_epochs", n_epochs)

    save_model_checkpoints = params["save_model_checkpoints"]

    with trange(n_epochs) as t:
        for epoch in t:

            objectives = train_epoch(
                model,
                optimizer,
                criterion,
                scorer,
                train_dataloader,
                test_dataloader,
                device,
            )
            rows.append(objectives)

            mlflow.log_metrics(objectives, step=epoch)

            score_test = objectives["score_test"]
            score_train = objectives["score_train"]

            if score_test > score_test_best:
                score_test_best = score_test

                mlflow.log_metrics(
                    {
                        "best_epoch": epoch,
                        "best_score_test": score_test,
                        "best_score_train": score_train,
                    },
                    step=epoch,
                )

                y_pred, y_true = predict_model(model, test_dataloader, device)
                report = scorer.report(y_pred, y_true)

                mlflow.log_text(
                    report["classification_report_text"],
                    "reports/best/classification_report.txt",
                )

                mlflow.log_figure(
                    report["confusion_matrix_figure"],
                    "figures/best/confusion_matrix.png",
                )

                if save_model_checkpoints:
                    torch.save(model.state_dict(), filename_save_model_st)
                    mlflow.log_artifact(filename_save_model_st)

                    model_scripted = torch.jit.script(model)
                    model_scripted.save(filename_save_model_pt)
                    mlflow.log_artifact(filename_save_model_pt)

            t.set_postfix(train=score_train, test=score_test, best=score_test_best)
