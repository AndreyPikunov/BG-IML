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


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scorer,
        dataloader_train,
        dataloader_test,
        params,
        device="cpu",
    ):

        self.device = get_device(device)
        self.model = model.to(device)

        self.optimizer = optimizer
        self.criterion = criterion
        self.scorer = scorer

        self.dataloader = {"train": dataloader_train, "test": dataloader_test}

        self.y_true = {"train": None, "test": None}

        self.y_pred = {"train": None, "test": None}

        self.params = params

    def predict_model(self, dataloader):

        device = self.device
        model = self.model

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

    def test_model(self):

        y_pred, y_true = self.predict_model(self.dataloader["test"])
        self.y_pred["test"] = y_pred
        self.y_true["test"] = y_true

        loss = self.criterion(y_pred, y_true).item()
        score = self.scorer(y_pred, y_true)

        return loss, score

    def train_model(self):

        y_true_list = []
        y_pred_list = []

        device = self.device
        model = self.model
        dataloader = self.dataloader["train"]
        criterion = self.criterion
        scorer = self.scorer
        optimizer = self.optimizer

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
        self.y_pred["train"] = y_pred
        self.y_true["train"] = y_true

        loss = criterion(y_pred, y_true).item()
        score = scorer(y_pred, y_true)

        return loss, score

    def train_epoch(self):

        objectives = {}

        loss, score = self.train_model()
        objectives["loss_train"] = loss
        objectives["score_train"] = score

        loss, score = self.test_model()
        objectives["loss_test"] = loss
        objectives["score_test"] = score

        return objectives

    def train(self):

        params = self.params

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

                objectives = self.train_epoch()
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

                    y_pred, y_true = self.y_pred["test"], self.y_true["test"]
                    report = self.scorer.report(y_pred, y_true)

                    mlflow.log_text(
                        report["classification_report_text"],
                        "reports/best/classification_report.txt",
                    )

                    fig = report["confusion_matrix_figure"]
                    mlflow.log_figure(
                        fig,
                        "figures/best/confusion_matrix.png",
                    )
                    del fig

                    if save_model_checkpoints:

                        model = self.model
                        model = model.to("cpu")
                        torch.save(model.state_dict(), filename_save_model_st)
                        mlflow.log_artifact(filename_save_model_st)

                        model_scripted = torch.jit.script(model)
                        model_scripted.save(filename_save_model_pt)
                        mlflow.log_artifact(filename_save_model_pt)
                        model = model.to(self.device)

                t.set_postfix(train=score_train, test=score_test, best=score_test_best)

        return score_test_best
