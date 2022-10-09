import os

import torch
import numpy as np
import pandas as pd
from tqdm import trange
import mlflow
import plotly.express as px


class TrainerTriplet:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        dataloader_train,
        dataloader_test,
        params,
        device="cpu",
        **kw,
    ):

        self.device = device
        self.model = model.to(device)

        self.optimizer = optimizer
        self.criterion = criterion

        self.dataloader = {"train": dataloader_train, "test": dataloader_test}

        self.params = params

        self.embeddings = {"train": None, "test": None}
        self.labels = {"train": None, "test": None}

        self.code2label = kw.get("code2label")
        self.scheduler = kw.get("scheduler")
        self.scorer = kw.get("scorer")

    def test_model(self):

        with torch.no_grad():

            device = self.device
            model = self.model
            dataloader = self.dataloader["test"]
            criterion = self.criterion

            loss_batches = []

            a_embeddings = []
            a_labels = []

            for (a, p, n), (a_label, p_label, n_label) in dataloader:

                assert (a_label == p_label).all()
                assert (a_label != n_label).all()

                a = a.to(device)
                p = p.to(device)
                n = n.to(device)

                a_embedded = model(a)
                p_embedded = model(p)
                n_embedded = model(n)

                a_embeddings.append(a_embedded)
                a_labels.append(a_label)

                loss = criterion(a_embedded, p_embedded, n_embedded)

                loss_batches.append(loss.item())

            loss = np.mean(loss_batches)

            self.embeddings["test"] = torch.concat(a_embeddings).detach()
            self.labels["test"] = torch.concat(a_labels)

            if self.scorer:
                scores = self.scorer(self.embeddings["test"], self.labels["test"])
            else:
                scores = {}
                
            return loss, scores

    def train_model(self):

        device = self.device
        model = self.model
        dataloader = self.dataloader["train"]
        criterion = self.criterion
        optimizer = self.optimizer

        loss_batches = []

        a_embeddings = []
        a_labels = []

        model.train()

        for (a, p, n), (a_label, p_label, n_label) in dataloader:

            assert (a_label == p_label).all()
            assert (a_label != n_label).all()

            a = a.to(device)
            p = p.to(device)
            n = n.to(device)

            a_embedded = model(a)
            p_embedded = model(p)
            n_embedded = model(n)

            a_embeddings.append(a_embedded)
            a_labels.append(a_label)

            optimizer.zero_grad()

            loss = criterion(a_embedded, p_embedded, n_embedded)

            loss.backward()

            optimizer.step()

            loss_batches.append(loss.item())

        model.eval()

        loss = np.mean(loss_batches)

        self.embeddings["train"] = torch.concat(a_embeddings).detach()
        self.labels["train"] = torch.concat(a_labels)

        if self.scorer:
            scores = self.scorer(self.embeddings["train"], self.labels["train"])
        else:
            scores = {}

        return loss, scores

    def train_epoch(self):

        objectives = {}

        loss, scores = self.train_model()
        objectives["loss_train"] = loss
        for key, value in scores.items():
            objectives[f"{key}_train"] = value

        loss, scores = self.test_model()
        objectives["loss_test"] = loss
        for key, value in scores.items():
            objectives[f"{key}_test"] = value

        return objectives

    def train(self):

        params = self.params
        n_epochs = params["n_epochs"]

        mlflow.log_param("n_epochs", n_epochs)

        save_model_checkpoints = params["save_model_checkpoints"]

        loss_test_best = 1e9

        with trange(n_epochs) as t:
            for epoch in t:

                objectives = self.train_epoch()

                lr = self.optimizer.param_groups[0]["lr"]
                mlflow.log_metric("lr", lr, step=epoch)

                if self.scheduler is not None:
                    self.scheduler.step()

                mlflow.log_metrics(objectives, step=epoch)

                loss_test = objectives["loss_test"]
                loss_train = objectives["loss_train"]

                fig = self.plot_embeddings()
                filename = f"figures/embeddings-latest.html"
                mlflow.log_figure(fig, filename)

                if loss_test < loss_test_best:
                    loss_test_best = loss_test

                    mlflow.log_metrics(
                        {
                            "best_epoch": epoch,
                            "best_loss_test": loss_test,
                            "best_loss_train": loss_train,
                        },
                        step=epoch,
                    )

                    if save_model_checkpoints:
                        self.dump_model()

                    fig = self.plot_embeddings()
                    filename = f"figures/embeddings-{epoch:03d}.html"
                    mlflow.log_figure(fig, filename)

                t.set_postfix(train=loss_train, test=loss_test, best=loss_test_best)

        return loss_test_best

    def dump_model(self):

        params = self.params

        folder_output_model = params["folder_output_model"]
        os.makedirs(folder_output_model, exist_ok=True)

        filename_save_model_st = os.path.join(folder_output_model, "model.st")

        model = self.model
        model = model.to("cpu")
        torch.save(model.state_dict(), filename_save_model_st)
        mlflow.log_artifact(filename_save_model_st)

        filename_save_optimizer_st = os.path.join(folder_output_model, "optimizer.st")
        torch.save(self.optimizer.state_dict(), filename_save_optimizer_st)
        mlflow.log_artifact(filename_save_optimizer_st)

        # filename_save_model_pt = os.path.join(folder_output_model, "model.pt")
        # model_scripted = torch.jit.script(model)
        # model_scripted.save(filename_save_model_pt)
        # mlflow.log_artifact(filename_save_model_pt)

        model = model.to(self.device)

    def collect_embeddings(self):
        emb_train = self.embeddings["train"]
        emb_test = self.embeddings["test"]
        data = torch.concat([emb_train, emb_test]).cpu().numpy()

        labels = torch.concat([self.labels["train"], self.labels["test"]]).cpu().numpy()

        df = pd.DataFrame(data)
        df["label"] = labels
        df["label"].replace(self.code2label, inplace=True)

        n_train = len(emb_train)
        df["fold"] = "test"
        df.loc[:n_train, "fold"] = "train"

        return df

    def plot_embeddings(self):

        df = self.collect_embeddings()

        args = dict(x=0, y=1, color="label", symbol="fold", width=800, height=600)

        if 2 in df:
            fig = px.scatter_3d(df, z=2, **args)
        else:
            fig = px.scatter(df, **args)

        return fig
