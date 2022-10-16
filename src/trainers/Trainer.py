import os

import torch
import numpy as np
import pandas as pd
from tqdm import trange
import mlflow
import plotly.express as px
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        dataloader_train,
        dataloader_test,
        criterion,
        scorer,
        score_target,
        params,
        code2label,
        device,
    ):

        self.device = device
        self.model = model.to(device)

        self.optimizer = optimizer
        self.criterion = criterion

        self.dataloader = {"train": dataloader_train, "test": dataloader_test}

        self.params = params

        self.embeddings = {"train": None, "test": None}
        self.trues = {"train": None, "test": None}
        self.preds = {"train": None, "test": None}

        self.code2label = code2label
        self.scheduler = scheduler
        self.scorer = scorer
        self.score_target = score_target

    def step(self, train=True):

        step_name = "train" if train else "test"

        device = self.device

        dataloader = self.dataloader[step_name]

        loss_batches = []

        embeddings = []
        trues = []
        preds = []

        if train:
            self.model.train()

        for x, y in dataloader:

            x = x.to(device)
            y = y.to(device)

            pred, embedding = self.model(x)

            if train:
                self.optimizer.zero_grad()

            loss = self.criterion(pred, y)

            if train:
                loss.backward()
                self.optimizer.step()

            embeddings.append(embedding.detach())
            trues.append(y.detach())
            preds.append(pred.detach())

            loss_batches.append(loss.item())

        if train:
            self.model.eval()

        loss = np.mean(loss_batches)

        self.embeddings[step_name] = torch.concat(embeddings)
        self.trues[step_name] = torch.concat(trues)
        self.preds[step_name] = torch.concat(preds)

        scores = self.scorer(
            pred=self.preds[step_name],
            true=self.trues[step_name],
            embedding=self.embeddings[step_name],
        )

        return loss, scores

    def train_model(self):
        return self.step()

    def test_model(self):
        with torch.no_grad():
            return self.step(train=False)

    def train_epoch(self):

        objectives = {}

        step_names = "train", "test"
        steps = self.train_model, self.test_model

        for step_name, step in zip(step_names, steps):

            loss, scores = step()
            objectives[f"loss_{step_name}"] = loss

            for key, value in scores.items():
                objectives[f"{key}_{step_name}"] = value

        return objectives

    def train(self):

        params = self.params
        n_epochs = params["n_epochs"]

        mlflow.log_param("n_epochs", n_epochs)

        save_model_checkpoints = params["save_model_checkpoints"]

        score_test_best = -1e9

        with trange(n_epochs) as t:
            for epoch in t:

                objectives = self.train_epoch()

                lr = self.optimizer.param_groups[0]["lr"]
                mlflow.log_metric("lr", lr, step=epoch)

                if self.scheduler is not None:
                    self.scheduler.step()

                mlflow.log_metrics(objectives, step=epoch)

                metric_test = f"{self.score_target}_test"
                metric_train = f"{self.score_target}_train"

                score_test = objectives[metric_test]
                score_train = objectives[metric_train]

                df = self.collect_embeddings()
                self.update_plot_columns(df)

                filename = f"predictions/data-{epoch:03d}.csv"
                mlflow.log_text(df.to_csv(), filename)

                for color in "label_true", "label_pred":
                    fig = self.plot_embeddings(df, color)
                    filename = f"figures/embeddings-{color}-{epoch:03d}.html"
                    mlflow.log_figure(fig, filename)

                is_best = score_test > score_test_best

                if is_best:
                    score_test_best = score_test

                    mlflow.log_metric("score_test_best", score_test_best, step=epoch)

                    if save_model_checkpoints:
                        self.dump_model()

                t.set_postfix(train=score_train, test=score_test, best=score_test_best)

        return score_test_best

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

        model = model.to(self.device)

    def collect_embeddings(self):

        emb_train = self.embeddings["train"]
        emb_test = self.embeddings["test"]
        emb = torch.concat([emb_train, emb_test]).cpu().numpy()

        columns = [f"emb_{i:03d}" for i in range(emb.shape[1])]
        df = pd.DataFrame(emb, columns=columns)

        n_train = len(emb_train)
        df["fold"] = "test"
        df.loc[:n_train, "fold"] = "train"

        true_train = self.trues["train"]
        true_test = self.trues["test"]
        true = torch.concat([true_train, true_test]).cpu()

        pred_train = self.preds["train"]
        pred_test = self.preds["test"]
        pred = torch.concat([pred_train, pred_test]).cpu()

        proba_pred = torch.nn.functional.softmax(pred, dim=1)
        proba_true = true

        for i, label in self.code2label.iteritems():
            key = f"proba_pred_{label}"
            df[key] = proba_pred[:, i]
            key = f"proba_true_{label}"
            df[key] = proba_true[:, i]

        df["label_code_pred"] = pred.argmax(dim=1)
        df["label_code_true"] = true.argmax(dim=1)

        df["label_pred"] = df["label_code_pred"].replace(self.code2label)
        df["label_true"] = df["label_code_true"].replace(self.code2label)

        df.sort_values("label_true", inplace=True)

        return df

    def update_plot_columns(self, df):

        columns = [c for c in df.columns if c.startswith("emb")]
        X = df[columns].values

        n_dims = len(columns)

        if n_dims > 3:
            reducer = UMAP(n_components=3, random_state=42)
            pipeline_umap = make_pipeline(StandardScaler(), reducer)
            Z = pipeline_umap.fit_transform(X)
            df["x"] = Z[:, 0]
            df["y"] = Z[:, 1]
            df["z"] = Z[:, 2]

        else:
            for i in range(n_dims):
                letter = "xyz"[i]
                df[letter] = X[:, i]

    def plot_embeddings(self, df, color="label_true"):

        args = dict(
            color=color,
            symbol="fold",
            category_orders={color: self.code2label.values},
            width=800,
            height=600,
        )

        if "z" in df:
            fig = px.scatter_3d(df, x="x", y="y", z="z", **args)

        else:
            fig = px.scatter(df, x="x", y="y", **args)

        return fig
