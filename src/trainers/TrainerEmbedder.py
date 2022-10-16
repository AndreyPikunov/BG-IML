import os
import logging

import torch
import numpy as np
import pandas as pd
from tqdm import trange
import mlflow
import plotly.express as px
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class TrainerEmbedder:
    def __init__(
        self,
        model,
        optimizers,
        scheduler,
        dataloader_train,
        dataloader_test,
        criterion,
        scorer,
        score_target,
        params,
        code2label_train,
        code2label_test,
        device,
    ):

        self.device = device
        self.model = model.to(device)

        self.optimizers = optimizers
        self.criterion = criterion

        self.dataloader = {"train": dataloader_train, "test": dataloader_test}

        self.params = params

        self.embeddings = {"train": None, "test": None}
        self.trues = {"train": None, "test": None}

        self.code2label = {"train": code2label_train, "test": code2label_test}
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

        if train:
            self.model.train()

        for x, y in dataloader:

            # y = y.to(device)
            x = x.to(device)
            embedding = self.model(x)

            if train:

                for opt in self.optimizers:
                    opt.zero_grad()

                msg = f"{x.shape} {y.shape} {embedding.shape}"
                logging.debug(msg)

                loss = self.criterion(embedding, y)
                loss.backward()

                for opt in self.optimizers:
                    opt.step()

                loss_batches.append(loss.item())

            embeddings.append(embedding.detach())
            trues.append(y.detach())

        if train:
            self.model.eval()

        loss = np.mean(loss_batches) if train else 0.0

        self.embeddings[step_name] = torch.concat(embeddings)
        self.trues[step_name] = torch.concat(trues)

        if self.scorer:
            scores = self.scorer(
                self.embeddings[step_name],
                self.trues[step_name],
            )
        else:
            scores = {}

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

                for i, opt in enumerate(self.optimizers):
                    lr = opt.param_groups[0]["lr"]
                    mlflow.log_metric(f"lr-{i}", lr, step=epoch)

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

                fig = self.plot_embeddings(df)
                filename = f"figures/embeddings-{epoch:03d}.html"
                mlflow.log_figure(fig, filename)

                is_best = score_test > score_test_best

                if is_best:
                    score_test_best = score_test

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

        for i, opt in enumerate(self.optimizers):
            filename_save_optimizer_st = os.path.join(
                folder_output_model, f"optimizer-{i}.st"
            )
            torch.save(opt.state_dict(), filename_save_optimizer_st)
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

        df = pd.DataFrame(data)

        true_train = [self.code2label["train"][i.item()] for i in self.trues["train"]]
        true_test = [self.code2label["test"][i.item()] for i in self.trues["test"]]
        true = true_train + true_test
        df["true"] = true

        n_train = len(true_train)
        df["fold"] = "test"
        df.iloc[:n_train, "fold"] = "train"

        df.sort_values("true", inplace=True)

        return df

    def update_plot_columns(self, df):

        columns_drop = ["true", "fold"]
        X = df.drop(columns=columns_drop).values

        n_dims = X.shape[-1]

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

    def plot_embeddings(self, df):

        args = dict(
            color="true",
            symbol="fold",
            width=800,
            height=600,
        )

        if "z" in df:
            fig = px.scatter_3d(df, x="x", y="y", z="z", **args)

        else:
            fig = px.scatter(df, x="x", y="y", **args)

        return fig
