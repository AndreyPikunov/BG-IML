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


class TrainerCombo:
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
        self.predictions = {"train": None, "test": None}
        self.trues = {"train": None, "test": None}

        self.code2label = code2label
        self.scheduler = scheduler
        self.scorer = scorer
        self.score_target = score_target

    def step(self, train=True):

        step_name = "train" if train else "test"

        device = self.device
        model = self.model

        dataloader = self.dataloader[step_name]
        criterion = self.criterion
        optimizer = self.optimizer

        loss_batches = []
        loss_supplement_batches = []

        a_embeddings = []
        a_predictions = []
        a_trues = []

        if train:
            model.train()

        for (a, p, n), (a_true, p_true, n_true) in dataloader:

            a = a.to(device)
            p = p.to(device)
            n = n.to(device)

            a_pred, a_embedded = model(a)
            p_pred, p_embedded = model(p)
            n_pred, n_embedded = model(n)

            if train:
                optimizer.zero_grad()

            triplet = a_embedded, p_embedded, n_embedded
            a_true = a_true.to(device)
            loss, loss_supplement = criterion(a_pred, a_true, triplet)

            if train:
                loss.backward()
                optimizer.step()

            a_embeddings.append(a_embedded.detach())
            a_predictions.append(a_pred.detach())
            a_trues.append(a_true.detach())

            loss_batches.append(loss.item())
            loss_supplement_batches.append(loss_supplement)

        if train:
            model.eval()

        loss = np.mean(loss_batches)

        loss_supplement = {}
        for key in loss_supplement_batches[0]:
            value_sum = 0
            for item in loss_supplement_batches:
                value = item[key].item()
                value_sum += value
            loss_supplement[key] = value_sum / len(loss_supplement_batches)

        self.embeddings[step_name] = torch.concat(a_embeddings)
        self.predictions[step_name] = torch.concat(a_predictions)
        self.trues[step_name] = torch.concat(a_trues)

        if self.scorer:
            scores = self.scorer(
                self.predictions[step_name],
                self.trues[step_name],
                self.embeddings[step_name],
            )
        else:
            scores = {}

        return loss, loss_supplement, scores

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

            loss, loss_supplement, scores = step()
            objectives[f"loss_{step_name}"] = loss

            for key, value in scores.items():
                objectives[f"{key}_{step_name}"] = value

            for key, value in loss_supplement.items():
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

                for color in "pred", "true":
                    fig = self.plot_embeddings(df, color)
                    filename = f"figures/embeddings-{color}-{epoch:03d}.html"
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

        pred_train = self.predictions["train"]
        pred_test = self.predictions["test"]
        pred = torch.concat([pred_train, pred_test]).argmax(dim=1).cpu().numpy()

        true_train = self.trues["train"]
        true_test = self.trues["test"]
        true = torch.concat([true_train, true_test]).argmax(dim=1).cpu().numpy()

        df = pd.DataFrame(data)

        df["pred"] = pred
        df["pred"].replace(self.code2label, inplace=True)

        df["true"] = true
        df["true"].replace(self.code2label, inplace=True)

        n_train = len(emb_train)
        df["fold"] = "test"
        df.loc[:n_train, "fold"] = "train"

        df.sort_values("true", inplace=True)

        return df

    def update_plot_columns(self, df):

        columns_drop = ["pred", "true", "fold"]
        X = df.drop(columns=columns_drop).values

        n_dims = X.shape[-1]

        if n_dims > 3:
            reducer = UMAP(n_components=3)
            pipeline_umap = make_pipeline(StandardScaler(), reducer)
            Z = pipeline_umap.fit_transform(X)
            df["x"] = Z[:, 0]
            df["y"] = Z[:, 1]
            df["z"] = Z[:, 2]

        else:
            for i in range(n_dims):
                letter = "xyz"[i]
                df[letter] = X[:, i]

    def plot_embeddings(self, df, color="pred"):

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
