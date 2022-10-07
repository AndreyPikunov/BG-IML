import os

import torch
import numpy as np
from tqdm import trange
import mlflow


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
    ):

        self.device = device
        self.model = model.to(device)

        self.optimizer = optimizer
        self.criterion = criterion

        self.dataloader = {"train": dataloader_train, "test": dataloader_test}

        self.params = params

    def test_model(self):

        with torch.no_grad():

            device = self.device
            model = self.model
            dataloader = self.dataloader["test"]
            criterion = self.criterion

            loss_batches = []

            for (a, p, n), (a_label, p_label, n_label) in dataloader:

                assert (a_label == p_label).all()
                assert (a_label != n_label).all()

                a = a.to(device)
                p = p.to(device)
                n = n.to(device)

                a_embedded = model(a)
                p_embedded = model(p)
                n_embedded = model(n)

                loss = criterion(a_embedded, p_embedded, n_embedded)

                loss_batches.append(loss.item())

            loss = np.mean(loss_batches)

            return loss

    def train_model(self):

        device = self.device
        model = self.model
        dataloader = self.dataloader["train"]
        criterion = self.criterion
        optimizer = self.optimizer

        loss_batches = []

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

            optimizer.zero_grad()

            loss = criterion(a_embedded, p_embedded, n_embedded)

            loss.backward()

            optimizer.step()

            loss_batches.append(loss.item())

        model.eval()

        loss = np.mean(loss_batches)

        return loss

    def train_epoch(self):

        objectives = {}

        loss = self.train_model()
        objectives["loss_train"] = loss

        loss = self.test_model()
        objectives["loss_test"] = loss

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

                mlflow.log_metrics(objectives, step=epoch)

                loss_test = objectives["loss_test"]
                loss_train = objectives["loss_train"]

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

                t.set_postfix(train=loss_train, test=loss_test, best=loss_test_best)

        return loss_test_best

    def dump_model(self):

        params = self.params

        folder_output_model = params["folder_output_model"]
        os.makedirs(folder_output_model, exist_ok=True)

        filename_save_model_st = os.path.join(folder_output_model, "model.st")
        filename_save_model_pt = os.path.join(folder_output_model, "model.pt")

        model = self.model
        model = model.to("cpu")
        torch.save(model.state_dict(), filename_save_model_st)
        mlflow.log_artifact(filename_save_model_st)

        model_scripted = torch.jit.script(model)
        model_scripted.save(filename_save_model_pt)
        mlflow.log_artifact(filename_save_model_pt)
        model = model.to(self.device)
