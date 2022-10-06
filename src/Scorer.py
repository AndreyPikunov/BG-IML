import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from utils.visualization import plot_confusion_matrix


class Scorer:
    def __init__(self, code2label=None):
        self.code2label = code2label

    def _prepare_y(self, y):
        with torch.no_grad():
            result = y.cpu().argmax(dim=1).numpy()
        return result

    def __call__(self, y_pred, y_true):
        y_true = self._prepare_y(y_true)
        y_pred = self._prepare_y(y_pred)
        score = f1_score(y_true, y_pred, average="micro")
        return score

    def report(self, y_pred, y_true):
        y_true = self._prepare_y(y_true)
        y_pred = self._prepare_y(y_pred)

        cl_report = classification_report(y_true, y_pred, target_names=self.code2label)

        confmat = confusion_matrix(y_true, y_pred)

        fig, ax = plot_confusion_matrix(
            confmat,
            labels=self.code2label,
            xticks_rotation="vertical",
            cmap="Purples",
            normalize_plot="true",
        )

        result = {
            "classification_report_text": cl_report,
            "confusion_matrix_figure": fig,
        }

        return result
