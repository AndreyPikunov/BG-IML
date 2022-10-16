import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    top_k_accuracy_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


class ScorerCombo:
    def __init__(self, top_k_list=3, code2label=None):
        self.code2label = code2label
        self.top_k_list = top_k_list
        self.n_classes = len(code2label)
        self.codes = list(range(self.n_classes))

    def prepare_y(self, y_onehot):
        with torch.no_grad():
            y = y_onehot.cpu().argmax(dim=1).numpy()
        return y

    def numpify(self, x):
        with torch.no_grad():
            result = x.cpu().numpy()
        return result

    def __call__(self, pred, true, embedding):

        class_weight = 1 / true.mean(axis=0).cpu().numpy()
        pred_scores = self.numpify(pred)
        pred = self.prepare_y(pred)
        true = self.prepare_y(true)
        X = self.numpify(embedding)

        precision, recall, fscore, _ = precision_recall_fscore_support(
            true, pred, average="weighted", zero_division=0
        )

        scores_classification = dict(
            precision=precision,
            recall=recall,
            fscore=fscore,
        )

        sample_weight = [class_weight[i] for i in true]

        for k in self.top_k_list:
            acc = top_k_accuracy_score(
                true, pred_scores, k=k, sample_weight=sample_weight, labels=self.codes
            )
            scores_classification[f"top_{k}_accuracy"] = acc

        scores_clustering = {
            "silhouette_pred": silhouette_score(X, pred),
            "davies_bouldin_pred": davies_bouldin_score(X, pred),
            "calinski_harabasz_pred": calinski_harabasz_score(X, pred),
            "silhouette_true": silhouette_score(X, true),
            "davies_bouldin_true": davies_bouldin_score(X, true),
            "calinski_harabasz_true": calinski_harabasz_score(X, true),
        }

        result = {**scores_classification, **scores_clustering}

        return result
