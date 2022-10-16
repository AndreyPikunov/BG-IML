import torch
from sklearn.metrics import silhouette_score, davies_bouldin_score


class ScorerClustering:
    def numpify(self, x):
        with torch.no_grad():
            result = x.cpu().numpy()
        return result

    def __call__(self, X, labels):
        X = self.numpify(X)
        result = {
            "silhouette_score": silhouette_score(X, labels),
            "davies_bouldin_score": davies_bouldin_score(X, labels),
        }
        return result
