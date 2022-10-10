from collections import OrderedDict

from torch import nn

from utils import load_resnet


class Embedder(nn.Module):
    def __init__(self, resnet_name, embedding_size):
        super().__init__()
        resnet_base, resnet_weights = load_resnet(resnet_name)
        self.resnet = resnet_base(weights=resnet_weights.DEFAULT)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.resnet.fc.out_features, embedding_size)
        )

    def forward(self, x):
        y = self.resnet(x)
        y = self.head(y)
        return y


class NNClassifier(nn.Module):
    def __init__(self, resnet_name, embedding_size, n_classes):

        super().__init__()

        self.embedder = Embedder(resnet_name, embedding_size)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_size, n_classes)
        )

    def embed(self, x):
        z = self.embedder(x)
        return z

    def forward(self, x):
        embedding = self.embed(x)
        y = self.head(embedding)
        return y, embedding
