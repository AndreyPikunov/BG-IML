from torch import nn

from utils import load_resnet


class Embedder(nn.Module):
    def __init__(self, resnet_name, embedding_size, freeze_resnet_cnn=True, freeze_resnet_fc=False):
        super().__init__()
        resnet_base, resnet_weights = load_resnet(resnet_name)
        self.resnet = resnet_base(weights=resnet_weights.DEFAULT)

        self.embedding_size = embedding_size
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.resnet.fc.out_features, embedding_size)
        )

        if freeze_resnet_cnn:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        if not freeze_resnet_fc:
            for param in self.resnet.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        y = self.resnet(x)
        y = self.head(y)
        return y


class NNClassifier(nn.Module):
    def __init__(self, n_classes, params_embedder):

        super().__init__()

        self.embedder = Embedder(**params_embedder)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedder.embedding_size, n_classes)
        )

    def embed(self, x):
        embedding = self.embedder(x)
        return embedding

    def forward(self, x):
        embedding = self.embed(x)
        y = self.head(embedding)
        return y, embedding
