from torch import nn


class ComboLoss(nn.Module):
    def __init__(self, classification_weight=0.5, class_weights=None, label_smoothing=None):

        super().__init__()

        self.CE_weight = classification_weight
        self.Triplet_weight = 1 - self.CE_weight

        if label_smoothing is None:
            label_smoothing = 0.
        self.CELoss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        
        self.TripletLoss = nn.TripletMarginLoss()

    def forward(self, y_pred, y_true, embeddings=None):

        loss_ce = self.CELoss(y_pred, y_true)

        if embeddings is not None:
            a, p, n = embeddings
            loss_triplet = self.TripletLoss(a, p, n)
        else:
            loss_triplet = 0

        loss_ce_weighted = self.CE_weight * loss_ce
        loss_triplet_weighted = self.Triplet_weight * loss_triplet
        loss = loss_ce_weighted + loss_triplet_weighted

        supplement = {
            "loss_ce": loss_ce,
            "loss_ce": loss_triplet,
            "loss_ce_weighted": loss_ce_weighted,
            "loss_triplet_weighted": loss_triplet_weighted,
        }

        return loss
