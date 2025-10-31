import torch.nn as nn
import torch.nn.functional as F
from Prior2Former import (
    apply_nonlinearity,
)


class DPNHead(nn.Module):
    def __init__(self, nonlinearity: str, hidden_dim: int, num_classes: int):
        super().__init__()

        self.nonlinearity = nonlinearity
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.class_embed(x)

        return logits

    def probabilities(self, logits):
        x = apply_nonlinearity(self.nonlinearity, logits) + 1
        return x / x.sum(-1).unsqueeze(-1)

    def get_uncertainty(self, cls_pred):
        # exspects that the softplus was already applied
        if isinstance(cls_pred, dict):
            evidence = cls_pred["cls_pred"]
        else:
            evidence = cls_pred
        alpha = evidence + 1
        alpha_sum = alpha.sum(dim=-1)
        uncertainty = alpha / alpha_sum.unsqueeze(-1)
        return uncertainty

    def get_alpha_sum(self, cls_pred):
        if isinstance(cls_pred, dict):
            evidence = cls_pred["cls_pred"]
        else:
            evidence = cls_pred
        alpha = evidence + 1
        alpha_sum = alpha.sum(dim=-1)
        return alpha_sum

class DPNHead_BinaryNNO(DPNHead):
    def probabilities(self, logits):
        x = apply_nonlinearity(self.nonlinearity, logits[...,:self.num_classes]) + 1
        return x / x.sum(-1).unsqueeze(-1)

    def get_uncertainty(self, cls_pred):
        if isinstance(cls_pred, dict):
            evidence = cls_pred["cls_pred"]
        else:
            evidence = cls_pred
        alpha = evidence[...,:self.num_classes] + 1
        alpha_sum = alpha.sum(dim=-1)
        uncertainty = alpha / alpha_sum.unsqueeze(1)
        return uncertainty

    def mask_probabilities(self, logits):
        x = F.sigmoid(logits[...,self.num_classes-1])
        return x

    def mask_certainty(self, cls_pred):
        certainty = cls_pred[...,self.num_classes].sigmoid()
        return 1 - certainty


class Linear(nn.Module):

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()

        self.class_embed = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.class_embed(x)

    def probabilities(self, x):
        return nn.functional.softmax(x, dim=-1)

    def get_uncertainty(self, cls_pred):
        certainty = cls_pred.softmax(0)
        return 1 - certainty
