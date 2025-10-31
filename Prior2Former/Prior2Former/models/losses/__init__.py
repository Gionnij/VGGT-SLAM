from typing import Optional

import torch
import torch.nn as nn
from .dice import DiceLoss
from .focal import FocalLoss
from .panoptic.common import (
    DeepLabCE,
    DirichletPrior,
)
from .panoptic.discriminative import DiscriminativeLoss
from .panoptic.discriminative_mixture import DiscriminativeMixtureLoss
from .panoptic.prototype import PrototypeInstanceLoss
from .embedding.CosineEmbeddingLoss import DiscriminativeCosineEmbeddingLoss
from .embedding.ContrastiveEmbeddingLoss import ContrastiveEmbeddingLoss


def get_loss(name: Optional[str] = None, num_classes=19, dm=None, **kwargs):
    if "counts" in kwargs:
        weights = torch.Tensor(kwargs["counts"])
        weights = weights.sum() / weights
        return nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    elif name == "Dice":
        return DiceLoss(ignore_index=255, num_classes=kwargs["num_classes"])
    elif name == "Focal":
        return FocalLoss(
            ignore_index=255, num_classes=kwargs["num_classes"], alpha=kwargs["alpha"]
        )
    elif name == "hard_pixel_mining":
        return DeepLabCE(**kwargs["args"])
    elif name == "dirichlet_prior":
        return DirichletPrior(num_classes=num_classes, dm=dm, **kwargs["args"])
    elif name == "mse":
        return nn.MSELoss(**kwargs["args"])
    elif name == "l1":
        return nn.L1Loss(**kwargs["args"])
    elif name == "discriminative":
        return DiscriminativeLoss(**kwargs["args"])
    elif name == "discriminative_mixture":
        return DiscriminativeMixtureLoss(**kwargs["args"])
    elif name == "prototype_instance":
        return PrototypeInstanceLoss(**kwargs["args"])
    elif name == "nll":
        return nn.NLLLoss(**kwargs["args"])
    elif name == "mask2former":
        from models.mask2former.modeling import SetCriterion

        return SetCriterion(**kwargs["args"], num_classes=num_classes, dm=dm)
    elif name == "DiscrimitativCosineEmbedding":
        return DiscriminativeCosineEmbeddingLoss(**kwargs["args"])
    elif name == "ContrastiveEmbedding":
        return ContrastiveEmbeddingLoss(**kwargs["args"])
    else:
        return nn.CrossEntropyLoss(ignore_index=255)


# Version of the CE loss that ignores all other arguments to forward()
class CELoss(nn.CrossEntropyLoss):
    def forward(
        self, input: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return super().forward(input, target)
