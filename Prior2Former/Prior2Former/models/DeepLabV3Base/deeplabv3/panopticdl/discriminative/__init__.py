from typing import Dict

from .. import get_backbone

from .discriminative_deeplab import DiscriminativeDeepLab


def discriminative_deeplab(backbone: Dict, args: Dict, **kwargs):
    backbone = get_backbone(**backbone)
    model = DiscriminativeDeepLab(backbone=backbone, **args)
    return model
