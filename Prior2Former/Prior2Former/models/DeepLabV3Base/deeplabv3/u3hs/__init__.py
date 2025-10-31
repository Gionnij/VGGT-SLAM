from typing import Dict

from ..panopticdl import get_backbone
from .protypicaldl import PrototypicalDeepLab


def u3hs(backbone: Dict, args: Dict, dm, **kwargs):
    backbone = get_backbone(**backbone)
    model = PrototypicalDeepLab(
        backbone=backbone,
        thing_classes=dm.mapped_thing_list,
        stuff_classes=dm.mapped_stuff_list,
        dm=dm,
        **args
    )
    return model
