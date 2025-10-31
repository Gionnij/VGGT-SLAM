import os
from typing import Dict, Optional

from ..backbone import resnet as resnet
from ..backbone.mobilenet import mobilenet_v2

from .panoptic_deeplab import PanopticDeepLab


def get_backbone(
    name: str,
    output_stride: int,
    pretrained_backbone: bool,
    backbone_file: Optional[str],
    replace_stride_with_dilation=None,
    **kwargs,
):
    backbone_file = (
        None
        if backbone_file is None or backbone_file in ["None", "none", "null"]
        else os.path.expanduser(backbone_file)
    )
    if name.startswith("mobilenet"):
        return mobilenet_v2(
            pretrained=pretrained_backbone,
            output_stride=output_stride,
            backbone_file=backbone_file,
            return_all_intermediaries=True,
            **kwargs,
        )

    if replace_stride_with_dilation is None:
        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
        elif output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        elif output_stride == 32:
            replace_stride_with_dilation = [False, False, False]
        else:
            raise Exception(f"Output stride of {output_stride} not supported")
    if backbone_file == "None":
        backbone_file = None
    backbone = resnet.__dict__[name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation,
        backbone_file=backbone_file,
        return_all_intermediaries=True,
        **kwargs,
    )
    return backbone


def panoptic_deeplab(backbone: Dict, args: Dict, **kwargs):
    backbone = get_backbone(**backbone)
    model = PanopticDeepLab(backbone=backbone, **args)
    return model
