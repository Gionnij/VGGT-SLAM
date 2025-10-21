# normalize_pyramid.py
import torch.nn as nn
import torch.nn.functional as F

def normalize_to_canonical_pyramid(res_feats: dict, H: int, W: int):
    """
    Rescale depth backbone outputs to a canonical pyramid p2..p5
    with consistent channel dimensions (256).
    """
    return {
        "p2": F.interpolate(res_feats["res2"], size=(H//8,  W//8),  mode="bilinear", align_corners=False),
        "p3": F.interpolate(res_feats["res3"], size=(H//16, W//16), mode="bilinear", align_corners=False),
        "p4": F.interpolate(res_feats["res4"], size=(H//32, W//32), mode="bilinear", align_corners=False),
        "p5": F.interpolate(res_feats["res5"], size=(H//64, W//64), mode="bilinear", align_corners=False),
    }
