import os
from typing import List, Optional

import torch

from .semantic_head import SemanticHead

_SEM_REGISTRY = {"head": None}


def _get_semantic_head(device: torch.device) -> SemanticHead:
    head = _SEM_REGISTRY.get("head")
    if head is None:
        num_classes = int(os.getenv("VGGT_SEM_CLASSES", "20"))
        num_queries = int(os.getenv("VGGT_SEM_QUERIES", "50"))
        head = SemanticHead(num_classes=num_classes, num_queries=num_queries).to(device=device)
        head.eval()
        _SEM_REGISTRY["head"] = head
    return head


@torch.no_grad()
def run_semantic_if_enabled(model, predictions: dict, device: torch.device) -> None:
    """
    Optionally attach semantic predictions into the predictions dict.
    Controlled via env `VGGT_SEMANTIC_HEAD=1`.
    """
    if os.getenv("VGGT_SEMANTIC_HEAD", "0") != "1":
        return

    depth_head = getattr(model, "depth_head", None)
    if depth_head is None:
        return

    pyramid: Optional[List[torch.Tensor]] = getattr(depth_head, "film_side_pyramid", None)
    if pyramid is None:
        pyramid = getattr(depth_head, "raw_pyramid", None)

    if not pyramid or any(level is None for level in pyramid):
        return

    head = _get_semantic_head(device)
    cls_logits, mask_logits = head(pyramid)
    predictions["sem_cls_logits"] = cls_logits
    predictions["sem_mask_logits"] = mask_logits
