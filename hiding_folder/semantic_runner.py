import os
from typing import Dict, List, Optional

import torch

from .semantic_head import SemanticHead

_SEM_REGISTRY: Dict[str, SemanticHead] = {}


def _maybe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_semantic_head(device: torch.device) -> SemanticHead:
    device_id = getattr(device, "index", None)
    key = f"{device.type}:{device_id}"
    head = _SEM_REGISTRY.get(key)
    if head is None:
        head = SemanticHead(
            device=device,
            config_path=os.getenv("VGGT_M2F_CFG"),
            weights_path=os.getenv("VGGT_M2F_WEIGHTS"),
            num_classes=_maybe_int(os.getenv("VGGT_SEM_CLASSES")),
            num_queries=_maybe_int(os.getenv("VGGT_SEM_QUERIES")),
        )
        head.eval()
        _SEM_REGISTRY[key] = head
    return head


def _select_frames(images: torch.Tensor) -> Optional[List[int]]:
    S = images.shape[1]
    if S == 0:
        return None

    idx_env = os.getenv("VGGT_SEM_FRAME_INDEX")
    if idx_env is not None:
        idx = int(idx_env)
        if idx < 0:
            idx += S
        idx = max(0, min(S - 1, idx))
        return [idx]

    mode = os.getenv("VGGT_SEM_FRAME_MODE", "last").lower()
    if mode == "all":
        return list(range(S))
    if mode == "first":
        return [0]
    return [S - 1]


@torch.no_grad()
def run_semantic_if_enabled(model, predictions: dict, device: torch.device) -> None:
    """
    Optionally attach semantic predictions into the predictions dict.
    Controlled via env `VGGT_SEMANTIC_HEAD=1`.
    """
    if os.getenv("VGGT_SEMANTIC_HEAD", "0") != "1":
        return

    images = predictions.get("images")
    if images is None or images.ndim != 5:
        return

    frame_indices = _select_frames(images)
    if not frame_indices:
        return

    head = _get_semantic_head(device)
    film_pyramid = predictions.get("film_pyramid")
    if not film_pyramid:
        return
    cls_logits, mask_logits, semantic_maps = head(
        images,
        frame_indices=frame_indices,
        film_pyramid=film_pyramid,
    )

    predictions["sem_frame_indices"] = frame_indices
    predictions["sem_cls_logits"] = cls_logits
    predictions["sem_mask_logits"] = mask_logits
    predictions["semantic_maps"] = semantic_maps
