import os
from typing import List, Optional

import torch

from .semantic_head import SemanticHead

_SEM_REGISTRY = {"head": None}


def _resolve_dtype(device: torch.device) -> torch.dtype:
    precision = os.getenv("VGGT_M2F_PRECISION", "fp32").lower()
    if precision in {"fp16", "half"} and device.type == "cuda":
        return torch.float16
    return torch.float32


def _get_semantic_head(device: torch.device) -> SemanticHead:
    head = _SEM_REGISTRY.get("head")
    if head is None:
        model_id = os.getenv("VGGT_M2F_MODEL_ID", "facebook/mask2former-swin-base-ade-semantic")
        trust_remote = os.getenv("VGGT_M2F_TRUST_REMOTE_CODE", "0") == "1"
        head = SemanticHead(
            model_id=model_id,
            device=device,
            torch_dtype=_resolve_dtype(device),
            trust_remote_code=trust_remote,
        )
        head.eval()
        _SEM_REGISTRY["head"] = head
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

    images_subset = images[:, frame_indices, ...]
    head = _get_semantic_head(device)
    cls_logits, mask_logits, semantic_maps = head(images_subset)

    predictions["sem_frame_indices"] = frame_indices
    predictions["sem_cls_logits"] = cls_logits
    predictions["sem_mask_logits"] = mask_logits
    predictions["semantic_maps"] = semantic_maps
