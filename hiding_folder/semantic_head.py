from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

if TYPE_CHECKING:
    import numpy as np


def _to_numpy_images(batch: torch.Tensor) -> Tuple[List["np.ndarray"], List[Tuple[int, int]]]:
    """
    Convert a (N, 3, H, W) float tensor in [0, 1] to a list of uint8 HWC numpy arrays.
    """
    if batch.ndim != 4:
        raise ValueError(f"Expected tensor of shape (N, 3, H, W), got {tuple(batch.shape)}")

    batch = batch.detach().cpu().clamp(0, 1)
    np_images: List["np.ndarray"] = []
    sizes: List[Tuple[int, int]] = []
    for img in batch:
        arr = (img * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
        np_images.append(arr)
        sizes.append((arr.shape[0], arr.shape[1]))
    return np_images, sizes


class SemanticHead(nn.Module):
    """
    Thin wrapper around Hugging Face's Mask2Former implementation.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        torch_dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.to(device)
        self._model_device = next(self.model.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images: tensor shaped (B, S, 3, H, W) in [0, 1].

        Returns:
            cls_logits: (B, S, Q, K)
            mask_logits: (B, S, Q, H/4, W/4)
            semantic_maps: (B, S, H, W) long tensor with per-pixel class ids
        """
        if images.ndim != 5:
            raise ValueError(f"Expected tensor of shape (B, S, 3, H, W), got {tuple(images.shape)}")

        B, S = images.shape[:2]
        if B == 0 or S == 0:
            raise ValueError("SemanticHead received an empty sequence of images.")

        flat = images.reshape(B * S, *images.shape[2:])
        np_images, target_sizes = _to_numpy_images(flat)

        encoded_inputs = self.processor(images=np_images, return_tensors="pt")
        encoded_inputs = {k: v.to(self._model_device) for k, v in encoded_inputs.items()}

        outputs = self.model(**encoded_inputs)
        cls_logits = outputs.class_queries_logits    # (B*S, Q, K)
        mask_logits = outputs.masks_queries_logits   # (B*S, Q, H/4, W/4)

        cls_logits = cls_logits.reshape(B, S, *cls_logits.shape[1:])
        mask_logits = mask_logits.reshape(B, S, *mask_logits.shape[1:])

        semantic_maps_list = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_sizes,
        )
        semantic_maps = torch.stack(semantic_maps_list, dim=0).reshape(B, S, *semantic_maps_list[0].shape)

        return cls_logits, mask_logits, semantic_maps
