from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from detectron2.config import get_cfg
    from detectron2.layers import ShapeSpec
    from detectron2.modeling import build_sem_seg_head
    from mask2former import add_maskformer2_config
except Exception:  # pragma: no cover - optional dependency
    get_cfg = None  # type: ignore[assignment]
    ShapeSpec = None  # type: ignore[assignment]
    build_sem_seg_head = None  # type: ignore[assignment]
    add_maskformer2_config = None  # type: ignore[assignment]


class SemanticHead(nn.Module):
    """
    Runs Detectron2's Mask2Former head directly on FiLM fused feature maps.
    """

    _DEFAULT_CFG = (
        Path(__file__).resolve().parents[1]
        / "mask2former"
        / "configs"
        / "coco"
        / "panoptic-segmentation"
        / "maskformer2_R50_bs16_50ep.yaml"
    )

    def __init__(
        self,
        *,
        device: torch.device,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        num_classes: Optional[int] = None,
        num_queries: Optional[int] = None,
    ) -> None:
        super().__init__()
        if get_cfg is None or build_sem_seg_head is None or add_maskformer2_config is None or ShapeSpec is None:
            raise ImportError(
                "detectron2/mask2former dependencies are missing. "
                "Ensure the mask2former submodule and detectron2 are on PYTHONPATH."
            )

        cfg = get_cfg()
        add_maskformer2_config(cfg)
        # Allow config keys not present in older detectron2 builds (e.g., STEM_TYPE).
        if hasattr(cfg, "set_new_allowed"):
            cfg.set_new_allowed(True)
        cfg_path = Path(config_path).expanduser() if config_path else self._DEFAULT_CFG
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Mask2Former config not found at {cfg_path}")
        cfg.merge_from_file(str(cfg_path))

        if num_classes is not None:
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(num_classes)
        if num_queries is not None:
            cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = int(num_queries)
        cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.DEVICE = "cuda" if device.type == "cuda" else "cpu"
        cfg.freeze()

        self._input_shapes: Dict[str, ShapeSpec] = {
            "res2": ShapeSpec(channels=256, stride=8),
            "res3": ShapeSpec(channels=512, stride=16),
            "res4": ShapeSpec(channels=1024, stride=32),
            "res5": ShapeSpec(channels=1024, stride=64),
        }
        self.device = device
        self.head = build_sem_seg_head(cfg, self._input_shapes).to(device)
        self.head.eval()
        if weights_path:
            self._load_head_weights(weights_path)

        self.scales = [8, 16, 32, 64]

    def _load_head_weights(self, weights_path: str) -> None:
        path = Path(weights_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Mask2Former weights not found at {path}")
        try:
            state = torch.load(path, map_location="cpu")
        except (RuntimeError, pickle.UnpicklingError):
            with path.open("rb") as f:
                state = pickle.load(f)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        head_state = {k.replace("sem_seg_head.", "", 1): v for k, v in state.items() if k.startswith("sem_seg_head.")}
        missing, unexpected = self.head.load_state_dict(head_state, strict=False)
        if missing:
            print(f"[SEM] Missing weights for keys: {missing}")
        if unexpected:
            print(f"[SEM] Unexpected weight keys ignored: {unexpected}")

    def forward(
        self,
        images: torch.Tensor,
        *,
        frame_indices: Sequence[int],
        film_pyramid: Sequence[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"Expected tensor of shape (B, S, 3, H, W), got {tuple(images.shape)}")
        if not frame_indices:
            raise ValueError("At least one frame index is required for semantic inference.")
        if len(film_pyramid) < 4 or any(level is None for level in film_pyramid[:4]):
            raise ValueError("FiLM pyramid is missing levels; ensure FiLM is enabled.")

        B = images.shape[0]
        H, W = images.shape[-2:]
        features, batch, num_frames = self._build_feature_dict(film_pyramid, frame_indices, target_hw=(H, W))
        outputs = self.head(features)

        cls_logits = outputs["pred_logits"]  # (B*num_frames, Q, K)
        mask_logits = outputs["pred_masks"]  # (B*num_frames, Q, h, w) low-res mask logits

        cls_logits = cls_logits.reshape(batch, num_frames, *cls_logits.shape[1:])
        mask_logits = mask_logits.reshape(batch, num_frames, *mask_logits.shape[1:])

        return cls_logits, mask_logits, None

    def _build_feature_dict(
        self,
        film_pyramid: Sequence[Optional[torch.Tensor]],
        frame_indices: Sequence[int],
        target_hw: Tuple[int, int],
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        # Allow inputs shaped either [B,S,C,H,W] or [B,C,H,W] (per-frame). If the latter, add the frame dim.
        norm_pyramid: List[torch.Tensor] = []
        for lvl in film_pyramid:
            if lvl is None:
                raise ValueError("FiLM pyramid contains None; cannot build feature dict.")
            if lvl.dim() == 4:
                lvl = lvl.unsqueeze(1)  # -> [B,1,C,H,W]
            norm_pyramid.append(lvl)

        idx_tensor = torch.as_tensor(frame_indices, dtype=torch.long, device=norm_pyramid[0].device)
        num_frames = int(idx_tensor.numel())
        B = norm_pyramid[0].shape[0]
        features: Dict[str, torch.Tensor] = {}
        keys = ["res2", "res3", "res4", "res5"]
        for key, level, stride in zip(keys, norm_pyramid[:4], self.scales):
            selected = level.index_select(1, idx_tensor)
            selected = selected.reshape(B * num_frames, *selected.shape[2:])
            size = (
                max(1, target_hw[0] // stride),
                max(1, target_hw[1] // stride),
            )
            resized = F.interpolate(selected, size=size, mode="bilinear", align_corners=False)
            features[key] = resized.to(self.device, dtype=torch.float32)
        return features, B, num_frames

    def _semantic_from_logits(self, cls_logits: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        class_probs = cls_logits.softmax(dim=-1)[..., :-1]
        mask_probs = mask_logits.sigmoid()
        seg_scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        return seg_scores.argmax(dim=1).to(torch.int64)
