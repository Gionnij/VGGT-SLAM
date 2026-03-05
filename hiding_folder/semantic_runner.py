import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .semantic_head import SemanticHead

_SEM_REGISTRY: Dict[str, SemanticHead] = {}
_FUSION_REGISTRY: Dict[str, torch.nn.Module] = {}
_SEM_CONFIG_WARNED = False
_FUSION_WARNED = False
_FUSION_MODULE = None


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


def _load_fusion_module():
    global _FUSION_MODULE
    if _FUSION_MODULE is not None:
        return _FUSION_MODULE
    default_path = Path(__file__).resolve().parents[1] / "scripts" / "train_film_m2f_optimized_png.py"
    script_path = Path(os.getenv("VGGT_FUSION_SCRIPT", str(default_path))).expanduser()
    if not script_path.is_file():
        raise FileNotFoundError(f"Fusion training script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("train_film_m2f_optimized_png_live", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load fusion module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _FUSION_MODULE = mod
    return mod


def _extract_state_dict(state_obj):
    if isinstance(state_obj, dict) and "model_state" in state_obj and isinstance(state_obj["model_state"], dict):
        return state_obj["model_state"]
    if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
        return state_obj["state_dict"]
    if isinstance(state_obj, dict) and "model" in state_obj and isinstance(state_obj["model"], dict):
        return state_obj["model"]
    if isinstance(state_obj, dict):
        return state_obj
    return {}


def _load_fusion_checkpoint(model_obj: torch.nn.Module, ckpt_path: str) -> None:
    p = Path(ckpt_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Fusion checkpoint not found: {p}")
    if p.suffix.lower() not in (".pt", ".pth", ".ckpt"):
        return
    raw = torch.load(str(p), map_location="cpu")
    state = _extract_state_dict(raw)
    remap = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        if isinstance(nk, str) and (nk.startswith("fusion.") or nk.startswith("sem_head.")):
            remap[nk] = v
    if not remap:
        raise RuntimeError(f"No fusion/semantic keys found in checkpoint: {p}")
    missing, unexpected = model_obj.load_state_dict(remap, strict=False)
    loaded = max(0, len(remap) - len(unexpected))
    print(
        f"[SEM-FUSION] Loaded checkpoint {p.name}: loaded={loaded} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def _get_fusion_model(device: torch.device) -> torch.nn.Module:
    device_id = getattr(device, "index", None)
    cfg = os.getenv("VGGT_M2F_CFG", "")
    weights = os.getenv("VGGT_M2F_WEIGHTS", "")
    num_classes = _maybe_int(os.getenv("VGGT_SEM_CLASSES")) or 60
    key = f"{device.type}:{device_id}|{cfg}|{weights}|{num_classes}"
    mdl = _FUSION_REGISTRY.get(key)
    if mdl is not None:
        return mdl

    mod = _load_fusion_module()
    mdl = mod.FusionMask2Former(
        device=device,
        num_classes=num_classes,
        config_path=cfg if cfg else None,
        weights_path=weights if weights else None,
    )
    if weights:
        _load_fusion_checkpoint(mdl, weights)
    mdl.eval()
    _FUSION_REGISTRY[key] = mdl
    return mdl


def _select_frames(seq_len: int) -> Optional[List[int]]:
    S = int(seq_len)
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


def _normalize_film_pyramid(film_pyramid) -> Optional[List[torch.Tensor]]:
    if not isinstance(film_pyramid, (list, tuple)) or len(film_pyramid) < 4:
        return None
    out: List[torch.Tensor] = []
    for lvl in film_pyramid[:4]:
        if lvl is None or not isinstance(lvl, torch.Tensor):
            return None
        if lvl.dim() == 4:
            lvl = lvl.unsqueeze(1)
        if lvl.dim() != 5:
            return None
        out.append(lvl)
    return out


def _align_dino_temporal(dino_seq: torch.Tensor, target_s: int) -> torch.Tensor:
    if dino_seq.shape[1] == target_s:
        return dino_seq
    idx = torch.linspace(
        0, dino_seq.shape[1] - 1, steps=target_s, device=dino_seq.device, dtype=torch.float32
    ).round().long()
    return dino_seq.index_select(1, idx)


@torch.no_grad()
def run_semantic_if_enabled(model, predictions: dict, device: torch.device) -> None:
    """
    Optionally attach semantic predictions into the predictions dict.
    Controlled via env `VGGT_SEMANTIC_HEAD=1`.
    """
    if os.getenv("VGGT_SEMANTIC_HEAD", "0") != "1":
        return

    global _SEM_CONFIG_WARNED
    cfg_path = os.getenv("VGGT_M2F_CFG")
    weights_path = os.getenv("VGGT_M2F_WEIGHTS")
    if not cfg_path or not weights_path:
        if not _SEM_CONFIG_WARNED:
            print(
                "[SEM] Disabled for this run: missing VGGT_M2F_CFG/VGGT_M2F_WEIGHTS. "
                "Refusing to run with uninitialized semantic weights."
            )
            _SEM_CONFIG_WARNED = True
        return

    images = predictions.get("images")
    if images is None or images.ndim != 5:
        return

    backend = os.getenv("VGGT_SEM_BACKEND", "fusion").strip().lower()
    film_pyramid = _normalize_film_pyramid(predictions.get("film_pyramid"))
    if film_pyramid is None:
        return

    if backend == "head":
        frame_indices = _select_frames(images.shape[1])
        if not frame_indices:
            return
        head = _get_semantic_head(device)
        cls_logits, mask_logits, semantic_maps = head(
            images,
            frame_indices=frame_indices,
            film_pyramid=film_pyramid,
        )
        predictions["sem_frame_indices"] = frame_indices
        predictions["sem_cls_logits"] = cls_logits
        predictions["sem_mask_logits"] = mask_logits
        predictions["semantic_maps"] = semantic_maps
        return

    global _FUSION_WARNED
    tapper = getattr(model, "_feature_tapper", None)
    if tapper is None or not hasattr(tapper, "extract_dino_fmap"):
        if not _FUSION_WARNED:
            print("[SEM-FUSION] Missing feature tapper; cannot extract live DINO fmap.")
            _FUSION_WARNED = True
        return

    dino_seq = tapper.extract_dino_fmap(tuple(images.shape))
    if dino_seq is None or dino_seq.ndim != 5:
        if not _FUSION_WARNED:
            print("[SEM-FUSION] Unable to extract DINO fmap from current batch.")
            _FUSION_WARNED = True
        return

    B = film_pyramid[0].shape[0]
    S_film = film_pyramid[0].shape[1]
    dino_seq = _align_dino_temporal(dino_seq, S_film)
    frame_indices = _select_frames(S_film)
    if not frame_indices:
        return

    idx = torch.as_tensor(frame_indices, dtype=torch.long, device=dino_seq.device)
    dino_sel = dino_seq.index_select(1, idx).reshape(-1, *dino_seq.shape[-3:])
    dpt_sel = [lvl.index_select(1, idx).reshape(-1, *lvl.shape[-3:]) for lvl in film_pyramid]

    fusion = _get_fusion_model(device)
    H, W = int(images.shape[-2]), int(images.shape[-1])
    cls_logits, mask_logits = fusion(dino_sel, dpt_sel, label_shape=(H, W))

    if cls_logits.dim() == 4 and cls_logits.shape[1] == 1:
        cls_logits = cls_logits[:, 0]
    if mask_logits.dim() == 5 and mask_logits.shape[1] == 1:
        mask_logits = mask_logits[:, 0]

    cls_logits = cls_logits.reshape(B, len(frame_indices), *cls_logits.shape[-2:])
    mask_logits = mask_logits.reshape(B, len(frame_indices), *mask_logits.shape[-3:])

    predictions["sem_frame_indices"] = frame_indices
    predictions["sem_cls_logits"] = cls_logits
    predictions["sem_mask_logits"] = mask_logits
    predictions["semantic_maps"] = None
