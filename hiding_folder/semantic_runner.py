import json
import os
import importlib.util
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .semantic_head import SemanticHead

_SEM_REGISTRY: Dict[str, SemanticHead] = {}
_FUSION_REGISTRY: Dict[str, torch.nn.Module] = {}
_SEM_CONFIG_WARNED = False
_FUSION_WARNED = False
_FUSION_MODULE = None
_SEM_PYRAMID_WARNED = False
_SEM_EXPORT_WARNED = False
_SEM_EXPORT_COUNTER = 0
_SEM_SOURCE_LOGGED = False


def _maybe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _truthy(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _to_disk_dtype(x: torch.Tensor, dtype_name: str) -> torch.Tensor:
    n = str(dtype_name).strip().lower()
    if n in ("fp16", "float16", "half"):
        return x.to(dtype=torch.float16)
    if n in ("bf16", "bfloat16"):
        return x.to(dtype=torch.bfloat16)
    if n in ("fp32", "float32", "full"):
        return x.to(dtype=torch.float32)
    return x.to(dtype=torch.float16)


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


def _normalize_pyramid(pyramid) -> Optional[List[torch.Tensor]]:
    if not isinstance(pyramid, (list, tuple)) or len(pyramid) < 4:
        return None
    out: List[torch.Tensor] = []
    for lvl in pyramid[:4]:
        if lvl is None or not isinstance(lvl, torch.Tensor):
            return None
        if lvl.dim() == 4:
            lvl = lvl.unsqueeze(1)
        if lvl.dim() != 5:
            return None
        out.append(lvl)
    return out


def _pick_dpt_source(predictions: Dict[str, Any]) -> Tuple[Optional[List[torch.Tensor]], str]:
    """
    Select which DPT pyramid feeds semantic inference.
    VGGT_SEM_DPT_SOURCE:
      - raw   -> predictions["pyramid"]   (matches offline export/training pipeline)
      - film  -> predictions["film_pyramid"]
      - auto  -> raw if present else film
    """
    mode = os.getenv("VGGT_SEM_DPT_SOURCE", "raw").strip().lower()
    if mode not in ("raw", "film", "auto"):
        mode = "raw"

    raw = _normalize_pyramid(predictions.get("pyramid"))
    film = _normalize_pyramid(predictions.get("film_pyramid"))

    if mode == "raw":
        return raw, "raw"
    if mode == "film":
        return film, "film"
    if raw is not None:
        return raw, "raw"
    return film, "film"


def _align_dino_temporal(dino_seq: torch.Tensor, target_s: int) -> torch.Tensor:
    if dino_seq.shape[1] == target_s:
        return dino_seq
    idx = torch.linspace(
        0, dino_seq.shape[1] - 1, steps=target_s, device=dino_seq.device, dtype=torch.float32
    ).round().long()
    return dino_seq.index_select(1, idx)


def _maybe_export_sem_inputs(
    *,
    predictions: Dict[str, Any],
    dino_sel: torch.Tensor,
    dpt_sel: List[torch.Tensor],
    frame_indices: List[int],
    dpt_source: str,
    film_selected: Optional[List[torch.Tensor]] = None,
) -> None:
    global _SEM_EXPORT_WARNED, _SEM_EXPORT_COUNTER
    if not _truthy(os.getenv("VGGT_SEM_EXPORT_EMBEDDINGS"), default=False):
        return

    every = _maybe_int(os.getenv("VGGT_SEM_EXPORT_EVERY")) or 1
    every = max(1, int(every))
    _SEM_EXPORT_COUNTER += 1
    if (_SEM_EXPORT_COUNTER % every) != 0:
        return

    # Prefer explicit export dir; otherwise place under current demo run.
    root = os.getenv("VGGT_SEM_EXPORT_DIR", "").strip()
    if not root:
        demo_run = os.getenv("VGGT_ACTIVE_DEMO_RUN_DIR", "").strip()
        root = str(Path(demo_run) / "semantic_inputs") if demo_run else "debug/semantic_inputs"
    out_dir = Path(root).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dump_dtype = os.getenv("VGGT_SEM_EXPORT_DTYPE", "float16")
    fmt = os.getenv("VGGT_SEM_EXPORT_FORMAT", "pt").strip().lower()
    export_film = _truthy(os.getenv("VGGT_SEM_EXPORT_INCLUDE_FILM"), default=False)

    step_id = predictions.get("_sem_step_id", -1)
    frame_ids = predictions.get("_sem_frame_ids", []) or []
    ts = time.time()
    stem = f"sem_inputs_step{int(step_id):08d}_n{_SEM_EXPORT_COUNTER:06d}"

    # semantic runner consumes selected frames; export in [B,S,C,H,W] shape.
    # Keep B inferred from selected tensors to avoid assumptions.
    selected = max(1, len(frame_indices))
    b = max(1, int(dino_sel.shape[0] // selected))
    dino_out = _to_disk_dtype(
        dino_sel.reshape(b, selected, *dino_sel.shape[-3:]).detach().cpu().contiguous(),
        dump_dtype,
    )
    dpt_out = [
        _to_disk_dtype(
            lvl.reshape(b, selected, *lvl.shape[-3:]).detach().cpu().contiguous(),
            dump_dtype,
        )
        for lvl in dpt_sel
    ]

    payload: Dict[str, Any] = {
        "dino_features": dino_out,
        "dpt_pyramid": dpt_out,
        "meta": {
            "step_id": int(step_id),
            "timestamp": float(ts),
            "frame_indices": [int(i) for i in frame_indices],
            "frame_ids_window": [str(x) for x in frame_ids],
            "dpt_source": dpt_source,
            "dtype": str(dino_out.dtype).replace("torch.", ""),
            "shapes": {
                "dino_features": list(dino_out.shape),
                "dpt_pyramid": [list(x.shape) for x in dpt_out],
            },
        },
    }

    if export_film and film_selected is not None:
        film_out = [
            _to_disk_dtype(
                lvl.detach().cpu().contiguous(),
                dump_dtype,
            )
            for lvl in film_selected
        ]
        payload["film_pyramid_selected"] = film_out
        payload["meta"]["shapes"]["film_pyramid_selected"] = [list(x.shape) for x in film_out]

    out_path = out_dir / f"{stem}.pt"
    if fmt == "safetensors":
        try:
            from safetensors.torch import save_file

            tensor_map: Dict[str, torch.Tensor] = {
                "dino_features": payload["dino_features"],
            }
            for i, lvl in enumerate(payload["dpt_pyramid"]):
                tensor_map[f"dpt_pyramid_{i}"] = lvl
            if "film_pyramid_selected" in payload:
                for i, lvl in enumerate(payload["film_pyramid_selected"]):
                    tensor_map[f"film_pyramid_selected_{i}"] = lvl
            out_path = out_dir / f"{stem}.safetensors"
            save_file(tensor_map, str(out_path), metadata={"meta_json": json.dumps(payload["meta"])})
            (out_dir / f"{stem}.json").write_text(json.dumps(payload["meta"], indent=2), encoding="utf-8")
        except Exception as exc:
            if not _SEM_EXPORT_WARNED:
                print(f"[SEM export][WARN] safetensors export failed, falling back to .pt: {exc}")
                _SEM_EXPORT_WARNED = True
            torch.save(payload, out_path)
    else:
        torch.save(payload, out_path)

    print(f"[SEM export] wrote semantic inputs to {out_path}")


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
    dpt_pyramid, dpt_source = _pick_dpt_source(predictions)
    if dpt_pyramid is None:
        global _SEM_PYRAMID_WARNED
        if not _SEM_PYRAMID_WARNED:
            mode = os.getenv("VGGT_SEM_DPT_SOURCE", "raw")
            print(f"[SEM][WARN] DPT pyramid unavailable for VGGT_SEM_DPT_SOURCE={mode}")
            _SEM_PYRAMID_WARNED = True
        return

    global _SEM_SOURCE_LOGGED
    if not _SEM_SOURCE_LOGGED:
        print(f"[SEM] DPT source for semantic backend '{backend}': {dpt_source}")
        _SEM_SOURCE_LOGGED = True

    if backend == "head":
        frame_indices = _select_frames(images.shape[1])
        if not frame_indices:
            return
        head = _get_semantic_head(device)
        cls_logits, mask_logits, semantic_maps = head(
            images,
            frame_indices=frame_indices,
            film_pyramid=dpt_pyramid,
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

    B = dpt_pyramid[0].shape[0]
    S_film = dpt_pyramid[0].shape[1]
    dino_seq = _align_dino_temporal(dino_seq, S_film)
    frame_indices = _select_frames(S_film)
    if not frame_indices:
        return

    idx = torch.as_tensor(frame_indices, dtype=torch.long, device=dino_seq.device)
    dino_sel = dino_seq.index_select(1, idx).reshape(-1, *dino_seq.shape[-3:])
    dpt_sel = [lvl.index_select(1, idx).reshape(-1, *lvl.shape[-3:]) for lvl in dpt_pyramid]
    film_sel = None
    if _truthy(os.getenv("VGGT_SEM_EXPORT_INCLUDE_FILM"), default=False):
        fp = _normalize_pyramid(predictions.get("film_pyramid"))
        if fp is not None:
            film_sel = [lvl.index_select(1, idx).reshape(-1, *lvl.shape[-3:]) for lvl in fp]

    _maybe_export_sem_inputs(
        predictions=predictions,
        dino_sel=dino_sel,
        dpt_sel=dpt_sel,
        frame_indices=frame_indices,
        dpt_source=dpt_source,
        film_selected=film_sel,
    )

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
