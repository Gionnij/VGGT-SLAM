"""
trace_hooks.py

Registers lightweight hooks on VGGT to log the concrete frame indices flowing
into the DINO and DPT branches. Hooks consult attributes populated by
Solver.run_predictions (TraceSink, step id, frame ids, and window tensor).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn


def _match_indices(batch: torch.Tensor, source: torch.Tensor) -> Optional[List[int]]:
    """
    Compute, for each row in *batch*, the index of the closest matching frame
    in *source* using an L2 distance. Assumes both tensors share (C,H,W).
    """
    if batch.dim() < 4 or source.dim() < 4:
        return None
    if batch.shape[1:] != source.shape[1:]:
        return None
    if batch.shape[0] == 0 or source.shape[0] == 0:
        return None

    # Flatten to (B, C*H*W) and compute pairwise distances.
    flat_batch = batch.detach().reshape(batch.shape[0], -1).float()
    flat_source = source.detach().reshape(source.shape[0], -1).float()
    if flat_source.device != flat_batch.device:
        flat_source = flat_source.to(flat_batch.device)

    # torch.cdist expects finite values; guard against NaNs.
    if not torch.isfinite(flat_batch).all() or not torch.isfinite(flat_source).all():
        return None

    dists = torch.cdist(flat_batch, flat_source, p=2)
    if dists.numel() == 0:
        return None
    indices = torch.argmin(dists, dim=1)
    return indices.detach().cpu().tolist()


def _log_indices(
    model: nn.Module,
    kind: str,
    batch: torch.Tensor,
) -> None:
    sink = getattr(model, "_trace_sink", None)
    if sink is None:
        return

    window_tensor = getattr(model, "_trace_window_tensor", None)
    if window_tensor is None:
        return

    window_len = getattr(model, "_trace_window_length", None)
    frame_ids_window = getattr(model, "_trace_frame_ids", None)
    step = getattr(model, "_trace_step", None)

    # Restrict to the sliding-window portion if available
    if isinstance(window_len, int) and window_len >= 0:
        window_tensor = window_tensor[:window_len]
        if isinstance(frame_ids_window, Sequence):
            frame_ids_window = list(frame_ids_window)[:window_len]

    indices = _match_indices(batch, window_tensor)
    if indices is None:
        return

    trace_state = dict(getattr(model, "_trace_state", {}))
    if trace_state.get(kind) == step:
        return
    trace_state[kind] = step
    model._trace_state = trace_state

    try:
        sink.log(
            step=step if step is not None else -1,
            kind=kind,
            indices=indices,
            note=frame_ids_window,
        )
    except Exception:
        # Logging should never break the forward path.
        pass


def install_trace_probes(model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Attach pre-forward hooks that log frame indices entering the DINO and DPT
    pipelines. Returns a list of handles so callers can remove them if needed.
    """
    modmap = {name: module for name, module in model.named_modules()}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    # DINO branch: target the patch embedding pre-hook (closest to image input).
    dino_candidates = [
        "aggregator.patch_embed.patch_embed",
        "aggregator.patch_embed.proj",
    ]
    for name in dino_candidates:
        if name not in modmap:
            continue

        def _dino_hook(module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                _log_indices(model, "dino_indices", x)

        handles.append(modmap[name].register_forward_pre_hook(_dino_hook))
        break

    # DPT branch: best effort to catch the image tensor before entering depth head.
    dpt_candidates = [
        "depth_head.backbone",
        "depth_head.encoder",
        "depth_head.model",
    ]
    for name in dpt_candidates:
        if name not in modmap:
            continue

        def _dpt_hook(module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                _log_indices(model, "dpt_indices", x)

        handles.append(modmap[name].register_forward_pre_hook(_dpt_hook))
        break

    if handles:
        combined = list(getattr(model, "_trace_handles", []))
        combined.extend(handles)
        model._trace_handles = combined

    return handles
