"""
trace_hooks.py

Registers lightweight hooks on VGGT to log the concrete frame indices flowing
into the DINO and DPT branches. Hooks consult attributes populated by
Solver.run_predictions (TraceSink, step id, frame ids, and window tensor).
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import types

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
    # Wrap aggregator patch embed to capture DINO indices.
    agg = getattr(model, "aggregator", None)
    patch_embed = getattr(agg, "patch_embed", None) if agg is not None else None
    if patch_embed is not None and not getattr(patch_embed, "_trace_wrapped", False):
        orig_forward = patch_embed.forward

        def _wrapped_patch(self, *args, _orig=orig_forward, **kwargs):
            x = args[0] if args else kwargs.get("x")
            if isinstance(x, torch.Tensor):
                _log_indices(model, "dino_indices", x)
            return _orig(*args, **kwargs)

        patch_embed.forward = types.MethodType(_wrapped_patch, patch_embed)
        patch_embed._trace_wrapped = True

    # Wrap depth head forward to capture DPT indices.
    depth_head = getattr(model, "depth_head", None)
    if depth_head is not None and not getattr(depth_head, "_trace_wrapped", False):
        orig_forward = depth_head.forward

        def _wrapped_depth(self, *args, _orig=orig_forward, **kwargs):
            x = args[0] if args else kwargs.get("x")
            if isinstance(x, torch.Tensor):
                _log_indices(model, "dpt_indices", x)
            return _orig(*args, **kwargs)

        depth_head.forward = types.MethodType(_wrapped_depth, depth_head)
        depth_head._trace_wrapped = True

    setattr(model, "_trace_wrapped", True)
    return []
