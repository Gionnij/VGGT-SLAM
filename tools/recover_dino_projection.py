#!/usr/bin/env python3
"""
Recover the DINO 1x1 projection used during offline embedding export.

Why this exists:
- Export-time chunks store projected DINO features [B,S,256,Htok,Wtok].
- The projector that produced them is created lazily in export code and is not
  persisted by default.
- Live inference currently creates a fresh random projector, which can cause a
  feature-space mismatch against a model trained on the exported chunks.

This script replays VGGT on original chunk frames, taps raw DINO patch tokens,
and solves a linear regression:

    y = W x + b

where:
- x is raw tapped DINO channel vector (Cin),
- y is stored projected DINO vector (256),
- W,b are converted to Conv2d( Cin -> 256, kernel=1 ) weights.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from hiding_folder.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def _read_list_file(path_str: Optional[str]) -> Optional[List[str]]:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"List file not found: {path}")
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        out.append(item)
    return out or None


def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    out = [x.strip() for x in value.split(",") if x.strip()]
    return out or None


def _chunk_json_path(chunk_path: Path) -> Path:
    return chunk_path.with_suffix(".json")


def _load_chunk_metadata(chunk_path: Path) -> Dict:
    if chunk_path.suffix == ".safetensors":
        meta_path = _chunk_json_path(chunk_path)
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing sidecar JSON for {chunk_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    payload = torch.load(chunk_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected .pt chunk payload type for {chunk_path}: {type(payload)}")
    return payload


def _load_dino_tensor(chunk_path: Path) -> torch.Tensor:
    if chunk_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("safetensors is required to read .safetensors chunks") from exc
        tensors = load_file(str(chunk_path))
        if "dino_features" not in tensors:
            raise KeyError(f"'dino_features' missing in {chunk_path}")
        dino = tensors["dino_features"]
    else:
        payload = torch.load(chunk_path, map_location="cpu")
        if "dino_features" not in payload:
            raise KeyError(f"'dino_features' missing in {chunk_path}")
        dino = payload["dino_features"]
    if dino.dim() == 4:
        dino = dino.unsqueeze(0)
    if dino.dim() != 5:
        raise ValueError(f"Unexpected dino_features shape in {chunk_path}: {tuple(dino.shape)}")
    return dino.detach().cpu().float().contiguous()


def _iter_chunk_paths(
    dataset_root: Path,
    scenes: Sequence[str],
    chunk_format: str,
) -> Iterable[Path]:
    for scene in scenes:
        chunk_dir = dataset_root / scene / "chunks"
        if not chunk_dir.is_dir():
            continue
        pt_files = sorted(chunk_dir.glob("*.pt"))
        st_files = sorted(chunk_dir.glob("*.safetensors"))

        if chunk_format == "pt":
            chosen = pt_files
        elif chunk_format == "safetensors":
            chosen = st_files
        else:
            # auto: prefer safetensors if both exist for same stem
            st_stems = {p.stem for p in st_files}
            pt_only = [p for p in pt_files if p.stem not in st_stems]
            chosen = st_files + pt_only
        for p in chosen:
            yield p


def _resolve_frame_path(
    raw_path: str,
    *,
    dataset_root: Path,
    scene_id: str,
    scene_dir: Path,
    prefix_from: Optional[str],
    prefix_to: Optional[str],
) -> Optional[Path]:
    path_str = raw_path
    if prefix_from and prefix_to and raw_path.startswith(prefix_from):
        path_str = prefix_to + raw_path[len(prefix_from) :]

    p = Path(path_str).expanduser()
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(scene_dir / p)
        candidates.append(dataset_root / scene_id / p)
        candidates.append(scene_dir / "images" / p.name)
        candidates.append(dataset_root / scene_id / "images" / p.name)
        # Common ScanNet++ layout:
        candidates.append(scene_dir / "dslr" / "resized_undistorted_images" / p.name)

    for c in candidates:
        if c.is_file():
            return c
    return None


class RawDinoTap:
    def __init__(
        self,
        model: nn.Module,
        *,
        tap_candidates: Sequence[str],
        patch_size: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self._cache: Optional[torch.Tensor] = None
        self.tap_name: Optional[str] = None
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

        modmap = {n: m for n, m in model.named_modules()}
        for cand in tap_candidates:
            if cand in modmap:
                self.tap_name = cand
                self._handle = modmap[cand].register_forward_hook(self._hook)
                break
        if self._handle is None:
            raise KeyError(f"No DINO tap candidate found. Tried: {tap_candidates}")

    def _hook(self, _module, _inp, output):
        if torch.is_tensor(output):
            self._cache = output
        elif isinstance(output, (list, tuple)):
            for item in output:
                if torch.is_tensor(item):
                    self._cache = item
                    break

    @torch.no_grad()
    def extract_raw_fmap(self, batch_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError("DINO tap cache empty after forward.")
        B, S, H, W = batch_shape
        tokens = self._cache
        self._cache = None

        if tokens.dim() != 3:
            raise ValueError(f"Unexpected token shape: {tuple(tokens.shape)}")
        BS, N, Cin = tokens.shape
        if BS != B * S:
            raise ValueError(f"Token batch mismatch: got {BS}, expected {B*S}")

        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            raise ValueError(f"Input {H}x{W} not divisible by patch size={self.patch_size}")
        Htok = H // self.patch_size
        Wtok = W // self.patch_size
        expected = Htok * Wtok
        if N < expected:
            raise ValueError(f"Not enough tokens: have {N}, expected >= {expected}")

        patch_tokens = tokens[:, N - expected :, :]
        fmap = patch_tokens.transpose(1, 2).reshape(B, S, Cin, Htok, Wtok)
        return fmap.detach()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class LinearAccumulator:
    def __init__(self, in_dim: int, out_dim: int, ridge: float):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.ridge = float(ridge)
        d = self.in_dim + 1  # + bias
        self.A = torch.zeros((d, d), dtype=torch.float64)
        self.B = torch.zeros((d, self.out_dim), dtype=torch.float64)
        self.num_samples = 0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        # x: [N, in_dim], y: [N, out_dim]
        if x.numel() == 0:
            return
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        xa = torch.cat([x, ones], dim=1).to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)
        self.A += xa.T @ xa
        self.B += xa.T @ y
        self.num_samples += int(x.shape[0])

    def solve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Solve (A + lambda I) beta = B
        reg = torch.eye(self.A.shape[0], dtype=self.A.dtype)
        reg[-1, -1] = 0.0  # do not regularize bias
        lhs = self.A + self.ridge * reg
        beta = torch.linalg.solve(lhs, self.B)  # [in_dim+1, out_dim]
        w = beta[:-1, :].T.contiguous()  # [out_dim, in_dim]
        b = beta[-1, :].contiguous()     # [out_dim]
        return w.to(dtype=torch.float32), b.to(dtype=torch.float32)


@dataclass
class ChunkEval:
    scene_id: str
    chunk_path: str
    num_eval: int
    x_eval: torch.Tensor  # [N, Cin]
    y_eval: torch.Tensor  # [N, 256]


def _rmse(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> float:
    pred = x @ w.T + b
    err = pred - y
    return float(torch.sqrt(torch.mean(err * err)).item())


def _load_vggt(device: torch.device, checkpoint_path: Optional[str], allow_download: bool) -> VGGT:
    model = VGGT().to(device)

    ckpt: Optional[Path] = None
    if checkpoint_path:
        ckpt = Path(checkpoint_path).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    else:
        default_local = Path("~/models/VGGT-1B/model.pt").expanduser()
        if default_local.is_file():
            ckpt = default_local

    if ckpt is not None:
        state = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        if not allow_download:
            raise RuntimeError(
                "No local VGGT checkpoint found. Pass --checkpoint-path or use --allow-download."
            )
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state, strict=True)

    model.eval()

    # Mirror export_embeddings.py behavior.
    dh = getattr(model, "depth_head", None)
    if dh is not None:
        orig_forward = dh.forward

        def depth_forward_no_chunk(self, aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=None):
            return orig_forward(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_chunk_size=frames_chunk_size,
            )

        dh.forward = depth_forward_no_chunk.__get__(dh, type(dh))

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recover missing DINO projection from existing chunks.")
    p.add_argument("--dataset-root", required=True, help="Path to dataset_ready root.")
    p.add_argument("--scenes-file", help="Text file with one scene id per line.")
    p.add_argument("--scenes", help="Comma-separated scene ids (used if scenes-file omitted).")
    p.add_argument(
        "--chunk-format",
        choices=["auto", "pt", "safetensors"],
        default="auto",
        help="Chunk file format preference.",
    )
    p.add_argument("--max-chunks", type=int, default=120, help="Maximum chunks to process.")
    p.add_argument(
        "--samples-per-chunk",
        type=int,
        default=1024,
        help="Fit samples per chunk (patch vectors).",
    )
    p.add_argument(
        "--eval-samples-per-chunk",
        type=int,
        default=128,
        help="Held-out eval samples per chunk for diagnostics.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=60000,
        help="Global cap on fit samples.",
    )
    p.add_argument(
        "--max-frames-per-chunk",
        type=int,
        default=0,
        help="If >0, only use first N frames from each chunk.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument(
        "--tap-candidates",
        default="aggregator.patch_embed.blocks.23,aggregator.patch_embed.blocks.21,aggregator.patch_embed.blocks.15",
        help="Comma-separated tap module candidates in priority order.",
    )
    p.add_argument("--checkpoint-path", help="Local VGGT base checkpoint path.")
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading VGGT checkpoint if no local checkpoint is found.",
    )
    p.add_argument("--device", default="", help="cuda/cpu. Default: cuda if available.")
    p.add_argument(
        "--frame-prefix-from",
        default="",
        help="Optional source prefix for frame path remapping.",
    )
    p.add_argument(
        "--frame-prefix-to",
        default="",
        help="Optional target prefix for frame path remapping.",
    )
    p.add_argument(
        "--fit-per-scene",
        action="store_true",
        help="Also fit per-scene projectors (diagnostic for multi-projection exports).",
    )
    p.add_argument(
        "--min-scene-fit-samples",
        type=int,
        default=4096,
        help="Minimum fit samples to report a per-scene projector.",
    )
    p.add_argument(
        "--out-proj",
        required=True,
        help="Output .pt file for recovered global projector.",
    )
    p.add_argument("--out-report", default="", help="Output JSON report path.")
    p.add_argument(
        "--out-scene-dir",
        default="",
        help="If --fit-per-scene, save per-scene projector .pt files here.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset-root not found: {dataset_root}")

    scenes = _read_list_file(args.scenes_file) or _parse_csv_list(args.scenes)
    if not scenes:
        scenes = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])
    if not scenes:
        raise RuntimeError("No scenes to process.")

    chunk_paths = list(_iter_chunk_paths(dataset_root, scenes, args.chunk_format))
    if not chunk_paths:
        raise RuntimeError("No chunk files found.")
    if args.max_chunks > 0:
        chunk_paths = chunk_paths[: int(args.max_chunks)]

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tap_candidates = [x.strip() for x in args.tap_candidates.split(",") if x.strip()]
    if not tap_candidates:
        raise ValueError("tap-candidates cannot be empty.")

    model = _load_vggt(device, args.checkpoint_path, bool(args.allow_download))
    tap = RawDinoTap(model, tap_candidates=tap_candidates, patch_size=int(args.patch_size))

    prefix_from = args.frame_prefix_from.strip() or None
    prefix_to = args.frame_prefix_to.strip() or None
    if (prefix_from is None) != (prefix_to is None):
        raise ValueError("frame-prefix-from and frame-prefix-to must be set together.")

    fit_acc: Optional[LinearAccumulator] = None
    scene_acc: Dict[str, LinearAccumulator] = {}
    eval_chunks: List[ChunkEval] = []
    chunk_fit_counts: Dict[str, int] = {}
    chunk_warnings: List[str] = []

    max_samples = max(1, int(args.max_samples))
    samples_per_chunk = max(1, int(args.samples_per_chunk))
    eval_per_chunk = max(0, int(args.eval_samples_per_chunk))
    max_frames_per_chunk = max(0, int(args.max_frames_per_chunk))

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(args.seed))
    total_fit = 0
    total_used_chunks = 0

    for chunk_path in tqdm(chunk_paths, desc="Recovering"):
        if total_fit >= max_samples:
            break
        try:
            meta = _load_chunk_metadata(chunk_path)
            dino_tgt = _load_dino_tensor(chunk_path)
            if "frame_paths" not in meta:
                raise KeyError("frame_paths missing in chunk metadata.")
            frame_paths = list(meta["frame_paths"])
            if not frame_paths:
                raise RuntimeError("empty frame_paths in metadata.")
            if max_frames_per_chunk > 0:
                frame_paths = frame_paths[:max_frames_per_chunk]
                dino_tgt = dino_tgt[:, : len(frame_paths)]

            scene_id = str(meta.get("scene_id") or chunk_path.parent.parent.name)
            scene_dir = chunk_path.parent.parent

            resolved: List[str] = []
            for fp in frame_paths:
                rp = _resolve_frame_path(
                    str(fp),
                    dataset_root=dataset_root,
                    scene_id=scene_id,
                    scene_dir=scene_dir,
                    prefix_from=prefix_from,
                    prefix_to=prefix_to,
                )
                if rp is None:
                    raise FileNotFoundError(f"Could not resolve frame path: {fp}")
                resolved.append(str(rp))

            images = load_and_preprocess_images(resolved)
            if images.ndim == 4:
                images = images.unsqueeze(0)
            images = images.to(device)
            B, S = int(images.shape[0]), int(images.shape[1])
            H, W = int(images.shape[-2]), int(images.shape[-1])

            with torch.no_grad():
                _ = model(images)
                raw = tap.extract_raw_fmap((B, S, H, W)).detach().cpu().float().contiguous()

            if dino_tgt.shape[0] != B and dino_tgt.shape[0] == 1 and B > 1:
                dino_tgt = dino_tgt.expand(B, -1, -1, -1, -1).contiguous()
            if raw.shape[:2] != dino_tgt.shape[:2]:
                raise ValueError(
                    f"Frame dims mismatch raw={tuple(raw.shape[:2])} target={tuple(dino_tgt.shape[:2])}"
                )
            if raw.shape[-2:] != dino_tgt.shape[-2:]:
                raise ValueError(
                    f"Spatial mismatch raw={tuple(raw.shape[-2:])} target={tuple(dino_tgt.shape[-2:])}"
                )

            cin = int(raw.shape[2])
            cout = int(dino_tgt.shape[2])
            if cout != 256:
                raise ValueError(f"Expected target channels=256, got {cout}")

            raw_flat = raw.permute(0, 1, 3, 4, 2).reshape(-1, cin)
            tgt_flat = dino_tgt.permute(0, 1, 3, 4, 2).reshape(-1, cout)
            total = int(raw_flat.shape[0])
            if total <= 0:
                raise RuntimeError("No patch samples in chunk.")

            remaining = max_samples - total_fit
            n_fit = min(samples_per_chunk, remaining, total)
            n_eval = min(eval_per_chunk, max(0, total - n_fit))
            if n_fit <= 0:
                break

            perm = torch.randperm(total, generator=rng)
            fit_idx = perm[:n_fit]
            eval_idx = perm[n_fit : n_fit + n_eval]

            x_fit = raw_flat[fit_idx].contiguous()
            y_fit = tgt_flat[fit_idx].contiguous()

            if fit_acc is None:
                fit_acc = LinearAccumulator(cin, cout, ridge=float(args.ridge))
            fit_acc.add(x_fit, y_fit)
            total_fit += n_fit
            total_used_chunks += 1
            chunk_fit_counts[str(chunk_path)] = n_fit

            if args.fit_per_scene:
                acc = scene_acc.get(scene_id)
                if acc is None:
                    acc = LinearAccumulator(cin, cout, ridge=float(args.ridge))
                    scene_acc[scene_id] = acc
                acc.add(x_fit, y_fit)

            if n_eval > 0:
                x_eval = raw_flat[eval_idx].contiguous()
                y_eval = tgt_flat[eval_idx].contiguous()
                eval_chunks.append(
                    ChunkEval(
                        scene_id=scene_id,
                        chunk_path=str(chunk_path),
                        num_eval=n_eval,
                        x_eval=x_eval,
                        y_eval=y_eval,
                    )
                )

        except Exception as exc:
            chunk_warnings.append(f"{chunk_path}: {exc}")
            continue

    tap.remove()

    if fit_acc is None or fit_acc.num_samples <= 0:
        raise RuntimeError("No usable samples collected.")

    w, b = fit_acc.solve()

    # Global eval diagnostics
    chunk_rmse: List[Dict[str, object]] = []
    scene_stats: Dict[str, Dict[str, float]] = {}
    for ce in eval_chunks:
        rmse = _rmse(ce.x_eval, ce.y_eval, w, b)
        chunk_rmse.append(
            {
                "scene_id": ce.scene_id,
                "chunk_path": ce.chunk_path,
                "num_eval": ce.num_eval,
                "rmse": rmse,
            }
        )
        s = scene_stats.setdefault(ce.scene_id, {"sum_rmse": 0.0, "num_chunks": 0.0, "num_eval": 0.0})
        s["sum_rmse"] += rmse
        s["num_chunks"] += 1.0
        s["num_eval"] += float(ce.num_eval)

    for sid, s in scene_stats.items():
        n = max(1.0, s["num_chunks"])
        s["mean_chunk_rmse"] = s["sum_rmse"] / n

    chunk_rmse_sorted = sorted(chunk_rmse, key=lambda d: float(d["rmse"]), reverse=True)

    # Optional per-scene fits
    per_scene_report: Dict[str, Dict[str, float]] = {}
    scene_out_dir: Optional[Path] = None
    if args.fit_per_scene and args.out_scene_dir:
        scene_out_dir = Path(args.out_scene_dir).expanduser()
        scene_out_dir.mkdir(parents=True, exist_ok=True)

    if args.fit_per_scene:
        eval_by_scene: Dict[str, List[ChunkEval]] = {}
        for ce in eval_chunks:
            eval_by_scene.setdefault(ce.scene_id, []).append(ce)

        for sid, acc in scene_acc.items():
            if acc.num_samples < int(args.min_scene_fit_samples):
                continue
            ws, bs = acc.solve()
            scene_eval = eval_by_scene.get(sid, [])
            if scene_eval:
                rmses = [_rmse(ce.x_eval, ce.y_eval, ws, bs) for ce in scene_eval]
                mean_scene_rmse = float(sum(rmses) / len(rmses))
                global_scene_rmse = float(
                    sum(_rmse(ce.x_eval, ce.y_eval, w, b) for ce in scene_eval) / len(scene_eval)
                )
            else:
                mean_scene_rmse = float("nan")
                global_scene_rmse = float("nan")

            delta = global_scene_rmse - mean_scene_rmse
            per_scene_report[sid] = {
                "fit_samples": float(acc.num_samples),
                "global_rmse_on_scene": global_scene_rmse,
                "scene_specific_rmse": mean_scene_rmse,
                "rmse_improvement_vs_global": delta,
            }

            if scene_out_dir is not None:
                out_scene = scene_out_dir / f"{sid}_dino_proj.pt"
                torch.save(
                    {
                        "weight": ws.view(256, ws.shape[1], 1, 1).contiguous(),
                        "bias": bs.contiguous(),
                        "in_channels": int(ws.shape[1]),
                        "out_channels": int(ws.shape[0]),
                        "scene_id": sid,
                        "tap_name": tap.tap_name,
                        "patch_size": int(args.patch_size),
                        "num_fit_samples": int(acc.num_samples),
                    },
                    out_scene,
                )

    out_proj = Path(args.out_proj).expanduser()
    out_proj.parent.mkdir(parents=True, exist_ok=True)
    proj_payload = {
        "weight": w.view(256, w.shape[1], 1, 1).contiguous(),
        "bias": b.contiguous(),
        "in_channels": int(w.shape[1]),
        "out_channels": int(w.shape[0]),
        "tap_name": tap.tap_name,
        "patch_size": int(args.patch_size),
        "num_fit_samples": int(fit_acc.num_samples),
        "num_used_chunks": int(total_used_chunks),
    }
    torch.save(proj_payload, out_proj)

    global_rmse_mean = float("nan")
    if chunk_rmse:
        global_rmse_mean = float(sum(float(x["rmse"]) for x in chunk_rmse) / len(chunk_rmse))

    report = {
        "dataset_root": str(dataset_root),
        "num_requested_scenes": len(scenes),
        "num_scanned_chunks": len(chunk_paths),
        "num_used_chunks": int(total_used_chunks),
        "num_fit_samples": int(fit_acc.num_samples),
        "tap_name": tap.tap_name,
        "patch_size": int(args.patch_size),
        "in_channels": int(w.shape[1]),
        "out_channels": int(w.shape[0]),
        "global_mean_chunk_rmse": global_rmse_mean,
        "chunk_rmse_top20": chunk_rmse_sorted[:20],
        "scene_stats": scene_stats,
        "per_scene_fit_report": per_scene_report,
        "num_chunk_warnings": len(chunk_warnings),
        "chunk_warnings_head": chunk_warnings[:30],
        "out_proj": str(out_proj),
    }

    out_report = Path(args.out_report).expanduser() if args.out_report else out_proj.with_suffix(".json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[recover] saved global projector to: {out_proj}")
    print(f"[recover] wrote report to: {out_report}")
    print(f"[recover] used chunks={total_used_chunks}, fit_samples={fit_acc.num_samples}")
    if chunk_warnings:
        print(f"[recover] warnings={len(chunk_warnings)} (see report)")


if __name__ == "__main__":
    main()
