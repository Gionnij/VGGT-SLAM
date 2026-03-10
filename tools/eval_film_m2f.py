#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse training helpers without requiring scripts/ to be a package.
import importlib.util


def _load_train_module() -> object:
    here = Path(__file__).resolve()
    train_path = here.parents[1] / "scripts" / "train_film_m2f_optimized.py"
    spec = importlib.util.spec_from_file_location("train_film_m2f_optimized", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load training module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_list_from_file(path_str: Optional[str]) -> Optional[List[str]]:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"List file not found: {path}")
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        val = line.strip()
        if not val or val.startswith("#"):
            continue
        out.append(val)
    return out or None


def _parse_ignore_classes(arg: Optional[str]) -> Set[int]:
    ignore: Set[int] = set()
    if not arg:
        return ignore
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        ignore.add(int(item))
    return ignore


def _make_palette(num_classes: int, seed: int = 123) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(0, 256, (num_classes, 3), dtype=torch.uint8, generator=g)


def _colorize_mask(mask: torch.Tensor, palette: torch.Tensor, ignore_index: int) -> "Image.Image":
    from PIL import Image
    import numpy as np

    mask_np = mask.detach().cpu().numpy().astype("int64")
    h, w = mask_np.shape
    out = np.zeros((h, w, 3), dtype="uint8")
    valid = (mask_np >= 0) & (mask_np < palette.shape[0])
    out[valid] = palette[mask_np[valid]].numpy()
    if ignore_index is not None:
        ignore = mask_np == ignore_index
        out[ignore] = 0
    return Image.fromarray(out, mode="RGB")


def _parse_epoch_step_from_name(name: str) -> Tuple[int, int]:
    m = re.search(r"epoch(\d+)_step(\d+)", name)
    if not m:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def _collect_checkpoints(ckpt_dir: Path) -> List[Path]:
    entries: List[Tuple[int, int, float, Path]] = []
    for p in ckpt_dir.glob("*.pt"):
        epoch, step = _parse_epoch_step_from_name(p.name)
        entries.append((epoch, step, p.stat().st_mtime, p))
    entries.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [p for _, _, _, p in entries]


def _extract_model_state(payload: Dict) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state" in payload and isinstance(payload["model_state"], dict):
        return payload["model_state"]
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint format: {type(payload)}")


def _resolve_rgb_path(frame_path: str, scene_id: str, dataset_root: Path) -> Optional[Path]:
    fpath = Path(frame_path)
    basename = fpath.name
    candidates = [
        fpath,
        dataset_root / scene_id / "images" / basename,
        dataset_root / scene_id / "rgb" / basename,
        dataset_root / scene_id / "dslr" / "resized_undistorted_images" / basename,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _build_export_indices(total_samples: int, export_samples: int) -> List[int]:
    n = int(total_samples)
    k = max(0, int(export_samples))
    if n <= 0 or k <= 0:
        return []
    if k == 1:
        return [0]
    if k >= n:
        return list(range(n))
    max_pos = n - 1
    picks = set()
    for i in range(k):
        idx = int(round(i * max_pos / (k - 1)))
        picks.add(idx)
    return sorted(picks)


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "checkpoint",
        "checkpoint_name",
        "epoch",
        "step",
        "val_loss",
        "miou",
        "pixel_acc",
        "num_samples",
        "num_batches",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def collate_fn_eval(batch: List[Dict]) -> Dict:
    dino = torch.stack([b["dino"] for b in batch], dim=0)
    dpt_levels = []
    for lvl_idx in range(len(batch[0]["dpt_levels"])):
        dpt_levels.append(torch.stack([b["dpt_levels"][lvl_idx] for b in batch], dim=0))
    labels = [b["label"] for b in batch]
    return {
        "dino": dino,
        "dpt_levels": dpt_levels,
        "labels": labels,
        "chunk_paths": [b["chunk_path"] for b in batch],
        "image_sizes": [b["image_size"] for b in batch],
        "scene_ids": [b["scene_id"] for b in batch],
        "frame_paths": [b["frame_path"] for b in batch],
    }


def evaluate_checkpoint(
    *,
    module: object,
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    focal_alpha: float,
    focal_gamma: float,
    export_masks: bool,
    export_raw: bool,
    export_indices: Sequence[int],
    export_dir: Path,
    dataset_root: Path,
    overlay_alpha: float,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")

    palette = _make_palette(num_classes)
    export_indices_set = set(export_indices)
    seen_exports: Set[int] = set()

    total_samples = len(dl.dataset)
    sample_idx = 0
    exported = 0
    image_mod = None
    if export_masks:
        from PIL import Image

        image_mod = Image

    with torch.no_grad():
        for batch in dl:
            dino = batch["dino"].to(device, non_blocking=True)
            dpt_levels = [lvl.to(device, non_blocking=True) for lvl in batch["dpt_levels"]]
            labels = batch["labels"]
            label_tensor = torch.stack(labels, dim=0).to(device, non_blocking=True)
            H, W = label_tensor.shape[-2:]

            cls_logits, mask_logits = model(dino, dpt_levels, label_shape=(H, W))
            s_frames = cls_logits.shape[1] if cls_logits.dim() == 4 else 1
            seg_logits = module.dense_logits_from_queries(
                cls_logits, mask_logits, B=dino.shape[0], S=s_frames
            )
            if seg_logits.dim() == 5:
                seg_logits = seg_logits.reshape(dino.shape[0] * s_frames, *seg_logits.shape[2:])
            target_size = seg_logits.shape[-2:]
            label_down = F.interpolate(
                label_tensor.unsqueeze(1).float(), size=target_size, mode="nearest"
            ).squeeze(1).long()

            loss = module.focal_loss(
                seg_logits,
                label_down,
                ignore_index=ignore_index,
                alpha=focal_alpha,
                gamma=focal_gamma,
            )
            loss_sum += float(loss.item())
            n_batches += 1

            preds = seg_logits.argmax(dim=1)
            valid = label_down != ignore_index
            if valid.any():
                gt = label_down[valid].clamp(0, num_classes - 1).view(-1).cpu()
                pd = preds[valid].clamp(0, num_classes - 1).view(-1).cpu()
                idx = gt * num_classes + pd
                conf.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.int64))

            if export_masks:
                for bi in range(preds.shape[0]):
                    if sample_idx in export_indices_set and sample_idx not in seen_exports:
                        scene_id = str(batch["scene_ids"][bi])
                        frame_token = Path(str(batch["frame_paths"][bi])).stem
                        pred_down = preds[bi : bi + 1].float()
                        pred_up = F.interpolate(
                            pred_down.unsqueeze(1),
                            size=(H, W),
                            mode="nearest",
                        ).squeeze(0).squeeze(0).long()

                        gt_mask = label_tensor[bi].detach().cpu().long()
                        pred_mask = pred_up.detach().cpu().long()

                        gt_img = _colorize_mask(gt_mask, palette, ignore_index)
                        pred_img = _colorize_mask(pred_mask, palette, ignore_index)

                        export_dir.mkdir(parents=True, exist_ok=True)
                        base = f"sample_{sample_idx:06d}_{scene_id}_{frame_token}"
                        gt_img.save(export_dir / f"{base}_gt.png")
                        pred_img.save(export_dir / f"{base}_pred.png")

                        rgb_path = _resolve_rgb_path(
                            str(batch["frame_paths"][bi]),
                            scene_id,
                            dataset_root,
                        )
                        rgb_img = None
                        if rgb_path is not None:
                            try:
                                rgb_img = image_mod.open(rgb_path).convert("RGB") if image_mod else None
                            except Exception:
                                rgb_img = None
                        if rgb_img is not None:
                            rgb_img.save(export_dir / f"{base}_rgb.png")
                            pred_for_overlay = pred_img
                            if pred_for_overlay.size != rgb_img.size:
                                pred_for_overlay = pred_for_overlay.resize(rgb_img.size, image_mod.NEAREST)
                            overlay = image_mod.blend(
                                rgb_img,
                                pred_for_overlay,
                                alpha=max(0.0, min(1.0, float(overlay_alpha))),
                            )
                            overlay.save(export_dir / f"{base}_overlay.png")

                        if export_raw:
                            import numpy as np
                            from PIL import Image

                            gt_raw = gt_mask.cpu().numpy().astype(np.uint16)
                            pred_raw = pred_mask.cpu().numpy().astype(np.uint16)
                            Image.fromarray(gt_raw, mode="I;16").save(export_dir / f"{base}_gt_id.png")
                            Image.fromarray(pred_raw, mode="I;16").save(export_dir / f"{base}_pred_id.png")
                        seen_exports.add(sample_idx)
                        exported += 1
                    sample_idx += 1
                continue

            sample_idx += preds.shape[0]

    tp = torch.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = (tp + fp + fn).clamp_min(1)
    iou = tp.float() / denom.float()
    miou = float(iou.mean().item())
    pixel_acc = float(tp.sum().float().item() / max(1, conf.sum().item()))

    return {
        "val_loss": loss_sum / max(1, n_batches),
        "miou": miou,
        "pixel_acc": pixel_acc,
        "num_samples": total_samples,
        "num_batches": n_batches,
        "num_exports": exported,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate FiLM+Mask2Former checkpoints with mIoU/val loss.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--labels-root")
    ap.add_argument("--label-chunks-root")
    ap.add_argument("--label-chunk-subdir", default="label_chunks")
    ap.add_argument("--label-chunk-ext", default=".pt")
    ap.add_argument("--chunk-format", choices=["auto", "pt", "safetensors"], default="auto")
    ap.add_argument("--scenes-file", required=True)
    ap.add_argument("--scenes", help="Comma-separated scenes list (overridden by --scenes-file).")
    ap.add_argument("--split-name", default="val", help="Label used in summary outputs (e.g. val/test).")
    ap.add_argument("--index-cache")
    ap.add_argument("--num-classes", type=int, default=200)
    ap.add_argument("--ignore-index", type=int, default=65535)
    ap.add_argument("--ignore-classes")
    ap.add_argument("--ignore-classes-file")
    ap.add_argument("--remap-classes-file")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--chunk-cache-size", type=int, default=4)
    ap.add_argument("--label-cache-size", type=int, default=0)
    ap.add_argument("--label-chunk-cache-size", type=int, default=2)
    ap.add_argument("--no-label-chunks", dest="use_label_chunks", action="store_false")
    ap.set_defaults(use_label_chunks=True)
    ap.add_argument("--config-path", default="mask2former/configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml")
    ap.add_argument("--weights-path", default="models/m2f_ade20k.pkl")
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--max-checkpoints", type=int, default=0, help="0 = no limit")
    ap.add_argument("--sort-by", choices=["miou", "pixel_acc", "val_loss"], default="miou")
    ap.add_argument("--sort-order", choices=["auto", "asc", "desc"], default="auto")
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-csv", type=Path)
    ap.add_argument("--export-masks", action="store_true", help="Export qualitative samples from top checkpoints.")
    ap.add_argument(
        "--export-raw",
        action="store_true",
        help="Also export raw label-id masks (uint16 PNG) alongside colorized masks.",
    )
    ap.add_argument("--export-dir", type=Path, default=Path("eval_masks"))
    ap.add_argument("--export-top-k", type=int, default=1, help="How many top-ranked checkpoints to export.")
    ap.add_argument("--export-samples", type=int, default=4, help="How many samples per exported checkpoint.")
    ap.add_argument("--overlay-alpha", type=float, default=0.45)
    args = ap.parse_args()

    module = _load_train_module()

    dataset_root = Path(args.dataset_root).expanduser()
    scenes_from_file = _load_list_from_file(args.scenes_file)
    if scenes_from_file and args.scenes:
        print("[info] scenes-file provided; overriding --scenes.")
    scenes_cli = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    scenes = scenes_from_file or scenes_cli

    ignore_classes = _parse_ignore_classes(args.ignore_classes)
    ignore_from_file = _load_list_from_file(args.ignore_classes_file)
    if ignore_from_file:
        ignore_classes.update(int(s) for s in ignore_from_file)
    if ignore_classes:
        print(f"[info] remapping {len(ignore_classes)} classes to ignore_index={args.ignore_index}")

    remap_classes = _load_list_from_file(args.remap_classes_file)
    remap_dict: Dict[int, int] = {}
    if remap_classes:
        remap_classes = [int(x) for x in remap_classes]
        remap_classes = sorted(set(remap_classes))
        remap_dict = {orig: new for new, orig in enumerate(remap_classes)}
        if args.num_classes == 200:
            args.num_classes = len(remap_classes)
            print(f"[info] remap_classes_file provided; setting num_classes={args.num_classes}")
        else:
            print(f"[info] remap_classes_file provided; keeping user num_classes={args.num_classes}")

    chunk_files = module.list_chunk_files(
        dataset_root, scenes=scenes, chunk_format=args.chunk_format
    )
    samples = module.build_or_load_index(
        chunk_files=chunk_files,
        labels_root=Path(args.labels_root).expanduser() if args.labels_root else None,
        cache_path=Path(args.index_cache).expanduser() if args.index_cache else None,
        label_chunk_root=Path(args.label_chunks_root).expanduser() if args.label_chunks_root else None,
        label_chunk_subdir=args.label_chunk_subdir,
        label_chunk_ext=args.label_chunk_ext,
        use_label_chunks=args.use_label_chunks,
    )

    ds = module.ChunkDataset(
        samples,
        ignore_classes=sorted(ignore_classes),
        ignore_value=args.ignore_index,
        remap_dict=remap_dict,
        labels_root=Path(args.labels_root).expanduser() if args.labels_root else None,
        cache_size=max(1, args.chunk_cache_size),
        label_cache_size=args.label_cache_size,
        label_chunk_cache_size=args.label_chunk_cache_size,
        use_label_chunks=args.use_label_chunks,
        prefetch_next_chunk=False,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn_eval,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.FusionMask2Former(
        device=device,
        num_classes=args.num_classes,
        config_path=args.config_path,
        weights_path=args.weights_path,
    )

    ckpt_dir = Path(args.checkpoint_dir).expanduser()
    checkpoints = _collect_checkpoints(ckpt_dir)
    if args.max_checkpoints and args.max_checkpoints > 0:
        checkpoints = checkpoints[: args.max_checkpoints]
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    results: List[Dict] = []
    for idx, ckpt_path in enumerate(checkpoints):
        print(f"[info] evaluating {ckpt_path.name} ({idx + 1}/{len(checkpoints)})")
        payload = torch.load(ckpt_path, map_location="cpu")
        model_state = _extract_model_state(payload)
        model.load_state_dict(model_state, strict=True)
        metrics = evaluate_checkpoint(
            module=module,
            model=model,
            dl=dl,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            export_masks=False,
            export_raw=False,
            export_indices=[],
            export_dir=args.export_dir,
            dataset_root=dataset_root,
            overlay_alpha=args.overlay_alpha,
        )
        epoch, step = _parse_epoch_step_from_name(ckpt_path.name)
        row = {
            "checkpoint": str(ckpt_path),
            "checkpoint_name": ckpt_path.name,
            "epoch": epoch,
            "step": step,
            **metrics,
        }
        results.append(row)
        print(
            f"[result] loss={metrics['val_loss']:.4f} "
            f"mIoU={metrics['miou']:.4f} "
            f"pix_acc={metrics['pixel_acc']:.4f}"
        )

    if args.sort_order == "auto":
        reverse = args.sort_by != "val_loss"
    else:
        reverse = args.sort_order == "desc"
    ranked = sorted(results, key=lambda x: float(x[args.sort_by]), reverse=reverse)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    print(
        f"[summary] split={args.split_name} samples={len(ds)} "
        f"checkpoints={len(ranked)} sort={args.sort_by} ({'desc' if reverse else 'asc'})"
    )
    for row in ranked[: min(10, len(ranked))]:
        print(
            f"[rank {row['rank']:>2}] "
            f"ep={row['epoch']:>4} step={row['step']:>8} "
            f"loss={row['val_loss']:.4f} miou={row['miou']:.4f} pix={row['pixel_acc']:.4f} "
            f"{row['checkpoint_name']}"
        )

    if args.out_csv:
        _write_csv(args.out_csv, ranked)
        print(f"[info] wrote {args.out_csv}")

    if args.out_json:
        payload = {
            "split": args.split_name,
            "dataset_root": str(dataset_root),
            "scenes_file": str(Path(args.scenes_file).expanduser()),
            "num_scenes": len(scenes) if scenes is not None else None,
            "num_samples": len(ds),
            "num_checkpoints": len(ranked),
            "sort_by": args.sort_by,
            "sort_order": "desc" if reverse else "asc",
            "results": ranked,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"[info] wrote {args.out_json}")

    if args.export_masks and args.export_top_k > 0:
        export_indices = _build_export_indices(len(ds), args.export_samples)
        print(f"[info] export indices: {export_indices}")
        export_count = min(int(args.export_top_k), len(ranked))
        for row in ranked[:export_count]:
            ckpt_path = Path(row["checkpoint"])
            payload = torch.load(ckpt_path, map_location="cpu")
            model_state = _extract_model_state(payload)
            model.load_state_dict(model_state, strict=True)
            export_dir = args.export_dir / f"rank_{int(row['rank']):02d}_{ckpt_path.stem}"
            print(f"[info] exporting qualitative samples for rank {row['rank']} -> {export_dir}")
            export_metrics = evaluate_checkpoint(
                module=module,
                model=model,
                dl=dl,
                device=device,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                export_masks=True,
                export_raw=args.export_raw,
                export_indices=export_indices,
                export_dir=export_dir,
                dataset_root=dataset_root,
                overlay_alpha=args.overlay_alpha,
            )
            print(f"[info] exported {int(export_metrics.get('num_exports', 0))} samples for {ckpt_path.name}")


if __name__ == "__main__":
    main()
