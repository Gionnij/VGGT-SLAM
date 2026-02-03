#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
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


def _collect_checkpoints(ckpt_dir: Path) -> List[Path]:
    ckpt_re = re.compile(r"epoch(\d+)_step(\d+)")
    entries: List[Tuple[int, int, float, Path]] = []
    for p in ckpt_dir.glob("*.pt"):
        m = ckpt_re.search(p.name)
        if m:
            epoch = int(m.group(1))
            step = int(m.group(2))
        else:
            epoch = -1
            step = -1
        entries.append((epoch, step, p.stat().st_mtime, p))
    entries.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [p for _, _, _, p in entries]


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
                        scene_id = batch["scene_ids"][bi]
                        frame_path = Path(batch["frame_paths"][bi]).name
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
                        gt_img.save(export_dir / f"sample_{sample_idx:06d}_{scene_id}_{frame_path}_gt.png")
                        pred_img.save(export_dir / f"sample_{sample_idx:06d}_{scene_id}_{frame_path}_pred.png")
                        if export_dir and export_dir.is_dir() and export_masks and export_raw:
                            # Save raw label ids as uint16 PNGs for exact inspection.
                            import numpy as np
                            from PIL import Image

                            gt_raw = gt_mask.cpu().numpy().astype(np.uint16)
                            pred_raw = pred_mask.cpu().numpy().astype(np.uint16)
                            Image.fromarray(gt_raw, mode="I;16").save(
                                export_dir / f"sample_{sample_idx:06d}_{scene_id}_{frame_path}_gt_id.png"
                            )
                            Image.fromarray(pred_raw, mode="I;16").save(
                                export_dir / f"sample_{sample_idx:06d}_{scene_id}_{frame_path}_pred_id.png"
                            )
                        seen_exports.add(sample_idx)
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
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--export-masks", action="store_true", help="Export 4 GT/pred masks for visual inspection.")
    ap.add_argument(
        "--export-raw",
        action="store_true",
        help="Also export raw label-id masks (uint16 PNG) alongside colorized masks.",
    )
    ap.add_argument("--export-dir", type=Path, default=Path("eval_masks"))
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

    export_indices = []
    if args.export_masks:
        n = len(ds)
        export_indices = sorted(set([0, n // 2, (3 * n) // 4, n - 1]))
        export_indices = [i for i in export_indices if 0 <= i < n]
        print(f"[info] exporting masks for indices: {export_indices}")

    results = []
    for idx, ckpt_path in enumerate(checkpoints):
        print(f"[info] evaluating {ckpt_path.name} ({idx + 1}/{len(checkpoints)})")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)

        export_dir = args.export_dir / ckpt_path.stem
        metrics = evaluate_checkpoint(
            module=module,
            model=model,
            dl=dl,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            export_masks=args.export_masks,
            export_raw=args.export_raw,
            export_indices=export_indices,
            export_dir=export_dir,
        )
        metrics["checkpoint"] = str(ckpt_path)
        results.append(metrics)
        print(
            f"[result] loss={metrics['val_loss']:.4f} "
            f"mIoU={metrics['miou']:.4f} "
            f"pix_acc={metrics['pixel_acc']:.4f}"
        )

    if args.out_json:
        payload = {"results": results}
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"[info] wrote {args.out_json}")


if __name__ == "__main__":
    main()
