#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import html
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
POLL_SEC = 1.0
_CLASS_LABEL_CACHE: Optional[list[tuple[int, str]]] = None

_TUNNEL_PROC: Optional[subprocess.Popen[bytes]] = None
_TUNNEL_SIG: Optional[tuple[str, str, str, str, int, int]] = None


def _prune_incompatible_user_site_paths() -> None:
    cur_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
    keep: list[str] = []
    pat = re.compile(r"/Library/Python/(\d+\.\d+)/lib/python/site-packages/?$")
    for path in sys.path:
        m = pat.search(path)
        if m is not None and m.group(1) != cur_mm:
            continue
        keep.append(path)
    sys.path[:] = keep


@dataclass
class SshConfig:
    host: str
    user: str = ""
    key_path: str = ""
    options: str = ""

    @property
    def target(self) -> str:
        if self.user.strip():
            return f"{self.user.strip()}@{self.host.strip()}"
        return self.host.strip()


def _quote(value: str) -> str:
    return shlex.quote(value)


def _shell_join(args: list[str]) -> str:
    return " ".join(_quote(arg) for arg in args)


def _shell_path_arg(path: str) -> str:
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME/' + path[2:].replace('"', '\\"') + '"'
    if path.startswith("$HOME/") or path == "$HOME":
        return '"' + path.replace('"', '\\"') + '"'
    return _quote(path)


def _tail(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return f"...(truncated {len(text) - limit} chars)\n{text[-limit:]}"


def _read_nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def _resolve_first_existing_path(candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_local_class_labels() -> list[tuple[int, str]]:
    global _CLASS_LABEL_CACHE
    if _CLASS_LABEL_CACHE is not None:
        return _CLASS_LABEL_CACHE

    repo_root = Path(__file__).resolve().parents[1]
    kept_candidates = []
    classes_candidates = []

    env_kept = os.getenv("VGGT_DASHBOARD_KEPT_CLASSES_PATH", "").strip()
    env_classes = os.getenv("VGGT_DASHBOARD_SEMANTIC_CLASSES_PATH", "").strip()
    if env_kept:
        kept_candidates.append(Path(env_kept).expanduser())
    if env_classes:
        classes_candidates.append(Path(env_classes).expanduser())

    kept_candidates.extend(
        [
            Path("/Users/giovannichiementin/Desktop/kept_classes_top60.txt"),
            Path.home() / "Desktop" / "kept_classes_top60.txt",
            repo_root / "kept_classes_top60.txt",
        ]
    )
    classes_candidates.extend(
        [
            Path("/Users/giovannichiementin/Desktop/semantic_classes.txt"),
            Path.home() / "Desktop" / "semantic_classes.txt",
            repo_root / "semantic_classes.txt",
        ]
    )

    kept_path = _resolve_first_existing_path(kept_candidates)
    classes_path = _resolve_first_existing_path(classes_candidates)
    if kept_path is None or classes_path is None:
        _CLASS_LABEL_CACHE = []
        return _CLASS_LABEL_CACHE

    try:
        kept_ids = [int(x) for x in _read_nonempty_lines(kept_path)]
        semantic_names = _read_nonempty_lines(classes_path)
    except Exception:
        _CLASS_LABEL_CACHE = []
        return _CLASS_LABEL_CACHE

    # Match the dense-id convention used by training/eval:
    # remap_classes = sorted(set(remap_classes))
    dense_to_orig = sorted(set(kept_ids))
    labels: list[tuple[int, str]] = []
    for global_idx in dense_to_orig:
        if 0 <= global_idx < len(semantic_names):
            labels.append((global_idx, semantic_names[global_idx]))
        else:
            labels.append((global_idx, f"class_{global_idx}"))
    _CLASS_LABEL_CACHE = labels
    return _CLASS_LABEL_CACHE


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'").strip('"')
    return out


def _run_local_bash(script: str, timeout_s: int, *, text: bool) -> tuple[int, object, object]:
    proc = subprocess.run(
        ["bash", "-lc", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _ssh_base_args(cfg: SshConfig) -> list[str]:
    args = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    key = cfg.key_path.strip()
    if key:
        args.extend(["-i", str(Path(key).expanduser())])
    opts = cfg.options.strip()
    if opts:
        args.extend(shlex.split(opts))
    return args


def _run_ssh_bash(cfg: SshConfig, script: str, timeout_s: int, *, text: bool) -> tuple[int, object, object]:
    if not cfg.host.strip():
        err = "missing ssh host"
        return 2, "" if text else b"", err if text else err.encode("utf-8")
    cmd = _ssh_base_args(cfg)
    cmd.extend([cfg.target, f"bash -lc {_quote(script)}"])
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _stop_tunnel() -> None:
    global _TUNNEL_PROC, _TUNNEL_SIG
    proc = _TUNNEL_PROC
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _TUNNEL_PROC = None
    _TUNNEL_SIG = None


def _ensure_tunnel(cfg: SshConfig, gpu_host: str, remote_port: int, local_port: int) -> tuple[bool, str]:
    global _TUNNEL_PROC, _TUNNEL_SIG
    sig = (
        cfg.target,
        cfg.key_path.strip(),
        cfg.options.strip(),
        gpu_host.strip(),
        int(remote_port),
        int(local_port),
    )

    if _TUNNEL_PROC is not None and _TUNNEL_PROC.poll() is None and _TUNNEL_SIG == sig:
        return True, "reused"

    _stop_tunnel()
    if not cfg.host.strip():
        return False, "missing ssh host"
    if not gpu_host.strip():
        return False, "missing gpu host"

    cmd = _ssh_base_args(cfg)
    cmd.extend(
        [
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"127.0.0.1:{int(local_port)}:{gpu_host.strip()}:{int(remote_port)}",
            cfg.target,
        ]
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.45)
    if proc.poll() is not None:
        err = ""
        try:
            err = (proc.stderr.read() if proc.stderr is not None else "") or ""
        except Exception:
            pass
        _TUNNEL_PROC = None
        _TUNNEL_SIG = None
        return False, f"failed: {_tail(err.strip(), 1000)}"

    _TUNNEL_PROC = proc  # keep alive for dashboard lifetime
    _TUNNEL_SIG = sig
    return True, "started"


atexit.register(_stop_tunnel)


def _viewer_html(url: str) -> str:
    safe_url = html.escape(url, quote=True)
    return (
        "<div style='border:1px solid #d0d7de; border-radius:10px; padding:14px 16px; "
        "display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;'>"
        "<div style='display:flex; flex-direction:column; gap:6px;'>"
        "<div style='font-size:14px; line-height:1.4; font-weight:600;'>3D Map Viewer</div>"
        "<div style='font-size:13px; line-height:1.4;'>"
        "Open the live map in a separate browser tab."
        "</div>"
        f"<div style='font-family:monospace; font-size:12px; word-break:break-all;'>{safe_url}</div>"
        "</div>"
        f"<a href='{safe_url}' target='_blank' rel='noopener noreferrer' "
        "style='display:inline-block; width:fit-content; padding:10px 14px; "
        "background:#0f766e; color:#ffffff; text-decoration:none; border-radius:8px; "
        "font-weight:600;'>Open 3D Map Viewer</a>"
        "</div>"
    )


def _viewer_placeholder(message: str) -> str:
    msg = html.escape(message)
    return (
        "<div style='border:1px dashed #d0d7de; border-radius:10px; padding:14px 16px; "
        "font-family:monospace; font-size:12px; text-align:center;'>"
        f"{msg}"
        "</div>"
    )


def _decode_image_bytes(blob: bytes) -> Optional[np.ndarray]:
    if not blob:
        return None
    if cv2 is not None:
        arr = np.frombuffer(blob, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if Image is not None:
        try:
            from io import BytesIO

            with Image.open(BytesIO(blob)) as im:
                return np.array(im.convert("RGB"))
        except Exception:
            return None
    return None


def _decode_mask_bytes(blob: bytes) -> Optional[np.ndarray]:
    if not blob:
        return None
    if cv2 is not None:
        arr = np.frombuffer(blob, dtype=np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None
        return mask
    if Image is not None:
        try:
            from io import BytesIO

            with Image.open(BytesIO(blob)) as im:
                return np.array(im)
        except Exception:
            return None
    return None


def _latest_file_in_dir(path: Path) -> Optional[Path]:
    if not path.is_dir():
        return None
    latest: Optional[Path] = None
    latest_mtime = -1.0
    for f in path.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest = f
    return latest


def _latest_image_in_dir(path: Path) -> tuple[Optional[np.ndarray], str]:
    if not path.is_dir():
        return None, f"{path} (missing)"
    latest = _latest_file_in_dir(path)
    if latest is None:
        return None, f"{path} (no images yet)"
    try:
        blob = latest.read_bytes()
    except Exception:
        return None, f"{latest} (failed to read)"
    img = _decode_image_bytes(blob)
    if img is None:
        return None, f"{latest} (failed to decode)"
    return img, str(latest)


def _latest_demo_run_dir(demo_root: Path) -> Optional[Path]:
    if not demo_root.is_dir():
        return None
    latest: Optional[Path] = None
    latest_mtime = -1.0
    for d in demo_root.iterdir():
        if not d.is_dir() or not d.name.startswith("demo_"):
            continue
        try:
            mtime = d.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest = d
    return latest


def _latest_overlay_image(demo_root: Path) -> tuple[Optional[np.ndarray], str]:
    run_dir = _latest_demo_run_dir(demo_root)
    if run_dir is None:
        return None, f"{demo_root} (no demo_* run yet)"
    return _latest_image_in_dir(run_dir / "segmentation_overlays")


def _latest_mask_path(demo_root: Path) -> str:
    run_dir = _latest_demo_run_dir(demo_root)
    if run_dir is None:
        return ""
    latest = _latest_file_in_dir(run_dir / "segmentation_masks")
    return str(latest) if latest is not None else ""


def _matching_rgb_path(run_dir: Optional[Path], anchor_path: str) -> str:
    if run_dir is None or not anchor_path:
        return ""
    stem = Path(anchor_path).stem
    rgb_dir = run_dir / "rgb"
    for ext in IMAGE_EXTS:
        candidate = rgb_dir / f"{stem}{ext}"
        if candidate.is_file():
            return str(candidate)
    return ""


def _capture_tmux_pane(session: str, window: str, lines: int = 120) -> str:
    if not session.strip():
        return ""
    proc = subprocess.run(
        ["tmux", "capture-pane", "-p", "-t", f"{session}:{window}", "-S", f"-{max(1, int(lines))}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _tail_text_file(path: Path, max_lines: int = 120) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max(1, int(max_lines)):])


def _semantic_log_excerpt(text: str, limit: int = 8) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    sem_lines = [ln for ln in lines if ("[SEM" in ln) or ("semantic_inference" in ln) or ("SEMHEAD" in ln)]
    if sem_lines:
        return sem_lines[-limit:]
    return lines[-limit:]


def _mask_summary(mask: Optional[np.ndarray], mask_path: str, overlay_path: str, gpu_recent: str) -> str:
    lines: list[str] = []
    lines.append(f"mask={mask_path or '(no mask yet)'}")
    lines.append(f"overlay={overlay_path or '(no overlay yet)'}")
    if mask is None:
        if mask_path:
            lines.append("mask_decode=failed")
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[..., 0]
            elif (
                arr.shape[2] >= 3
                and np.array_equal(arr[..., 0], arr[..., 1])
                and np.array_equal(arr[..., 0], arr[..., 2])
            ):
                arr = arr[..., 0]
            else:
                lines.append(f"warning=mask file decoded as {arr.shape}; expected single-channel labels")
                arr = arr[..., 0]
        uniq, counts = np.unique(arr.reshape(-1), return_counts=True)
        order = np.argsort(counts)[::-1]
        top = [f"{int(uniq[i])}:{int(counts[i])}" for i in order[:6]]
        lines.append(f"mask_shape={tuple(arr.shape)} dtype={arr.dtype}")
        lines.append(f"unique_labels={int(len(uniq))}")
        lines.append("top_labels=" + (", ".join(top) if top else "(none)"))
        if len(uniq) <= 1:
            lines.append("warning=latest semantic mask has only one label")
    excerpt = _semantic_log_excerpt(gpu_recent)
    if excerpt:
        lines.append("recent_semantic_log:")
        lines.extend(excerpt)
    else:
        lines.append("recent_semantic_log=(none)")
    return "\n".join(lines)


def _legend_color_rgb(
    label_mask: np.ndarray,
    label_id: int,
    overlay_img: Optional[np.ndarray],
    rgb_img: Optional[np.ndarray],
) -> tuple[int, int, int]:
    region = label_mask == label_id
    if not np.any(region):
        return (127, 127, 127)

    if (
        overlay_img is not None
        and rgb_img is not None
        and overlay_img.shape[:2] == label_mask.shape
        and rgb_img.shape[:2] == label_mask.shape
    ):
        pure = np.clip(
            2.0 * overlay_img.astype(np.float32) - rgb_img.astype(np.float32),
            0.0,
            255.0,
        ).astype(np.uint8)
        pixels = pure[region]
    elif overlay_img is not None and overlay_img.shape[:2] == label_mask.shape:
        pixels = overlay_img[region]
    else:
        return (127, 127, 127)

    if pixels.ndim != 2 or pixels.shape[0] == 0:
        return (127, 127, 127)
    color = np.median(pixels, axis=0).astype(np.uint8)
    return (int(color[0]), int(color[1]), int(color[2]))


def _legend_html(
    mask: Optional[np.ndarray],
    overlay_img: Optional[np.ndarray],
    rgb_img: Optional[np.ndarray],
) -> str:
    if mask is None:
        return (
            "<div style='border:1px dashed #d0d7de; border-radius:10px; padding:12px 14px; "
            "font-size:13px; color:#4b5563;'>Legend will appear once a semantic mask is available.</div>"
        )

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return (
            "<div style='border:1px dashed #d0d7de; border-radius:10px; padding:12px 14px; "
            "font-size:13px; color:#4b5563;'>Legend unavailable: mask shape is not 2D.</div>"
        )

    class_labels = _load_local_class_labels()
    uniq, counts = np.unique(arr.reshape(-1), return_counts=True)
    if uniq.size == 0:
        return (
            "<div style='border:1px dashed #d0d7de; border-radius:10px; padding:12px 14px; "
            "font-size:13px; color:#4b5563;'>Legend unavailable: no labels found in current mask.</div>"
        )

    order = np.argsort(counts)[::-1]
    chips: list[str] = []
    for idx in order:
        label_id = int(uniq[idx])
        rgb = _legend_color_rgb(arr, label_id, overlay_img, rgb_img)
        color_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        if 0 <= label_id < len(class_labels):
            original_id, name = class_labels[label_id]
        else:
            original_id, name = (-1, f"label {label_id}")
        title = html.escape(
            f"remapped {label_id}"
            + (f" -> original {original_id}" if original_id >= 0 else "")
        )
        chips.append(
            "<div style='display:inline-flex; align-items:center; gap:8px; padding:6px 10px; "
            "border:1px solid #d0d7de; border-radius:999px; background:#ffffff; white-space:nowrap;' "
            f"title='{title}'>"
            f"<span style='display:inline-block; width:12px; height:12px; border-radius:3px; "
            f"background:{color_hex}; border:1px solid rgba(0,0,0,0.18);'></span>"
            f"<span style='font-size:12px; line-height:1.2;'>{html.escape(name)}</span>"
            "</div>"
        )

    return (
        "<div style='border:1px solid #d0d7de; border-radius:10px; padding:10px 12px; overflow-x:auto;'>"
        "<div style='display:flex; gap:8px; align-items:center; width:max-content;'>"
        + "".join(chips)
        + "</div></div>"
    )


def _latest_processed_rgb(demo_root: Path) -> tuple[Optional[np.ndarray], str]:
    run_dir = _latest_demo_run_dir(demo_root)
    if run_dir is None:
        return None, f"{demo_root} (no demo_* run yet)"
    return _latest_image_in_dir(run_dir / "rgb")


def _pipeline_status_text_local(session: str) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    has = subprocess.run(
        ["bash", "-lc", f"tmux has-session -t {_quote(session)}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if has.returncode != 0:
        return f"[{now}] stopped (tmux session '{session}' not found)"

    pane = subprocess.run(
        ["bash", "-lc", f"tmux display-message -p -t {_quote(session + ':gpu-pipeline')} '#{{pane_dead}}'"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if pane.returncode == 0 and pane.stdout.strip() == "1":
        return f"[{now}] session alive; gpu-pipeline pane exited"
    if pane.returncode == 0 and pane.stdout.strip() == "0":
        return f"[{now}] running"
    return f"[{now}] session alive (gpu-pipeline pane state unavailable)"


def _extract_json_from_output(text: str) -> Optional[dict]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def _remote_probe(
    cfg: SshConfig,
    session: str,
    demo_root: str,
    raw_dir: str,
    state_file: str,
    timeout_s: int = 25,
) -> dict:
    py = f"""
import json
import subprocess
import time
from pathlib import Path

exts = {tuple(IMAGE_EXTS)!r}
session = {session!r}
demo_root = Path({demo_root!r}).expanduser()
raw_dir = Path({raw_dir!r}).expanduser()
state_file_arg = {state_file!r}.strip()
state_path = Path(state_file_arg).expanduser() if state_file_arg else (Path.home() / f".vggt_active_gpu_{{session}}.env")

def latest_demo_run(root: Path):
    if not root.is_dir():
        return None
    best = None
    best_m = -1.0
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith("demo_"):
            continue
        try:
            m = d.stat().st_mtime
        except OSError:
            continue
        if m > best_m:
            best_m = m
            best = d
    return best

def latest_image(path: Path):
    if not path.is_dir():
        return ""
    best = None
    best_m = -1.0
    for f in path.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in exts:
            continue
        try:
            m = f.stat().st_mtime
        except OSError:
            continue
        if m > best_m:
            best_m = m
            best = f
    return str(best) if best is not None else ""

def matching_rgb(run_dir: Path, anchor_path: str):
    if not anchor_path:
        return ""
    stem = Path(anchor_path).stem
    rgb_dir = run_dir / "rgb"
    for ext in exts:
        candidate = rgb_dir / (stem + ext)
        if candidate.is_file():
            return str(candidate)
    return ""

def capture_recent_tmux(session_name: str, window_name: str, lines: int):
    proc = subprocess.run(
        ["tmux", "capture-pane", "-p", "-t", session_name + ":" + window_name, "-S", "-" + str(max(1, lines))],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout

def tail_text_file(path: Path, max_lines: int):
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    return "\\n".join(lines[-max(1, max_lines):])

now = time.strftime("%Y-%m-%d %H:%M:%S")
status = f"[{{now}}] stopped (tmux session '{{session}}' not found)"
tmux_has = subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if tmux_has.returncode == 0:
    pane = subprocess.run(
        ["tmux", "display-message", "-p", "-t", f"{{session}}:gpu-pipeline", "#{{pane_dead}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    p = pane.stdout.strip() if pane.returncode == 0 else ""
    if p == "0":
        status = f"[{{now}}] running"
    elif p == "1":
        status = f"[{{now}}] session alive; gpu-pipeline pane exited"
    else:
        status = f"[{{now}}] session alive (gpu-pipeline pane state unavailable)"

gpu_ip = ""
if state_path.is_file():
    try:
        for raw in state_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if line.startswith("GPU_IP="):
                gpu_ip = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    except Exception:
        gpu_ip = ""

run_dir = latest_demo_run(demo_root)
input_path = latest_image(raw_dir)
if not input_path and run_dir is not None:
    input_path = latest_image(run_dir / "rgb")
overlay_path = ""
mask_path = ""
legend_rgb_path = ""
log_path = ""
if run_dir is not None:
    overlay_path = latest_image(run_dir / "segmentation_overlays")
    mask_path = latest_image(run_dir / "segmentation_masks")
    legend_anchor = mask_path or overlay_path
    if legend_anchor:
        legend_rgb_path = matching_rgb(run_dir, legend_anchor)
    log_candidate = run_dir / "gpu_pipeline.log"
    if log_candidate.is_file():
        log_path = str(log_candidate)
gpu_recent = tail_text_file(Path(log_path), 120) if log_path else capture_recent_tmux(session, "gpu-pipeline", 120)

payload = {{
    "status": status,
    "gpu_ip": gpu_ip,
    "state_file": str(state_path),
    "run_dir": str(run_dir) if run_dir is not None else "",
    "input_path": input_path,
    "overlay_path": overlay_path,
    "mask_path": mask_path,
    "legend_rgb_path": legend_rgb_path,
    "log_path": log_path,
    "gpu_recent": gpu_recent,
}}
print(json.dumps(payload))
"""
    script = "python3 - <<'PY'\n" + py + "\nPY"
    rc, out, err = _run_ssh_bash(cfg, script, timeout_s, text=True)
    out_s = str(out)
    err_s = str(err)
    if rc != 0:
        return {
            "status": f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ssh probe failed (exit {rc})",
            "error": _tail(err_s, 1200),
            "input_path": "",
            "overlay_path": "",
            "mask_path": "",
            "legend_rgb_path": "",
            "log_path": "",
            "gpu_recent": "",
            "gpu_ip": "",
            "state_file": state_file,
            "run_dir": "",
        }
    obj = _extract_json_from_output(out_s)
    if obj is None:
        return {
            "status": f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ssh probe parse failed",
            "error": _tail(out_s + "\n" + err_s, 1200),
            "input_path": "",
            "overlay_path": "",
            "mask_path": "",
            "legend_rgb_path": "",
            "log_path": "",
            "gpu_recent": "",
            "gpu_ip": "",
            "state_file": state_file,
            "run_dir": "",
        }
    return obj


def _remote_read_file_bytes(cfg: SshConfig, path: str, timeout_s: int = 25) -> tuple[Optional[bytes], str]:
    if not path.strip():
        return None, "(no file)"
    py = f"""
from pathlib import Path
import sys
p = Path({path!r}).expanduser()
if not p.is_file():
    raise SystemExit(2)
sys.stdout.buffer.write(p.read_bytes())
"""
    script = "python3 - <<'PY'\n" + py + "\nPY"
    try:
        rc, out, err = _run_ssh_bash(cfg, script, timeout_s, text=False)
    except subprocess.TimeoutExpired:
        return None, f"ssh read timed out after {timeout_s}s"
    out_b = out if isinstance(out, (bytes, bytearray)) else b""
    err_b = err if isinstance(err, (bytes, bytearray)) else b""
    if rc != 0:
        msg = err_b.decode("utf-8", errors="ignore").strip() or f"ssh read failed (exit {rc})"
        return None, _tail(msg, 1000)
    return bytes(out_b), ""


def _viewer_url_local(
    *,
    session: str,
    viewer_port: int,
    manual_url: str,
    state_file: str,
) -> tuple[str, str]:
    manual = manual_url.strip()
    if manual:
        return manual, "manual"
    if state_file.strip():
        sf = Path(state_file.strip()).expanduser()
    else:
        sf = Path.home() / f".vggt_active_gpu_{session}.env"
    values = _parse_env_file(sf)
    host = values.get("GPU_IP", "").strip() or "127.0.0.1"
    return f"http://{host}:{int(viewer_port)}", str(sf)


def poll_dashboard(
    session: str,
    demo_root: str,
    raw_dir: str,
    viewer_port: float,
    manual_viewer_url: str,
    state_file: str,
    execution_mode: str,
    ssh_host: str,
    ssh_user: str,
    ssh_key: str,
    ssh_options: str,
    local_viewer_port: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str, str, str, str, str]:
    mode = execution_mode.strip().lower()
    vp = int(viewer_port)
    local_vp = int(local_viewer_port)

    if mode == "ssh":
        cfg = SshConfig(host=ssh_host.strip(), user=ssh_user.strip(), key_path=ssh_key.strip(), options=ssh_options.strip())
        probe = _remote_probe(cfg, session, demo_root, raw_dir, state_file)

        input_img = None
        overlay_img = None
        mask_img = None
        legend_rgb_img = None
        input_info = probe.get("input_path", "") or "(no input image yet)"
        overlay_info = probe.get("overlay_path", "") or "(no overlay yet)"
        mask_info = probe.get("mask_path", "") or "(no mask yet)"

        if probe.get("input_path"):
            blob, msg = _remote_read_file_bytes(cfg, probe["input_path"])
            if blob is not None:
                input_img = _decode_image_bytes(blob)
                if input_img is None:
                    input_info = f"{probe['input_path']} (decode failed)"
            else:
                input_info = f"{probe['input_path']} ({msg})"

        if probe.get("overlay_path"):
            blob, msg = _remote_read_file_bytes(cfg, probe["overlay_path"])
            if blob is not None:
                overlay_img = _decode_image_bytes(blob)
                if overlay_img is None:
                    overlay_info = f"{probe['overlay_path']} (decode failed)"
            else:
                overlay_info = f"{probe['overlay_path']} ({msg})"

        if probe.get("mask_path"):
            blob, msg = _remote_read_file_bytes(cfg, probe["mask_path"])
            if blob is not None:
                mask_img = _decode_mask_bytes(blob)
                if mask_img is None:
                    mask_info = f"{probe['mask_path']} (decode failed)"
            else:
                mask_info = f"{probe['mask_path']} ({msg})"

        if probe.get("legend_rgb_path"):
            blob, msg = _remote_read_file_bytes(cfg, probe["legend_rgb_path"])
            if blob is not None:
                legend_rgb_img = _decode_image_bytes(blob)
                if legend_rgb_img is None:
                    probe["legend_rgb_path"] = f"{probe['legend_rgb_path']} (decode failed)"
            else:
                probe["legend_rgb_path"] = f"{probe['legend_rgb_path']} ({msg})"

        manual = manual_viewer_url.strip()
        viewer_source = ""
        if manual:
            viewer_url = manual
            viewer_source = "manual"
            viewer_panel = _viewer_html(viewer_url)
        else:
            gpu_ip = str(probe.get("gpu_ip", "")).strip()
            if gpu_ip:
                ok, msg = _ensure_tunnel(cfg, gpu_ip, vp, local_vp)
                if ok:
                    viewer_url = f"http://127.0.0.1:{local_vp}"
                    viewer_source = f"ssh tunnel ({msg}) via {cfg.target} -> {gpu_ip}:{vp}"
                    viewer_panel = _viewer_html(viewer_url)
                else:
                    viewer_source = f"ssh tunnel failed: {msg}"
                    viewer_panel = _viewer_placeholder(viewer_source)
            else:
                viewer_source = "missing GPU_IP in state file; waiting for pipeline allocation"
                viewer_panel = _viewer_placeholder(viewer_source)

        status = str(probe.get("status", f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] status unavailable"))
        extra_err = str(probe.get("error", "")).strip()
        stream_status = (
            f"mode=ssh target={cfg.target}\n"
            f"input={input_info}\n"
            f"overlay={overlay_info}\n"
            f"mask={mask_info}\n"
            f"log={probe.get('log_path', '(none yet)')}\n"
            f"viewer={viewer_source}\n"
            f"state_file={probe.get('state_file', state_file)}"
        )
        if extra_err:
            stream_status += f"\nprobe_error={extra_err}"
        semantic_status = _mask_summary(
            mask=mask_img,
            mask_path=str(probe.get("mask_path", "")).strip(),
            overlay_path=str(probe.get("overlay_path", "")).strip(),
            gpu_recent=str(probe.get("gpu_recent", "")),
        )
        legend_html = _legend_html(mask=mask_img, overlay_img=overlay_img, rgb_img=legend_rgb_img)
        return input_img, overlay_img, legend_html, viewer_panel, status, stream_status, semantic_status

    _stop_tunnel()
    demo_root_path = Path(demo_root).expanduser()
    raw_dir_path = Path(raw_dir).expanduser()
    input_img, input_src = _latest_image_in_dir(raw_dir_path)
    if input_img is None:
        input_img, input_src = _latest_processed_rgb(demo_root_path)
    overlay_img, overlay_src = _latest_overlay_image(demo_root_path)
    mask_src = _latest_mask_path(demo_root_path)
    mask_img = None
    if mask_src:
        try:
            mask_img = _decode_mask_bytes(Path(mask_src).read_bytes())
        except Exception:
            mask_img = None
            mask_src = f"{mask_src} (failed to read)"
    run_dir = _latest_demo_run_dir(demo_root_path)
    legend_rgb_img = None
    log_path = ""
    if run_dir is not None:
        legend_rgb_path = _matching_rgb_path(run_dir, mask_src if mask_src and " (" not in mask_src else overlay_src if overlay_src and " (" not in overlay_src else "")
        if legend_rgb_path:
            try:
                legend_rgb_img = _decode_image_bytes(Path(legend_rgb_path).read_bytes())
            except Exception:
                legend_rgb_img = None
        log_candidate = run_dir / "gpu_pipeline.log"
        if log_candidate.is_file():
            log_path = str(log_candidate)
    gpu_recent = _tail_text_file(Path(log_path), max_lines=120) if log_path else _capture_tmux_pane(session, "gpu-pipeline", lines=120)

    url, source = _viewer_url_local(
        session=session,
        viewer_port=vp,
        manual_url=manual_viewer_url,
        state_file=state_file,
    )
    status = _pipeline_status_text_local(session)
    stream_status = (
        f"mode=local\ninput={input_src}\noverlay={overlay_src}\nmask={mask_src or '(no mask yet)'}\n"
        f"log={log_path or '(none yet)'}\nviewer_source={source}"
    )
    semantic_status = _mask_summary(mask=mask_img, mask_path=mask_src, overlay_path=overlay_src, gpu_recent=gpu_recent)
    legend_html = _legend_html(mask=mask_img, overlay_img=overlay_img, rgb_img=legend_rgb_img)
    return input_img, overlay_img, legend_html, _viewer_html(url), status, stream_status, semantic_status


def start_pipeline(
    bashrc_shared: str,
    setup_script: str,
    session: str,
    checkpoint: str,
    demo_root: str,
    raw_dir: str,
    viewer_port: float,
    window_size: float,
    log_results: bool,
    existing_job_id: str,
    execution_mode: str,
    ssh_host: str,
    ssh_user: str,
    ssh_key: str,
    ssh_options: str,
) -> tuple[str, str]:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    mode = execution_mode.strip().lower()
    if not checkpoint.strip():
        return f"[{now}] start aborted", "checkpoint is required"

    setup_script = setup_script.strip()
    viewer_port_i = int(viewer_port)
    window_size_i = max(1, int(window_size))
    log_results_i = 1 if log_results else 0
    base_exports = [
        f"export VGGT_FINETUNE_CKPT={_quote(checkpoint.strip())}",
        f"export VGGT_DEMO_ROOT={_quote(demo_root.strip())}",
        "export VGGT_LOG_RESULTS=" + _quote(str(log_results_i)),
        f"export HPC_ROBOT_SAVE_DIR={_quote(raw_dir.strip())}",
        "export HPC_ROBOT_SAVE_EVERY=1",
        "export VGGT_VIS_MAP=1",
        f"export VGGT_VISER_PORT={_quote(str(viewer_port_i))}",
        f"export VGGT_WINDOW_SIZE={_quote(str(window_size_i))}",
    ]

    script_lines = [
        "set -euo pipefail",
        f"source {_quote(bashrc_shared.strip())}",
    ]
    script_lines.extend(base_exports)
    if setup_script:
        script_lines.append(f"source {_quote(setup_script)}")
    # Re-export UI-controlled values after sourcing the setup script so the
    # script can derive dependent vars from the checkpoint while the UI keeps
    # final control over the launch-critical paths.
    script_lines.extend(base_exports)
    job_id = existing_job_id.strip()
    job_arg = f" --job-id {_quote(job_id)}" if job_id else ""
    script_lines.append(
        "start_robot_pipeline_tmux -k --no-attach "
        f"--session {_quote(session.strip())} "
        "--checkpoint \"$VGGT_FINETUNE_CKPT\" "
        "--demo-root \"$VGGT_DEMO_ROOT\" "
        "--log-results \"$VGGT_LOG_RESULTS\" "
        "--max-live-steps \"${VGGT_MAX_LIVE_STEPS:-0}\""
        f"{job_arg}"
    )
    script = "; ".join(script_lines)

    if mode == "ssh":
        cfg = SshConfig(host=ssh_host.strip(), user=ssh_user.strip(), key_path=ssh_key.strip(), options=ssh_options.strip())
        if not cfg.host:
            return f"[{now}] start aborted", "ssh host is required for execution_mode=ssh"
        _stop_tunnel()
        try:
            rc, out, err = _run_ssh_bash(cfg, script, timeout_s=300, text=True)
        except subprocess.TimeoutExpired:
            return f"[{now}] start timeout", "start command timed out over ssh"
        log_text = (
            f"mode=ssh target={cfg.target}\n"
            f"exit_code={rc}\n"
            f"stdout:\n{_tail(str(out))}\n"
            f"stderr:\n{_tail(str(err))}"
        )
        if rc != 0:
            return f"[{now}] start failed (exit {rc})", log_text
        probe = _remote_probe(cfg, session, demo_root, raw_dir, "")
        status = str(probe.get("status", f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] status unavailable"))
        if "stopped (tmux session" in status:
            status = f"[{now}] start finished but tmux session '{session}' was not created"
        return status, log_text

    bashrc_path = Path(bashrc_shared).expanduser()
    if not bashrc_path.is_file():
        return f"[{now}] start aborted", f"missing bashrc_shared at {bashrc_path}"
    if setup_script:
        setup_path = Path(setup_script).expanduser()
        if not setup_path.is_file():
            return f"[{now}] start aborted", f"setup script not found: {setup_path}"
    script_local = script.replace(bashrc_shared.strip(), str(bashrc_path))
    try:
        rc, out, err = _run_local_bash(script_local, timeout_s=240, text=True)
    except subprocess.TimeoutExpired:
        return f"[{now}] start timeout", "start command timed out"
    log_text = (
        f"mode=local\n"
        f"exit_code={rc}\n"
        f"stdout:\n{_tail(str(out))}\n"
        f"stderr:\n{_tail(str(err))}"
    )
    if rc != 0:
        return f"[{now}] start failed (exit {rc})", log_text
    status = _pipeline_status_text_local(session)
    if "stopped (tmux session" in status:
        status = f"[{now}] start finished but tmux session '{session}' was not created"
    return status, log_text


def stop_pipeline(
    bashrc_shared: str,
    session: str,
    stop_timeout_s: float,
    execution_mode: str,
    ssh_host: str,
    ssh_user: str,
    ssh_key: str,
    ssh_options: str,
) -> tuple[str, str]:
    mode = execution_mode.strip().lower()
    timeout_i = max(1, int(stop_timeout_s))
    script = (
        "set -euo pipefail; "
        f"source {_quote(bashrc_shared.strip())}; "
        f"stop_robot_pipeline_graceful {_quote(session.strip())} {_quote(str(timeout_i))}"
    )

    if mode == "ssh":
        cfg = SshConfig(host=ssh_host.strip(), user=ssh_user.strip(), key_path=ssh_key.strip(), options=ssh_options.strip())
        if not cfg.host:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            return f"[{now}] stop aborted", "ssh host is required for execution_mode=ssh"
        try:
            rc, out, err = _run_ssh_bash(cfg, script, timeout_s=max(90, timeout_i + 40), text=True)
        except subprocess.TimeoutExpired:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            return f"[{now}] stop timeout", "stop command timed out over ssh"
        probe = _remote_probe(cfg, session, "/tmp", "/tmp", "")
        status = str(probe.get("status", f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] status unavailable"))
        _stop_tunnel()
        return status, (
            f"mode=ssh target={cfg.target}\n"
            f"exit_code={rc}\n"
            f"stdout:\n{_tail(str(out))}\n"
            f"stderr:\n{_tail(str(err))}"
        )

    bashrc_path = Path(bashrc_shared).expanduser()
    if not bashrc_path.is_file():
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{now}] stop aborted", f"missing bashrc_shared at {bashrc_path}"
    script_local = script.replace(bashrc_shared.strip(), str(bashrc_path))
    try:
        rc, out, err = _run_local_bash(script_local, timeout_s=max(60, timeout_i + 30), text=True)
    except subprocess.TimeoutExpired:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{now}] stop timeout", "stop command timed out"
    _stop_tunnel()
    return _pipeline_status_text_local(session), (
        f"mode=local\n"
        f"exit_code={rc}\n"
        f"stdout:\n{_tail(str(out))}\n"
        f"stderr:\n{_tail(str(err))}"
    )


def start_robot_image_stream(
    robot_host: str,
    robot_user: str,
    robot_ssh_key: str,
    robot_ssh_options: str,
    robot_session: str,
    robot_interface: str,
    robot_streamer_path: str,
    robot_stream_port: float,
    robot_stream_fps: float,
    robot_hpc_key: str,
    ssh_host: str,
    ssh_user: str,
) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cfg = SshConfig(
        host=robot_host.strip(),
        user=robot_user.strip(),
        key_path=robot_ssh_key.strip(),
        options=robot_ssh_options.strip(),
    )
    if not cfg.host:
        return f"[{now}] robot stream start aborted\nmissing robot ssh host"
    if not ssh_host.strip() or not ssh_user.strip():
        return f"[{now}] robot stream start aborted\nmissing HPC head host/user for reverse tunnel"

    session_name = robot_session.strip() or "robot_image_stream"
    robot_port = max(1, int(robot_stream_port))
    robot_fps = max(0.1, float(robot_stream_fps))
    robot_iface = robot_interface.strip() or "eth0"
    streamer_path = robot_streamer_path.strip() or "~/Workspace/gio_ws/unitree_robot_tcp_streamer.py"

    streamer_cmd = (
        "bash -lc "
        + _quote(
            "set -euo pipefail; "
            "source ~/venvs/unitree_sdk2/bin/activate; "
            f"python3 {_shell_path_arg(streamer_path)} "
            f"--interface {_quote(robot_iface)} "
            "--bind 127.0.0.1 "
            f"--port {_quote(str(robot_port))} "
            f"--fps {_quote(str(robot_fps))} "
            "--timeout 3.0"
        )
    )

    tunnel_args = [
        "ssh",
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if robot_hpc_key.strip():
        tunnel_args.extend(["-i", robot_hpc_key.strip()])
    tunnel_args.extend(
        [
            "-R",
            f"127.0.0.1:{robot_port}:127.0.0.1:{robot_port}",
            f"{ssh_user.strip()}@{ssh_host.strip()}",
        ]
    )
    tunnel_cmd = "bash -lc " + _quote("set -euo pipefail; " + _shell_join(tunnel_args))

    remote_script = "; ".join(
        [
            "set -euo pipefail",
            "if ! command -v tmux >/dev/null 2>&1; then echo 'tmux not found on robot' >&2; exit 1; fi",
            f"if tmux has-session -t {_quote(session_name)} 2>/dev/null; then tmux kill-session -t {_quote(session_name)}; fi",
            f"tmux new-session -d -s {_quote(session_name)} -n streamer",
            f"tmux set-option -t {_quote(session_name)} remain-on-exit on",
            f"tmux new-window -t {_quote(session_name)} -n tunnel",
            f"tmux send-keys -t {_quote(session_name + ':streamer')} {_quote(streamer_cmd)} C-m",
            f"tmux send-keys -t {_quote(session_name + ':tunnel')} {_quote(tunnel_cmd)} C-m",
            f"echo session={_quote(session_name)}",
            "echo windows=streamer,tunnel",
            f"echo streamer_port={_quote(str(robot_port))}",
            f"echo hpc_head={_quote(ssh_host.strip())}",
        ]
    )

    try:
        rc, out, err = _run_ssh_bash(cfg, remote_script, timeout_s=120, text=True)
    except subprocess.TimeoutExpired:
        return f"[{now}] robot stream start timeout\ntimed out while contacting {cfg.target}"
    return (
        f"[{now}] robot stream start "
        + ("ok" if rc == 0 else f"failed (exit {rc})")
        + "\n"
        + f"mode=ssh target={cfg.target}\n"
        + f"exit_code={rc}\n"
        + f"stdout:\n{_tail(str(out))}\n"
        + f"stderr:\n{_tail(str(err))}"
    )


def stop_robot_image_stream(
    robot_host: str,
    robot_user: str,
    robot_ssh_key: str,
    robot_ssh_options: str,
    robot_session: str,
) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cfg = SshConfig(
        host=robot_host.strip(),
        user=robot_user.strip(),
        key_path=robot_ssh_key.strip(),
        options=robot_ssh_options.strip(),
    )
    if not cfg.host:
        return f"[{now}] robot stream stop aborted\nmissing robot ssh host"

    session_name = robot_session.strip() or "robot_image_stream"
    remote_script = "; ".join(
        [
            "set -euo pipefail",
            f"if tmux has-session -t {_quote(session_name)} 2>/dev/null; then tmux kill-session -t {_quote(session_name)}; echo 'killed'; else echo 'session not found'; fi",
        ]
    )
    try:
        rc, out, err = _run_ssh_bash(cfg, remote_script, timeout_s=60, text=True)
    except subprocess.TimeoutExpired:
        return f"[{now}] robot stream stop timeout\ntimed out while contacting {cfg.target}"
    return (
        f"[{now}] robot stream stop "
        + ("ok" if rc == 0 else f"failed (exit {rc})")
        + "\n"
        + f"mode=ssh target={cfg.target}\n"
        + f"exit_code={rc}\n"
        + f"stdout:\n{_tail(str(out))}\n"
        + f"stderr:\n{_tail(str(err))}"
    )


def build_app(
    *,
    bashrc_shared_default: str,
    setup_script_default: str,
    session_default: str,
    checkpoint_default: str,
    demo_root_default: str,
    raw_dir_default: str,
    viewer_port_default: int,
    window_size_default: int,
    existing_job_id_default: str,
    robot_host_default: str,
    robot_user_default: str,
    robot_ssh_key_default: str,
    robot_ssh_options_default: str,
    robot_session_default: str,
    robot_interface_default: str,
    robot_streamer_path_default: str,
    robot_stream_port_default: int,
    robot_stream_fps_default: float,
    robot_hpc_key_default: str,
    state_file_default: str,
    execution_mode_default: str,
    ssh_host_default: str,
    ssh_user_default: str,
    ssh_key_default: str,
    ssh_options_default: str,
    local_viewer_port_default: int,
):
    _prune_incompatible_user_site_paths()
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(
            "Failed to import gradio in the current Python environment.\n"
            f"python={sys.executable} version={sys.version.split()[0]}\n"
            "Recommended fix:\n"
            "  PYTHONNOUSERSITE=1 python3 -m pip install -U gradio numpy opencv-python pillow\n"
            "  PYTHONNOUSERSITE=1 python3 /Users/giovannichiementin/Desktop/Thesis/VGGT-SLAM/scripts/live_dashboard.py ...\n"
            f"Original import error: {exc}"
        ) from exc

    with gr.Blocks(title="VGGT Live Dashboard") as demo:
        gr.Markdown("## VGGT Live Dashboard")

        with gr.Row():
            input_img = gr.Image(label="Input RGB", type="numpy", format="png")
            overlay_img = gr.Image(label="Semantic Overlay", type="numpy", format="png")

        legend_html = gr.HTML(label="Overlay Legend")

        with gr.Row():
            viewer_html = gr.HTML(label="3D Map Viewer")

        with gr.Row():
            start_btn = gr.Button("Start Pipeline", variant="primary")
            stop_btn = gr.Button("Stop Pipeline", variant="stop")
            start_stream_btn = gr.Button("Start Image Stream")
            stop_stream_btn = gr.Button("Stop Image Stream")
            refresh_btn = gr.Button("Refresh Now")

        with gr.Row():
            pipeline_status = gr.Textbox(label="Pipeline Status", lines=4, interactive=False)
            stream_status = gr.Textbox(label="Stream Status", lines=4, interactive=False)

        with gr.Row():
            semantic_status = gr.Textbox(label="Semantic Diagnostics", lines=10, interactive=False)
            action_log = gr.Textbox(label="Command Output", lines=10, interactive=False)

        with gr.Accordion("Runtime Configuration", open=True):
            execution_mode = gr.Radio(
                label="Execution mode",
                choices=["ssh", "local"],
                value=execution_mode_default,
            )
            ssh_host = gr.Textbox(label="SSH head host", value=ssh_host_default)
            ssh_user = gr.Textbox(label="SSH user", value=ssh_user_default)
            ssh_key = gr.Textbox(label="SSH key (optional)", value=ssh_key_default)
            ssh_options = gr.Textbox(
                label="Extra SSH options (optional)",
                value=ssh_options_default,
                placeholder="-p 22 -J jump-host",
            )
            local_viewer_port = gr.Number(
                label="Local forwarded viewer port (Mac)",
                value=local_viewer_port_default,
                precision=0,
            )

            bashrc_shared = gr.Textbox(
                label="bashrc_shared path (remote path in ssh mode)",
                value=bashrc_shared_default,
            )
            setup_script = gr.Textbox(
                label="Optional setup script with exports (remote path in ssh mode)",
                value=setup_script_default,
            )
            session = gr.Textbox(label="tmux session", value=session_default)
            existing_job_id = gr.Textbox(
                label="Existing Slurm job id (optional)",
                value=existing_job_id_default,
                placeholder="Reuse a RUNNING salloc allocation, e.g. 463349",
            )
            checkpoint = gr.Textbox(label="Checkpoint path", value=checkpoint_default)
            demo_root = gr.Textbox(label="Demo root path", value=demo_root_default)
            raw_dir = gr.Textbox(label="Raw RGB cache directory", value=raw_dir_default)

            with gr.Row():
                viewer_port = gr.Number(label="Remote Viser port", value=viewer_port_default, precision=0)
                window_size = gr.Number(label="VGGT window size", value=window_size_default, precision=0)
                log_results = gr.Checkbox(label="VGGT log_results", value=False)
                stop_timeout = gr.Number(label="Stop timeout (s)", value=120, precision=0)

            manual_viewer_url = gr.Textbox(
                label="Optional viewer URL override",
                value="",
                placeholder="e.g. http://127.0.0.1:18080",
            )
            state_file = gr.Textbox(
                label="Optional state file override (remote in ssh mode)",
                value=state_file_default,
                placeholder="default: ~/.vggt_active_gpu_<session>.env",
            )

            with gr.Row():
                robot_host = gr.Textbox(label="Robot SSH host", value=robot_host_default)
                robot_user = gr.Textbox(label="Robot SSH user", value=robot_user_default)
                robot_ssh_key = gr.Textbox(label="Robot SSH key (optional)", value=robot_ssh_key_default)
            robot_ssh_options = gr.Textbox(
                label="Robot SSH options (optional)",
                value=robot_ssh_options_default,
                placeholder="-p 22",
            )
            with gr.Row():
                robot_session = gr.Textbox(label="Robot tmux session", value=robot_session_default)
                robot_interface = gr.Textbox(label="Robot camera interface", value=robot_interface_default)
                robot_hpc_key = gr.Textbox(label="Robot->HPC SSH key (optional)", value=robot_hpc_key_default)
            with gr.Row():
                robot_streamer_path = gr.Textbox(label="Robot TCP streamer path", value=robot_streamer_path_default)
                robot_stream_port = gr.Number(label="Robot/HPC stream port", value=robot_stream_port_default, precision=0)
                robot_stream_fps = gr.Number(label="Robot stream FPS", value=robot_stream_fps_default)

        poll_inputs = [
            session,
            demo_root,
            raw_dir,
            viewer_port,
            manual_viewer_url,
            state_file,
            execution_mode,
            ssh_host,
            ssh_user,
            ssh_key,
            ssh_options,
            local_viewer_port,
        ]
        poll_outputs = [input_img, overlay_img, legend_html, viewer_html, pipeline_status, stream_status, semantic_status]

        timer = gr.Timer(value=POLL_SEC)
        timer.tick(fn=poll_dashboard, inputs=poll_inputs, outputs=poll_outputs)
        refresh_btn.click(fn=poll_dashboard, inputs=poll_inputs, outputs=poll_outputs)

        start_btn.click(
            fn=start_pipeline,
            inputs=[
                bashrc_shared,
                setup_script,
                session,
                checkpoint,
                demo_root,
                raw_dir,
                viewer_port,
                window_size,
                log_results,
                existing_job_id,
                execution_mode,
                ssh_host,
                ssh_user,
                ssh_key,
                ssh_options,
            ],
            outputs=[pipeline_status, action_log],
        )
        stop_btn.click(
            fn=stop_pipeline,
            inputs=[
                bashrc_shared,
                session,
                stop_timeout,
                execution_mode,
                ssh_host,
                ssh_user,
                ssh_key,
                ssh_options,
            ],
            outputs=[pipeline_status, action_log],
        )
        start_stream_btn.click(
            fn=start_robot_image_stream,
            inputs=[
                robot_host,
                robot_user,
                robot_ssh_key,
                robot_ssh_options,
                robot_session,
                robot_interface,
                robot_streamer_path,
                robot_stream_port,
                robot_stream_fps,
                robot_hpc_key,
                ssh_host,
                ssh_user,
            ],
            outputs=[action_log],
        )
        stop_stream_btn.click(
            fn=stop_robot_image_stream,
            inputs=[
                robot_host,
                robot_user,
                robot_ssh_key,
                robot_ssh_options,
                robot_session,
            ],
            outputs=[action_log],
        )

        demo.load(fn=poll_dashboard, inputs=poll_inputs, outputs=poll_outputs)

    return demo


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    local_default_bashrc = str((repo_root / "shell" / "bashrc_shared").resolve())
    local_default_demo = str((repo_root / "demo").resolve())
    local_default_raw = str((repo_root / "robot_probe_frames").resolve())

    default_mode = os.getenv("VGGT_DASHBOARD_EXEC_MODE", "ssh").strip().lower()
    if default_mode not in ("ssh", "local"):
        default_mode = "ssh"

    default_ssh_host = os.getenv("VGGT_DASHBOARD_SSH_HOST", "hpc-head1.ewi.utwente.nl")
    default_ssh_user = os.getenv("VGGT_DASHBOARD_SSH_USER", "s2984792")
    default_robot_host = os.getenv("VGGT_DASHBOARD_ROBOT_HOST", "192.168.225.122")
    default_robot_user = os.getenv("VGGT_DASHBOARD_ROBOT_USER", "unitree")
    default_remote_root = os.getenv(
        "VGGT_DASHBOARD_REMOTE_ROOT",
        f"/home/{default_ssh_user}/src/VGGT-SLAM" if default_ssh_user else "~/src/VGGT-SLAM",
    )

    default_bashrc = os.getenv(
        "VGGT_DASHBOARD_BASHRC_SHARED",
        f"{default_remote_root}/shell/bashrc_shared" if default_mode == "ssh" else local_default_bashrc,
    )
    default_setup_script = os.getenv(
        "VGGT_DASHBOARD_SETUP_SCRIPT",
        f"{default_remote_root}/shell/dashboard_env.sh" if default_mode == "ssh" else str((repo_root / "shell" / "dashboard_env.sh").resolve()),
    )
    default_demo = os.getenv(
        "VGGT_DEMO_ROOT",
        f"{default_remote_root}/demo" if default_mode == "ssh" else local_default_demo,
    )
    default_raw = os.getenv(
        "HPC_ROBOT_SAVE_DIR",
        f"{default_remote_root}/robot_probe_frames" if default_mode == "ssh" else local_default_raw,
    )

    parser = argparse.ArgumentParser(description="VGGT live UI (RGB + map + overlay + start/stop)")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--execution-mode", choices=["ssh", "local"], default=default_mode)
    parser.add_argument("--ssh-host", default=default_ssh_host, help="SSH host (head node)")
    parser.add_argument("--ssh-user", default=default_ssh_user, help="SSH username")
    parser.add_argument("--ssh-key", default=os.getenv("VGGT_DASHBOARD_SSH_KEY", ""), help="SSH key path (optional)")
    parser.add_argument("--ssh-options", default=os.getenv("VGGT_DASHBOARD_SSH_OPTIONS", ""), help="Extra SSH options")
    parser.add_argument("--local-viewer-port", type=int, default=int(os.getenv("VGGT_LOCAL_VISER_PORT", "18080")))

    parser.add_argument("--bashrc-shared", default=default_bashrc, help="Path to bashrc_shared")
    parser.add_argument("--setup-script", default=default_setup_script, help="Optional setup script")
    parser.add_argument("--session", default=os.getenv("VGGT_DASHBOARD_SESSION", "robot_pipeline"), help="tmux session")
    parser.add_argument("--checkpoint", default=os.getenv("VGGT_FINETUNE_CKPT", ""), help="Checkpoint path")
    parser.add_argument("--demo-root", default=default_demo, help="Demo root")
    parser.add_argument("--raw-dir", default=default_raw, help="Raw RGB cache directory")
    parser.add_argument("--viewer-port", type=int, default=int(os.getenv("VGGT_VISER_PORT", "8080")), help="Remote Viser port")
    parser.add_argument("--window-size", type=int, default=int(os.getenv("VGGT_WINDOW_SIZE", "15")), help="VGGT live window size")
    parser.add_argument(
        "--existing-job-id",
        default=os.getenv("VGGT_EXISTING_GPU_JOB_ID", os.getenv("HPC_EXISTING_GPU_JOB_ID", "")),
        help="Optional existing RUNNING Slurm job id created via salloc",
    )
    parser.add_argument("--robot-host", default=default_robot_host, help="Robot SSH host")
    parser.add_argument("--robot-user", default=default_robot_user, help="Robot SSH username")
    parser.add_argument("--robot-ssh-key", default=os.getenv("VGGT_DASHBOARD_ROBOT_SSH_KEY", ""), help="Robot SSH key path (optional)")
    parser.add_argument("--robot-ssh-options", default=os.getenv("VGGT_DASHBOARD_ROBOT_SSH_OPTIONS", ""), help="Extra SSH options for robot SSH")
    parser.add_argument("--robot-session", default=os.getenv("VGGT_DASHBOARD_ROBOT_SESSION", "robot_image_stream"), help="Robot tmux session name")
    parser.add_argument("--robot-interface", default=os.getenv("VGGT_DASHBOARD_ROBOT_INTERFACE", "eth0"), help="Robot camera interface")
    parser.add_argument(
        "--robot-streamer-path",
        default=os.getenv("VGGT_DASHBOARD_ROBOT_STREAMER_PATH", "~/Workspace/gio_ws/unitree_robot_tcp_streamer.py"),
        help="Robot-side TCP streamer path",
    )
    parser.add_argument("--robot-stream-port", type=int, default=int(os.getenv("VGGT_DASHBOARD_ROBOT_STREAM_PORT", "15001")), help="Robot/HPC TCP stream port")
    parser.add_argument("--robot-stream-fps", type=float, default=float(os.getenv("VGGT_DASHBOARD_ROBOT_STREAM_FPS", "2.0")), help="Robot TCP stream FPS")
    parser.add_argument("--robot-hpc-key", default=os.getenv("VGGT_DASHBOARD_ROBOT_HPC_KEY", ""), help="Private key path on the robot for SSH to the HPC head")
    parser.add_argument("--state-file", default=os.getenv("VGGT_GPU_STATE_FILE", ""), help="State file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(
        bashrc_shared_default=args.bashrc_shared,
        setup_script_default=args.setup_script,
        session_default=args.session,
        checkpoint_default=args.checkpoint,
        demo_root_default=args.demo_root,
        raw_dir_default=args.raw_dir,
        viewer_port_default=args.viewer_port,
        window_size_default=args.window_size,
        existing_job_id_default=args.existing_job_id,
        robot_host_default=args.robot_host,
        robot_user_default=args.robot_user,
        robot_ssh_key_default=args.robot_ssh_key,
        robot_ssh_options_default=args.robot_ssh_options,
        robot_session_default=args.robot_session,
        robot_interface_default=args.robot_interface,
        robot_streamer_path_default=args.robot_streamer_path,
        robot_stream_port_default=args.robot_stream_port,
        robot_stream_fps_default=args.robot_stream_fps,
        robot_hpc_key_default=args.robot_hpc_key,
        state_file_default=args.state_file,
        execution_mode_default=args.execution_mode,
        ssh_host_default=args.ssh_host,
        ssh_user_default=args.ssh_user,
        ssh_key_default=args.ssh_key,
        ssh_options_default=args.ssh_options,
        local_viewer_port_default=args.local_viewer_port,
    )
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
