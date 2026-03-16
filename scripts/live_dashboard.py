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


def _tail(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return f"...(truncated {len(text) - limit} chars)\n{text[-limit:]}"


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
        "<div style='height:460px; border:1px solid #d0d7de; border-radius:8px; overflow:hidden;'>"
        f"<iframe src='{safe_url}' style='width:100%; height:100%; border:0;'></iframe>"
        "</div>"
        f"<div style='margin-top:8px; font-family:monospace; font-size:12px;'>viewer: {safe_url}</div>"
    )


def _viewer_placeholder(message: str) -> str:
    msg = html.escape(message)
    return (
        "<div style='height:460px; border:1px dashed #d0d7de; border-radius:8px; "
        "display:flex; align-items:center; justify-content:center; padding:12px; "
        "font-family:monospace; font-size:12px;'>"
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


def _latest_image_in_dir(path: Path) -> tuple[Optional[np.ndarray], str]:
    if not path.is_dir():
        return None, f"{path} (missing)"
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
if run_dir is not None:
    overlay_path = latest_image(run_dir / "segmentation_overlays")

payload = {{
    "status": status,
    "gpu_ip": gpu_ip,
    "state_file": str(state_path),
    "run_dir": str(run_dir) if run_dir is not None else "",
    "input_path": input_path,
    "overlay_path": overlay_path,
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
    rc, out, err = _run_ssh_bash(cfg, script, timeout_s, text=False)
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
) -> tuple[Optional[np.ndarray], str, Optional[np.ndarray], str, str]:
    mode = execution_mode.strip().lower()
    vp = int(viewer_port)
    local_vp = int(local_viewer_port)

    if mode == "ssh":
        cfg = SshConfig(host=ssh_host.strip(), user=ssh_user.strip(), key_path=ssh_key.strip(), options=ssh_options.strip())
        probe = _remote_probe(cfg, session, demo_root, raw_dir, state_file)

        input_img = None
        overlay_img = None
        input_info = probe.get("input_path", "") or "(no input image yet)"
        overlay_info = probe.get("overlay_path", "") or "(no overlay yet)"

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
            f"viewer={viewer_source}\n"
            f"state_file={probe.get('state_file', state_file)}"
        )
        if extra_err:
            stream_status += f"\nprobe_error={extra_err}"
        return input_img, viewer_panel, overlay_img, status, stream_status

    _stop_tunnel()
    demo_root_path = Path(demo_root).expanduser()
    raw_dir_path = Path(raw_dir).expanduser()
    input_img, input_src = _latest_image_in_dir(raw_dir_path)
    if input_img is None:
        input_img, input_src = _latest_processed_rgb(demo_root_path)
    overlay_img, overlay_src = _latest_overlay_image(demo_root_path)

    url, source = _viewer_url_local(
        session=session,
        viewer_port=vp,
        manual_url=manual_viewer_url,
        state_file=state_file,
    )
    status = _pipeline_status_text_local(session)
    stream_status = f"mode=local\ninput={input_src}\noverlay={overlay_src}\nviewer_source={source}"
    return input_img, _viewer_html(url), overlay_img, status, stream_status


def start_pipeline(
    bashrc_shared: str,
    setup_script: str,
    session: str,
    checkpoint: str,
    demo_root: str,
    raw_dir: str,
    viewer_port: float,
    log_results: bool,
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
    log_results_i = 1 if log_results else 0

    script_lines = [
        "set -euo pipefail",
        f"source {_quote(bashrc_shared.strip())}",
    ]
    if setup_script:
        script_lines.append(f"source {_quote(setup_script)}")
    script_lines.extend(
        [
            f"export VGGT_FINETUNE_CKPT={_quote(checkpoint.strip())}",
            f"export VGGT_DEMO_ROOT={_quote(demo_root.strip())}",
            f"export HPC_ROBOT_SAVE_DIR={_quote(raw_dir.strip())}",
            "export HPC_ROBOT_SAVE_EVERY=1",
            "export VGGT_VIS_MAP=1",
            f"export VGGT_VISER_PORT={_quote(str(viewer_port_i))}",
            (
                "start_robot_pipeline_tmux -k --no-attach "
                f"--session {_quote(session.strip())} "
                f"--checkpoint {_quote(checkpoint.strip())} "
                f"--demo-root {_quote(demo_root.strip())} "
                f"--log-results {log_results_i}"
            ),
        ]
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


def build_app(
    *,
    bashrc_shared_default: str,
    setup_script_default: str,
    session_default: str,
    checkpoint_default: str,
    demo_root_default: str,
    raw_dir_default: str,
    viewer_port_default: int,
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
        gr.Markdown(
            "## VGGT Live Dashboard\n"
            "Run this UI on your Mac. In `ssh` mode, control commands and file polling happen over SSH on the HPC head node. "
            "The map pane uses an automatic SSH local port forward from Mac -> head -> GPU."
        )

        with gr.Row():
            start_btn = gr.Button("Start Pipeline", variant="primary")
            stop_btn = gr.Button("Stop Pipeline", variant="stop")
            refresh_btn = gr.Button("Refresh Now")

        with gr.Row():
            pipeline_status = gr.Textbox(label="Pipeline Status", lines=2, interactive=False)
            stream_status = gr.Textbox(label="Stream Status", lines=5, interactive=False)

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
            checkpoint = gr.Textbox(label="Checkpoint path", value=checkpoint_default)
            demo_root = gr.Textbox(label="Demo root path", value=demo_root_default)
            raw_dir = gr.Textbox(label="Raw RGB cache directory", value=raw_dir_default)

            with gr.Row():
                viewer_port = gr.Number(label="Remote Viser port", value=viewer_port_default, precision=0)
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
            input_img = gr.Image(label="Input RGB", type="numpy", format="png")
            viewer_html = gr.HTML(label="Live Map Viewer")
            overlay_img = gr.Image(label="Semantic Overlay", type="numpy", format="png")

        action_log = gr.Textbox(label="Start/Stop Command Output", lines=14, interactive=False)

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
        poll_outputs = [input_img, viewer_html, overlay_img, pipeline_status, stream_status]

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
                log_results,
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
    default_remote_root = os.getenv(
        "VGGT_DASHBOARD_REMOTE_ROOT",
        f"/home/{default_ssh_user}/src/VGGT-SLAM" if default_ssh_user else "~/src/VGGT-SLAM",
    )

    default_bashrc = os.getenv(
        "VGGT_DASHBOARD_BASHRC_SHARED",
        f"{default_remote_root}/shell/bashrc_shared" if default_mode == "ssh" else local_default_bashrc,
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
    parser.add_argument("--setup-script", default=os.getenv("VGGT_DASHBOARD_SETUP_SCRIPT", ""), help="Optional setup script")
    parser.add_argument("--session", default=os.getenv("VGGT_DASHBOARD_SESSION", "robot_pipeline"), help="tmux session")
    parser.add_argument("--checkpoint", default=os.getenv("VGGT_FINETUNE_CKPT", ""), help="Checkpoint path")
    parser.add_argument("--demo-root", default=default_demo, help="Demo root")
    parser.add_argument("--raw-dir", default=default_raw, help="Raw RGB cache directory")
    parser.add_argument("--viewer-port", type=int, default=int(os.getenv("VGGT_VISER_PORT", "8080")), help="Remote Viser port")
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
