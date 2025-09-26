# taps_runtime.py
import os, json, time, threading
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- small async logger to avoid blocking the model thread ----
class _AsyncWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict):
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with self._lock:
            with self.path.open("a") as f:
                f.write(line)

def _tensor_stats(x: torch.Tensor):
    return {
        "shape": list(x.shape),
        "mean": float(x.detach().mean().item()),
        "std": float(x.detach().std().item()),
        "min": float(x.detach().amin().item()),
        "max": float(x.detach().amax().item()),
        "dtype": str(x.dtype).replace("torch.", ""),
        "device": str(x.device),
    }

class VGGTFeatureTapper:
    """
    Registers forward hooks on a VGGT model to capture:
      - depth DPT refinenet{1..4}.out_conv -> p2..p5-like maps
      - DINO block tokens -> trimmed to patch tokens, reshaped to (B, C, Htok, Wtok), 1x1->256
    Writes JSON lines with shapes+stats to tap_logs/taps.jsonl (configurable).
    """
    def __init__(self, model: nn.Module, logdir="tap_logs", capture_every=1, save_small_tensors=False):
        self.model = model
        self.capture_every = max(1, int(capture_every))
        self.save_small = bool(save_small_tensors)
        self.writer = _AsyncWriter(Path(logdir) / "taps.jsonl")
        self._handles = []
        self._cache = {}
        self._step = 0
        self._initialized_proj = False
        self._dino_proj = None
        self._patch_size = 14  # VGGT/DINOv2 default in your runs

        # Map names for depth DPT
        self.dpt_taps = {
            "p2_src": "depth_head.scratch.refinenet1.out_conv",  # finest
            "p3_src": "depth_head.scratch.refinenet2.out_conv",
            "p4_src": "depth_head.scratch.refinenet3.out_conv",
            "p5_src": "depth_head.scratch.refinenet4.out_conv",  # coarsest
        }
        # Candidate DINO taps (use the deepest that exists)
        self.dino_candidates = [
            "aggregator.patch_embed.blocks.23",
            "aggregator.patch_embed.blocks.21",
            "aggregator.patch_embed.blocks.15",
        ]
        self.dino_tap = None

        # Register hooks
        modmap = {n: m for n, m in self.model.named_modules()}
        # DPT hooks
        for nm in self.dpt_taps.values():
            if nm not in modmap:
                raise KeyError(f"[VGGTFeatureTapper] Missing DPT tap: {nm}")
            self._handles.append(modmap[nm].register_forward_hook(self._hook_tensor(nm)))
        # DINO hook (pick first available)
        for cand in self.dino_candidates:
            if cand in modmap:
                self.dino_tap = cand
                self._handles.append(modmap[cand].register_forward_hook(self._hook_tensor(cand)))
                break
        if self.dino_tap is None:
            raise KeyError("[VGGTFeatureTapper] No DINO block candidate found.")

        # Also hook early patch embed to capture input H,W once per step (optional)
        pe_name = "aggregator.patch_embed.patch_embed"
        if pe_name in modmap:
            self._handles.append(modmap[pe_name].register_forward_pre_hook(self._hook_input_hw(pe_name)))

    def _hook_tensor(self, name):
        def fn(m, inp, out):
            # normalize to Tensor
            if isinstance(out, torch.Tensor):
                self._cache[name] = out
            elif isinstance(out, (list, tuple)):
                for x in out:
                    if torch.is_tensor(x):
                        self._cache[name] = x; break
            elif isinstance(out, dict):
                for v in out.values():
                    if torch.is_tensor(v):
                        self._cache[name] = v; break
        return fn

    def _hook_input_hw(self, name):
        def fn(m, inp):
            # record input H,W from first tensor arg if present
            if len(inp) and torch.is_tensor(inp[0]) and inp[0].dim() >= 4:
                x = inp[0]
                self._cache["_hw"] = (int(x.shape[-2]), int(x.shape[-1]))
        return fn

    @torch.no_grad()
    def maybe_log(self):
        """Call this once per outer step; will log every capture_every steps."""
        self._step += 1
        if (self._step % self.capture_every) != 0:
            self._cache.clear()
            return

        ts = time.time()
        H_in, W_in = self._cache.get("_hw", (None, None))

        # Build DPT pyramid dict if present
        dpt = {}
        name_map = self.dpt_taps
        got_all = all(nm in self._cache for nm in name_map.values())
        if got_all:
            dpt = {
                "p2": self._cache[name_map["p2_src"]],
                "p3": self._cache[name_map["p3_src"]],
                "p4": self._cache[name_map["p4_src"]],
                "p5": self._cache[name_map["p5_src"]],
            }

        # Build DINO fmap if present
        dino_fmap = None
        if self.dino_tap in self._cache and H_in is not None and W_in is not None:
            tokens = self._cache[self.dino_tap]  # (B, N, C)
            if tokens.dim() == 3:
                B, N, Cin = tokens.shape
                Htok, Wtok = H_in // self._patch_size, W_in // self._patch_size
                expected = Htok * Wtok
                # trim specials (CLS/reg) from the front -> keep last expected tokens
                if N >= expected and expected > 0:
                    patch_tokens = tokens[:, N - expected:, :]
                    fmap = patch_tokens.transpose(1, 2).reshape(B, Cin, Htok, Wtok)
                    # lazy 1x1 to 256
                    if not self._initialized_proj:
                        self._dino_proj = nn.Conv2d(Cin, 256, kernel_size=1).to(fmap.device)
                        self._initialized_proj = True
                    dino_fmap = self._dino_proj(fmap)

        # Prepare record
        rec = {
            "t": ts,
            "step": self._step,
            "input_hw": [H_in, W_in],
            "dpt": {k: _tensor_stats(v) for k, v in dpt.items()} if dpt else None,
            "dino": _tensor_stats(dino_fmap) if dino_fmap is not None else None,
        }

        # Optionally save small tensor samples (first item only, CPU, half precision)
        save_tensors = os.getenv("TAP_SAVE_TENSORS", "0") == "1"
        if self.save_small or save_tensors:
            outdir = Path("tap_logs") / f"step_{self._step}"
            outdir.mkdir(parents=True, exist_ok=True)
            if dino_fmap is not None:
                torch.save(dino_fmap[:1].cpu().half(), outdir / "dino_fmap.pt")
            for k, v in dpt.items():
                torch.save(v[:1].cpu().half(), outdir / f"{k}.pt")

        self.writer.write(rec)
        # clear cache to avoid holding big tensors
        self._cache.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def attach_vggt_taps(model: nn.Module, logdir="tap_logs", capture_every=1, save_small_tensors=False):
    """
    Call this ONCE after you construct VGGT in main_live_stream.py.
    Then, at the end of each live step, call tapper.maybe_log() to emit a record.
    """
    tapper = VGGTFeatureTapper(model, logdir=logdir, capture_every=capture_every, save_small_tensors=save_small_tensors)
    return tapper