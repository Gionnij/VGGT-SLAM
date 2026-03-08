# taps_runtime.py — verification-first taps (no FiLM)
import os, json, time, threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- small async logger ----------
class _AsyncWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict):
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

# ---------- helpers ----------
def _tensor_stats(x: torch.Tensor):
    return {
        "shape": list(x.shape),
        "B": int(x.shape[0]) if x.dim() >= 1 else None,
        "C": int(x.shape[1]) if x.dim() >= 2 else None,
        "H": int(x.shape[-2]) if x.dim() >= 4 else None,
        "W": int(x.shape[-1]) if x.dim() >= 4 else None,
        "mean": float(x.detach().mean().item()),
        "std": float(x.detach().std().item()),
        "min": float(x.detach().amin().item()),
        "max": float(x.detach().amax().item()),
        "dtype": str(x.dtype).replace("torch.", ""),
        "device": str(x.device),
    }

class VGGTFeatureTapper:
    """
    Registers forward hooks on VGGT to capture:
      - DPT pyramid: depth_head.scratch.refinenet{1..4}.out_conv  -> p2..p5
      - DINO tokens at aggregator.patch_embed.blocks.{23|21|15}   -> patch tokens only,
        reshaped to fmap (B,C,Htok,Wtok) via patch size.
    Writes JSONL to tap_logs/taps.jsonl with rich meta so we can verify structure.
    """
    def __init__(self, model: nn.Module, logdir="tap_logs", capture_every=1, save_small_tensors=False):
        self.model = model
        self.capture_every = max(1, int(capture_every))
        self.save_small = bool(save_small_tensors)
        self.capture_dpt = str(os.getenv("VGGT_TAPS_CAPTURE_DPT", "1")).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self.writer = _AsyncWriter(Path(logdir) / "taps.jsonl")
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._cache: Dict[str, torch.Tensor] = {}
        self._step = 0
        self._initialized_proj = False
        self._dino_proj: Optional[nn.Module] = None
        self._patch_size = 14  # DINOv2/VGGT default in your runs
        self._dino_proj_payload = None
        self._dino_proj_source = ""
        self._dino_proj_loaded = False

        # Will be filled every step from main_live_stream via set_batch_meta
        self._meta: Dict = {}

        # DPT taps
        self.dpt_taps = {
            "p2_src": "depth_head.scratch.refinenet1.out_conv",  # finest
            "p3_src": "depth_head.scratch.refinenet2.out_conv",
            "p4_src": "depth_head.scratch.refinenet3.out_conv",
            "p5_src": "depth_head.scratch.refinenet4.out_conv",  # coarsest
        }

        # DINO candidates (pick first that exists)
        self.dino_candidates = [
            "aggregator.patch_embed.blocks.23",
            "aggregator.patch_embed.blocks.21",
            "aggregator.patch_embed.blocks.15",
        ]
        self.dino_tap: Optional[str] = None

        # Register hooks
        modmap = {n: m for n, m in self.model.named_modules()}
        # DPT
        if self.capture_dpt:
            for nm in self.dpt_taps.values():
                if nm not in modmap:
                    raise KeyError(f"[VGGTFeatureTapper] Missing DPT tap: {nm}")
                self._handles.append(modmap[nm].register_forward_hook(self._hook_tensor(nm)))
        else:
            print("[TAPS] DPT tap capture disabled (VGGT_TAPS_CAPTURE_DPT=0)")
        # DINO
        for cand in self.dino_candidates:
            if cand in modmap:
                self.dino_tap = cand
                self._handles.append(modmap[cand].register_forward_hook(self._hook_tensor(cand)))
                break
        if self.dino_tap is None:
            raise KeyError("[VGGTFeatureTapper] No DINO block candidate found.")

        # Also hook early patch embed to capture input H,W (optional but helpful)
        pe_name = "aggregator.patch_embed.patch_embed"
        if pe_name in modmap:
            self._handles.append(modmap[pe_name].register_forward_pre_hook(self._hook_input_hw(pe_name)))

        proj_path = (
            os.getenv("VGGT_DINO_PROJ_WEIGHTS", "").strip()
            or os.getenv("VGGT_DINO_PROJ_PATH", "").strip()
        )
        if proj_path:
            try:
                payload = torch.load(proj_path, map_location="cpu")
                if isinstance(payload, dict) and torch.is_tensor(payload.get("weight")):
                    self._dino_proj_payload = payload
                    self._dino_proj_source = proj_path
                    print(f"[DINO proj] queued external projector: {proj_path}")
                else:
                    print(f"[DINO proj][WARN] invalid projector payload in {proj_path}; expected keys weight/bias")
            except Exception as exc:
                print(f"[DINO proj][WARN] failed loading {proj_path}: {exc}")

    def _ensure_dino_proj(self, cin: int, device: torch.device) -> None:
        if self._initialized_proj and self._dino_proj is not None:
            return

        proj = nn.Conv2d(cin, 256, kernel_size=1).to(device)
        loaded = False
        if self._dino_proj_payload is not None:
            try:
                w = self._dino_proj_payload.get("weight")
                b = self._dino_proj_payload.get("bias")
                if not torch.is_tensor(w):
                    raise ValueError("weight missing or non-tensor")
                if w.dim() != 4 or w.shape[0] != 256 or w.shape[2] != 1 or w.shape[3] != 1:
                    raise ValueError(f"unexpected weight shape {tuple(w.shape)}")
                if int(w.shape[1]) != int(cin):
                    raise ValueError(f"in_channels mismatch: projector has {int(w.shape[1])}, live tap has {int(cin)}")
                proj.weight.data.copy_(w.to(device=proj.weight.device, dtype=proj.weight.dtype))
                if b is not None:
                    if not torch.is_tensor(b) or b.dim() != 1 or b.shape[0] != 256:
                        raise ValueError(f"unexpected bias shape {None if b is None else tuple(b.shape)}")
                    proj.bias.data.copy_(b.to(device=proj.bias.device, dtype=proj.bias.dtype))
                loaded = True
            except Exception as exc:
                print(f"[DINO proj][WARN] could not apply external projector ({self._dino_proj_source}): {exc}")

        self._dino_proj = proj
        self._initialized_proj = True
        self._dino_proj_loaded = loaded
        if loaded:
            print(f"[DINO proj] loaded external projector from {self._dino_proj_source}")
        else:
            print("[DINO proj] using random projector initialization")

    # ----- hooks -----
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

    # ----- public: called by main_live_stream once per step -----
    def set_batch_meta(self, **kwargs):
        """
        Example:
        tapper.set_batch_meta(
            step=step_idx,
            t=time.time(),
            input_hw=(H,W),                  # optional, we also try to sniff it
            window_len=window_len,
            frame_ids_dpt=[...],             # len == B_dpt
            frame_ids_dino=[...],            # len == B_dino
            note="any extra info you want logged",
        )
        """
        self._meta = dict(kwargs)

    @torch.no_grad()
    def maybe_log(self):
        """Call this once per outer step; will log every capture_every steps."""
        self._step += 1
        if (self._step % self.capture_every) != 0:
            self._cache.clear()
            return

        ts = time.time()
        sniffed_hw = self._cache.get("_hw", (None, None))
        H_in, W_in = sniffed_hw

        # ---- DPT pyramid tensors ----
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

        # ---- DINO fmap from tokens ----
        dino = {}
        dino_fmap = None
        dino_trimmed = None
        dino_expected = None
        dino_tokens_shape = None
        if self.dino_tap in self._cache:
            tokens = self._cache[self.dino_tap]  # expected (B, N, C)
            if tokens.dim() == 3:
                Btok, N, Cin = tokens.shape
                dino_tokens_shape = [Btok, N, Cin]
                # If we have input H,W, compute Htok,Wtok by patch size; otherwise infer from N
                if H_in and W_in:
                    Htok, Wtok = H_in // self._patch_size, W_in // self._patch_size
                    dino_expected = Htok * Wtok
                else:
                    # try to guess a square-ish grid
                    Htok = Wtok = int(N ** 0.5)
                    dino_expected = Htok * Wtok

                # Trim leading specials if any (keep last expected tokens)
                trimmed = max(0, N - dino_expected) if dino_expected else 0
                dino_trimmed = trimmed
                if dino_expected and N >= dino_expected:
                    patch_tokens = tokens[:, N - dino_expected:, :]  # (B, dino_expected, C)
                    fmap = patch_tokens.transpose(1, 2).reshape(Btok, Cin, Htok, Wtok)
                    # 1x1 to 256 (lazy init)
                    self._ensure_dino_proj(Cin, fmap.device)
                    dino_fmap = self._dino_proj(fmap)
                    dino["p5_like"] = dino_fmap  # name it p5-like since spatial matches DPT p5 in practice

        # ---- record ----
        rec = {
            "t": ts,
            "step": self._step,
            "sniffed_input_hw": [H_in, W_in],
            "film": (lambda: (
                None
                if not hasattr(self.model, "depth_head")
                else {
                    "enabled": bool(getattr(self.model.depth_head, "film_enabled", False)),
                    "mode": getattr(self.model.depth_head, "film_mode", None),
                    "gates": (
                        torch.sigmoid(self.model.depth_head.film_gates).detach().cpu().tolist()
                        if getattr(self.model.depth_head, "film_enabled", False)
                        and hasattr(self.model.depth_head, "film_gates")
                        else None
                    ),
                }
            ))(),
            "meta": self._meta or None,  # includes frame_ids_* if you pass them
            "dpt": {k: _tensor_stats(v) for k, v in dpt.items()} if dpt else None,
            "dino": {
                "tokens_shape": dino_tokens_shape,
                "expected_patch_tokens": dino_expected,
                "trimmed_special_tokens": dino_trimmed,
                "p5_like": _tensor_stats(dino_fmap) if dino_fmap is not None else None,
            } if self.dino_tap in self._cache else None,
        }

        # Quick mismatch flags to make tailing easy
        try:
            Bdpt = dpt["p5"].shape[0] if dpt else None
        except Exception:
            Bdpt = None
        Bdino = dino_fmap.shape[0] if dino_fmap is not None else None
        rec["batch_mismatch"] = {
            "Bdpt_p5": Bdpt,
            "Bdino_p5like": Bdino,
            "equal": (Bdpt == Bdino) if (Bdpt is not None and Bdino is not None) else None,
        }

        # ---- optional: alignment summary between DPT p5 and DINO p5_like ----
        align_summary = None
        try:
            dpt_p5 = dpt.get("p5", None) if dpt else None
            if dpt_p5 is not None and dino_fmap is not None:
                # GAP to vectors
                dpt_vec = dpt_p5.mean(dim=(2, 3))       # [Bdpt, 256]
                dino_vec = dino_fmap.mean(dim=(2, 3))   # [Bdino, 256]
                # normalize for cosine
                dpt_norm = torch.nn.functional.normalize(dpt_vec, dim=1)
                dino_norm = torch.nn.functional.normalize(dino_vec, dim=1)
                # similarity: [Bdpt, Bdino]
                S = dpt_norm @ dino_norm.t()
                # best match per DPT frame
                best = torch.argmax(S, dim=1)          # [Bdpt], indices into Bdino
                best_list = best.detach().cpu().tolist()

                # quick heuristic: are these all-even or all-odd (stride-2 pattern)?
                if len(best_list) >= 2:
                    parity = [i % 2 for i in best_list]
                    if all(p == 0 for p in parity):
                        guess = "even_indices"
                    elif all(p == 1 for p in parity):
                        guess = "odd_indices"
                    else:
                        guess = "mixed"
                else:
                    guess = "unknown"

                align_summary = {
                    "best_dino_idx_per_dpt": best_list,    # e.g., [0,2,4,6,8,10,12,14]
                    "stride2_guess": guess,                # "even_indices" / "odd_indices" / "mixed"
                    "sim_max_per_dpt": S.max(dim=1).values.detach().cpu().tolist(),
                }
        except Exception as _e:
            align_summary = {"error": str(_e)}

        # Attach it to the record:
        rec["alignment"] = align_summary

        self.writer.write(rec)

        # optional: persist tiny samples for offline checks
        save_tensors = os.getenv("TAP_SAVE_TENSORS", "0") == "1"
        if (self.save_small or save_tensors):
            outdir = Path("tap_logs") / f"step_{self._step}"
            outdir.mkdir(parents=True, exist_ok=True)
            if dino_fmap is not None:
                torch.save(dino_fmap[:1].cpu().half(), outdir / "dino_p5_like.pt")
            for k, v in dpt.items():
                torch.save(v[:1].cpu().half(), outdir / f"{k}.pt")

        # clear cache to avoid holding big tensors
        self._cache.clear()
        self._meta = {}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def extract_dino_fmap(self, batch_shape: Tuple[int, int, int, int, int]) -> Optional[torch.Tensor]:
        """
        Return projected DINO fmap as [B,S,256,Htok,Wtok] for the latest forward pass.
        Uses cached tokens from self.dino_tap.
        """
        if self.dino_tap is None or self.dino_tap not in self._cache:
            return None
        tokens = self._cache[self.dino_tap]
        if not isinstance(tokens, torch.Tensor) or tokens.dim() != 3:
            return None

        B_in, S_in, _, H_in, W_in = batch_shape
        Btok, N, Cin = tokens.shape
        Htok, Wtok = max(1, H_in // self._patch_size), max(1, W_in // self._patch_size)
        expected = Htok * Wtok
        if N < expected:
            return None

        patch_tokens = tokens[:, N - expected :, :]  # drop specials if present
        fmap = patch_tokens.transpose(1, 2).reshape(Btok, Cin, Htok, Wtok)

        self._ensure_dino_proj(Cin, fmap.device)
        dino = self._dino_proj(fmap)

        # Common case: flattened (B*S, C, Htok, Wtok)
        if Btok == B_in * S_in:
            return dino.reshape(B_in, S_in, dino.shape[1], dino.shape[2], dino.shape[3])
        # Single-batch case where token batch equals temporal length.
        if B_in == 1:
            return dino.unsqueeze(0)
        return None

def attach_vggt_taps(model: nn.Module, logdir="tap_logs", capture_every=1, save_small_tensors=False):
    """
    Call this ONCE after you construct VGGT in main_live_stream.py.
    Then, once per step:
        tapper.set_batch_meta(...); model(...); tapper.maybe_log()
    """
    tapper = VGGTFeatureTapper(model, logdir=logdir, capture_every=capture_every, save_small_tensors=save_small_tensors)
    return tapper
