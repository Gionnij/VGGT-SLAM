import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from depth_pyramid import DepthDPTBackbone
from normalize_pyramid import normalize_to_canonical_pyramid
from mask2former_lite import Mask2FormerLite

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, C, H, W = 1, 3, 448, 448     # keep H,W divisible by 14
NUM_CLASSES = 21
NUM_QUERIES = 100

# ---- Helper: tap a DINO token map and reshape to 2D ----
class DinoTap(nn.Module):
    """
    Taps a DINO/DINOv2 transformer block output from VGGT aggregator,
    reshapes tokens -> (B, C, Htok, Wtok), projects to 256 ch.
    """
    def __init__(self, vggt, tap_candidates=None, out_ch=256):
        super().__init__()
        self.vggt = vggt
        self.out_ch = out_ch
        self._cache = {}
        self._handles = []

        # Reasonable default candidates (adjust if your names differ)
        self.tap_candidates = tap_candidates or [
            "aggregator.patch_embed.blocks.23",  # late
            "aggregator.patch_embed.blocks.15",  # mid
            "aggregator.patch_embed.blocks.7",   # early-mid
        ]
        modmap = {n: m for n, m in self.vggt.named_modules()}

        self.tap_name = None
        for nm in self.tap_candidates:
            if nm in modmap:
                self.tap_name = nm
                self._handles.append(modmap[nm].register_forward_hook(self._hook(nm)))
                break
        if self.tap_name is None:
            raise KeyError(f"None of DINO tap candidates exist: {self.tap_candidates}")

        # 1x1 projection for channel alignment after reshape
        self.proj = None  # lazily created when we know C_in

    def _hook(self, name):
        def fn(m, inp, out):
            # Expect (B, N, C)
            if torch.is_tensor(out):
                self._cache[name] = out
            elif isinstance(out, (list, tuple)):
                for x in out:
                    if torch.is_tensor(x): self._cache[name] = x; break
        return fn

    def forward(self, images):
        self._cache.clear()
        # This call runs VGGT and fills the hook cache
        _ = self.vggt(images)
    
        tokens = self._cache[self.tap_name]   # (B, N, C_in), includes CLS/REG tokens
    
        B, N, Cin = tokens.shape
        Htok = images.shape[-2] // 14
        Wtok = images.shape[-1] // 14
        expected = Htok * Wtok
        assert expected > 0 and (images.shape[-2] % 14 == 0) and (images.shape[-1] % 14 == 0), \
            "H and W must be multiples of 14."

        # >>> Trim off special tokens at the front (CLS + reg tokens) <<<
        if N < expected:
            raise RuntimeError(f"Not enough tokens: N={N}, expected={expected}")
        patch_tokens = tokens[:, N - expected : N, :]   # take the last Htok*Wtok tokens
    
        fmap = patch_tokens.transpose(1, 2).reshape(B, Cin, Htok, Wtok)  # (B,Cin,Htok,Wtok)
    
        if self.proj is None:
            self.proj = nn.Conv2d(Cin, self.out_ch, kernel_size=1).to(fmap.device)
    
        return {"dino": self.proj(fmap)}
    
    def remove(self):
        for h in self._handles: h.remove()
        self._handles.clear()

# ---- Simple fusion: merge DINO fmap into p5 (coarsest) and concatenate to p2 path ----
class DinoDepthFusion(nn.Module):
    """
    Fuses DINO fmap with the depth-DPT pyramid. Simple baseline:
      - Upsample DINO fmap to p5 size and add to p5 (both 256ch).
      - Also upsample DINO fmap to p2 size and concatenate into p2 (beef up local detail).
    Returns a dict with the same keys 'p2'..'p5' (all 256ch except p2->512ch after concat).
    """
    def __init__(self):
        super().__init__()
        self.mix_p5 = nn.Identity()   # kept simple; could be a Conv2d(256,256,1)
        self.mix_p2 = nn.Conv2d(256+256, 256, kernel_size=1)  # compress back to 256

    def forward(self, p_feats: dict, dino_fmap: torch.Tensor):
        # Align DINO to p5 and fuse by addition
        p5 = p_feats["p5"]
        d5 = F.interpolate(dino_fmap, size=p5.shape[-2:], mode="bilinear", align_corners=False)
        p5_fused = self.mix_p5(p5 + d5)

        # Align DINO to p2 and fuse by concat+1x1
        p2 = p_feats["p2"]
        d2 = F.interpolate(dino_fmap, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p2_fused = self.mix_p2(torch.cat([p2, d2], dim=1))

        # Pass-through for p3,p4
        return {"p2": p2_fused, "p3": p_feats["p3"], "p4": p_feats["p4"], "p5": p5_fused}

# ---- Main plumbing test ----
if __name__ == "__main__":
    torch.manual_seed(0)
    vggt = VGGT().to(DEVICE).eval()

    # Freeze VGGT for this plumbing test
    for p in vggt.parameters(): p.requires_grad_(False)

    # Build adapters
    dino_tapper = DinoTap(vggt).to(DEVICE)
    depth_backbone = DepthDPTBackbone(vggt).to(DEVICE).eval()
    head = Mask2FormerLite(num_classes=NUM_CLASSES, hidden_dim=256, num_queries=NUM_QUERIES).to(DEVICE).train()
    fusion = DinoDepthFusion().to(DEVICE)

    # Dummy image
    images = torch.randn(B, C, H, W, device=DEVICE)

    # 1) Run DINO tap (this calls vggt forward internally)
    with torch.no_grad():
        dino_feats = dino_tapper(images)   # {"dino": (B,256,H/14,W/14)}

    # 2) Get depth DPT pyramid WITHOUT re-running vggt if possible
    # For the test we re-run; in your real pipeline, refactor to share one forward.
    with torch.no_grad():
        res_feats = depth_backbone(images) # res2..res5
        p_feats = normalize_to_canonical_pyramid(res_feats, H, W)  # p2..p5

    # 3) Fuse DINO + depth
    fused = fusion(p_feats, dino_feats["dino"])

    # 4) Forward through head (Prior2Former-like)
    pred_logits, pred_masks = head(fused)
    print("logits:", tuple(pred_logits.shape), "masks:", tuple(pred_masks.shape))

    # 5) Dummy loss & backward to confirm gradients reach the head
    loss = pred_logits.mean() + pred_masks.mean()
    loss.backward()

    # Check some gradients exist in the head (VGGT remains frozen)
    grad_ok = any((p.grad is not None) and torch.isfinite(p.grad).all() for p in head.parameters())
    print("gradients_in_head:", grad_ok)

    # Clean up hooks
    dino_tapper.remove()
    depth_backbone.remove_hooks()
