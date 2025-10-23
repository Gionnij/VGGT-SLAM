import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class TinyPixelDecoder(nn.Module):
    def __init__(self, in_chs=None, out_ch=256):
        super().__init__()
        in_chs = in_chs or [256, 512, 1024, 1024]
        self.proj = nn.ModuleList([nn.Conv2d(c, out_ch, kernel_size=1) for c in in_chs])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * len(in_chs), out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pyramid: list of 4 tensors shaped (B, S, C, H, W) corresponding to p2..p5.
        Returns:
            Fused feature map shaped (B, S, 256, H, W).
        """
        assert len(pyramid) == 4, "Expected 4 pyramid levels (p2..p5)."
        B, S = pyramid[0].shape[:2]
        target_hw = pyramid[0].shape[-2:]
        fused = []
        for level, feat in enumerate(pyramid):
            feat = feat.reshape(B * S, *feat.shape[2:])  # (B*S, C, H, W)
            proj = self.proj[level](feat)                # (B*S, out_ch, H, W)
            if proj.shape[-2:] != target_hw:
                proj = F.interpolate(proj, size=target_hw, mode="bilinear", align_corners=True)
            fused.append(proj)
        x = torch.cat(fused, dim=1)                      # (B*S, out_ch*4, H, W)
        x = self.fuse(x)                                 # (B*S, out_ch, H, W)
        return x.view(B, S, *x.shape[1:])                # (B, S, out_ch, H, W)


class TinyTransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, num_queries=50, num_classes=20):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for "no-object"
        self.mask_embed = nn.Linear(d_model, d_model)
        self.mask_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, pixel_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_features: (B, S, 256, H, W)
        Returns:
            cls_logits: (B, S, Q, num_classes+1)
            mask_logits: (B, S, Q, H, W)
        """
        B, S, C, H, W = pixel_features.shape
        mem = pixel_features.view(B * S, C, H, W)
        mem_proj = self.mask_proj(mem)                       # (B*S, C, H, W)
        mem_seq = mem_proj.flatten(2).transpose(1, 2)        # (B*S, HW, C)

        queries = self.query.unsqueeze(0).expand(B * S, -1, -1)  # (B*S, Q, C)
        decoded = self.decoder(queries, mem_seq)                 # (B*S, Q, C)

        cls_logits = self.class_head(decoded).view(B, S, self.num_queries, -1)

        mask_embed = self.mask_embed(decoded)                    # (B*S, Q, C)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mem_proj)
        masks = masks.view(B, S, self.num_queries, H, W)
        return cls_logits, masks


class SemanticHead(nn.Module):
    def __init__(self, in_chs=None, num_classes=20, num_queries=50):
        super().__init__()
        in_chs = in_chs or [256, 512, 1024, 1024]
        self.pixel = TinyPixelDecoder(in_chs=in_chs, out_ch=256)
        self.decoder = TinyTransformerDecoder(
            d_model=256,
            nhead=8,
            num_layers=2,
            num_queries=num_queries,
            num_classes=num_classes,
        )

    @torch.no_grad()
    def forward(self, pyramid: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pixel_feats = self.pixel(pyramid)
        return self.decoder(pixel_feats)
