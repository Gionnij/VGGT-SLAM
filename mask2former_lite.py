# mask2former_lite.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mask2FormerLite(nn.Module):
    """
    Tiny query-based segmentation head, Mask2Former-like.
    """
    def __init__(self, num_classes=21, hidden_dim=256, num_queries=100):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.cls_head = nn.Linear(hidden_dim, num_classes)

        # pixel decoder: just take p2 features
        self.proj = nn.Conv2d(256, hidden_dim, 1)

    def forward(self, feats: dict):
        p2 = self.proj(feats["p2"])  # (B, C, H, W)
        B, C, H, W = p2.shape

        # flatten for dot-product masks
        pixel_features = p2.view(B, C, H*W)          # (B,C,HW)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B,Q,C)

        cls_logits = self.cls_head(queries)          # (B,Q,K)
        masks = torch.einsum("bqc,bch->bqh", queries, pixel_features)  # (B,Q,HW)
        masks = masks.view(B, -1, H, W)

        return cls_logits, masks
