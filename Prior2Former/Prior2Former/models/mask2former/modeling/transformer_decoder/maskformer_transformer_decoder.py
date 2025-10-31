# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

# from detectron2.config import configurable
from models.mask2former.detectron2.layers import Conv2d

# from detectron2.utils.registry import Registry

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer
from ..class_embedding import get_class_embedding


# TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
# TRANSFORMER_DECODER_REGISTRY.__doc__ = """
# Registry for transformer module in MaskFormer.
# """


def get_transformer_decoder(name, cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    if name == "MultiScaleMaskedTransformerDecoder":
        from . import MultiScaleMaskedTransformerDecoder

        return MultiScaleMaskedTransformerDecoder(
            **MultiScaleMaskedTransformerDecoder.from_config(
                cfg, in_channels, mask_classification
            )
        )
    elif name=="MultiScaleMaskedTransformerDecoder_PN2":
        from . import MultiScaleMaskedTransformerDecoder_PN2

        return MultiScaleMaskedTransformerDecoder_PN2(
            **MultiScaleMaskedTransformerDecoder_PN2.from_config(
                cfg, in_channels, mask_classification
            )
        )
    elif name == "MultiScaleMaskedTransformerDecoder_GMA":
        from . import MultiScaleMaskedTransformerDecoder_GMA

        return MultiScaleMaskedTransformerDecoder_GMA(
            **MultiScaleMaskedTransformerDecoder_GMA.from_config(
                cfg, in_channels, mask_classification
            )
        )
    elif name == "StandardTransformerDecoder":

        return StandardTransformerDecoder(
            **StandardTransformerDecoder.from_config(
                cfg, in_channels, mask_classification
            )
        )
    else:
        raise NotImplementedError(f"Transformer decoder {name} not implemented")
    # name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return None
    # return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    assert False, "use get_transformer_decoder instead"
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    # return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)
    return None


# @TRANSFORMER_DECODER_REGISTRY.register()
class StandardTransformerDecoder(nn.Module):
    # @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        deep_supervision: bool = True,
        class_embed: dict = {"name": "linear", "args": {}},
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # output FFNs
        if self.mask_classification:
            class_embed["args"]["hidden_dim"] = hidden_dim
            class_embed["args"]["num_classes"] = num_classes + 1
            self.class_embed = get_class_embedding(
                class_embed["name"], class_embed["args"]
            )
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg["SEM_SEG_HEAD"]["NUM_CLASSES"]
        ret["hidden_dim"] = cfg["args"]["HIDDEN_DIM"]
        ret["num_queries"] = cfg["args"]["NUM_OBJECT_QUERIES"]
        # Transformer parameters:
        ret["nheads"] = cfg["args"]["NHEADS"]
        ret["dropout"] = cfg["args"]["DROPOUT"]
        ret["dim_feedforward"] = cfg["args"]["DIM_FEEDFORWARD"]
        ret["enc_layers"] = cfg["args"]["ENC_LAYERS"]
        ret["dec_layers"] = cfg["args"]["DEC_LAYERS"]
        ret["pre_norm"] = cfg["args"]["PRE_NORM"]
        ret["enforce_input_project"] = cfg["args"]["ENFORCE_INPUT_PROJ"]

        ret["mask_dim"] = cfg["SEM_SEG_HEAD"]["MASK_DIM"]

        return ret

    def forward(self, x, mask_features, mask=None):
        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[
                0
            ]
        pos = self.pe_layer(x, mask)

        src = x
        hs, memory = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos
        )

        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            out = {"pred_logits": outputs_class[-1]}
        else:
            out = {}

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum(
                "lbqc,bchw->lbqhw", mask_embed, mask_features
            )
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum(
                "bqc,bchw->bqhw", mask_embed, mask_features
            )
            out["pred_masks"] = outputs_seg_masks
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
