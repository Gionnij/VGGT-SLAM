# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# from detectron2.config import configurable
from models.mask2former.detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from ..class_embedding import get_class_embedding
import os

# from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
VIS = os.getenv("vis_attn", "false").lower() == "true"


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, attn_weights = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt, attn_weights

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, attn_weights = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        return tgt, attn_weights

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class GlobalCrossAttentionLayer(nn.Module):

    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.multihead_attn_foreground = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )
        self.multihead_attn_background = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.fusion_layer = nn.Conv2d(
            in_channels=200, out_channels=100, kernel_size=1, stride=1, padding=0
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask_foreground: Optional[Tensor] = None,
        memory_mask_background: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # print(self.with_pos_embed(tgt, query_pos).shape, self.with_pos_embed(memory, pos).shape, memory.shape, memory_mask.shape)
        tgt_foreground = self.multihead_attn_foreground(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask_foreground,
            key_padding_mask=memory_key_padding_mask,
        )[0]

        tgt_background = self.multihead_attn_background(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask_background,
            key_padding_mask=memory_key_padding_mask,
        )[0]

        tgt2 = tgt_background + tgt_foreground  # (Model_v1)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, None

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt, None

    def forward(
        self,
        tgt,
        memory,
        memory_mask_foreground: Optional[Tensor] = None,
        memory_mask_background: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask_foreground,
                memory_mask_background,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            memory_mask_foreground,
            memory_mask_background,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt, attn_weights

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout(tgt2)
        return tgt, attn_weights

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):

    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN) including Dropout"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout(self.dropout_rate) for _ in range(num_layers - 1)
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = self.dropouts[i](x)  # Apply dropout before the layer
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class MultiScaleMaskedTransformerDecoder_GMA(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        mask_embed_type: str = "binary",
        class_embed: dict = {"name": "linear", "args": {}},
        dropout_mlp: float = 0.0,
        **kwargs,
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
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                GlobalCrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.mask_embed_type = mask_embed_type
        # output FFNs
        if self.mask_classification:
            class_embed["args"]["hidden_dim"] = hidden_dim
            class_embed["args"]["num_classes"] = num_classes + 1
            self.class_embed = get_class_embedding(
                class_embed["name"], class_embed["args"]
            )
        if mask_embed_type == "binary":
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        elif mask_embed_type == "beta":
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim * 2, 3)
        else:
            raise NotImplementedError(
                f"Mask embedding of type: >> {mask_embed_type} << not implemented."
            )
        # Wrap the forward_prediction_heads in a dummy module for hooking
        self._forward_prediction_heads_wrapper = nn.Module()
        self._forward_prediction_heads_wrapper.forward = self.forward_prediction_heads

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
        ret["dim_feedforward"] = cfg["args"]["DIM_FEEDFORWARD"]

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg["args"]["DEC_LAYERS"] >= 1
        ret["dec_layers"] = cfg["args"]["DEC_LAYERS"] - 1
        ret["pre_norm"] = cfg["args"]["PRE_NORM"]
        ret["enforce_input_project"] = cfg["args"]["ENFORCE_INPUT_PROJ"]

        ret["mask_dim"] = cfg["SEM_SEG_HEAD"]["MASK_DIM"]

        ret["class_embed"] = (
            cfg["args"]["class_embed"]
            if "class_embed" in cfg["args"]
            else {"name": "linear", "args": {}}
        )
        if "mask_embed_type" in cfg["args"]:
            ret["mask_embed_type"] = cfg["args"]["mask_embed_type"]
        ret["dropout_mlp"] = (
            cfg["args"]["dropout_mlp"] if "dropout_mlp" in cfg["args"] else 0.0
        )

        return ret

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask_foreground, attn_mask_background = (
            self._forward_prediction_heads_wrapper(
                output, mask_features, attn_mask_target_size=size_list[0]
            )
        )

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        for i in range(self.num_layers):

            level_index = i % self.num_feature_levels
            attn_mask_foreground[
                torch.where(
                    attn_mask_foreground.sum(-1) == attn_mask_foreground.shape[-1]
                )
            ] = False
            attn_mask_background[
                torch.where(
                    attn_mask_background.sum(-1) == attn_mask_background.shape[-1]
                )
            ] = False

            # attention: cross-attention first
            output, _ = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask_foreground=attn_mask_foreground,
                memory_mask_background=attn_mask_background,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output, _ = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask_foreground, attn_mask_background = (
                self._forward_prediction_heads_wrapper(
                    output,
                    mask_features,
                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                )
            )

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1
        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
            ),
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)

        if self.mask_embed_type == "binary":
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(
                outputs_mask,
                size=attn_mask_target_size,
                mode="bilinear",
                align_corners=False,
            )
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask_foreground = (
                attn_mask.sigmoid()
                .flatten(2)
                .unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
                < 0.5
            ).bool()
            attn_mask_foreground = attn_mask_foreground.detach()

            attn_mask_background = (
                attn_mask.sigmoid()
                .flatten(2)
                .unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
                > 0.5
            ).bool()
            attn_mask_background = attn_mask_background.detach()

        elif self.mask_embed_type == "beta":
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            outputs_mask_alpha = torch.einsum(
                "bqc,bchw->bqhw",
                mask_embed[:, :, : mask_features.shape[1]],
                mask_features,
            )  # (b,100 ,256,512)
            outputs_mask_beta = torch.einsum(
                "bqc,bchw->bqhw",
                mask_embed[:, :, mask_features.shape[1] :],
                mask_features,
            )  # (b,100 ,256,512)
            outputs_mask_alpha = F.softplus(outputs_mask_alpha) + 1
            outputs_mask_beta = F.softplus(outputs_mask_beta) + 1
            attn_mask = F.interpolate(
                outputs_mask_alpha
                / (outputs_mask_alpha + outputs_mask_beta),  # mean of the beta
                size=attn_mask_target_size,
                mode="bilinear",
                align_corners=False,
            )
            STORE_NP = os.getenv("STORE_NP", None)
            if STORE_NP:
                path = STORE_NP
                os.makedirs(path, exist_ok=True)
                num_files = len(os.listdir(path))
                np.save(
                    f"{path}/{num_files}_attn_mask.npy",
                    attn_mask.detach().mean(1).squeeze().cpu().numpy(),
                )
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask_foreground = (
                attn_mask.flatten(2)
                .unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
                < 0.5
            ).bool()
            attn_mask_foreground = attn_mask_foreground.detach()

            attn_mask_background = (
                attn_mask.flatten(2)
                .unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
                > 0.5
            ).bool()
            attn_mask_background = attn_mask_background.detach()
            outputs_mask = torch.cat(
                (outputs_mask_alpha.unsqueeze(1), outputs_mask_beta.unsqueeze(1)), dim=1
            )

        return outputs_class, outputs_mask, attn_mask_foreground, attn_mask_background

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


# @TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

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
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        attn_threshold: float = 0.5,
        mask_embed_type: str = "binary",
        class_embed: dict = {"name": "linear", "args": {}},
        dropout_mlp: float = 0.0,
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
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.dropout_mlp = dropout_mlp
        self.attn_threshold = attn_threshold
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        self.mask_embed_type = mask_embed_type
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            class_embed["args"]["hidden_dim"] = hidden_dim
            class_embed["args"]["num_classes"] = num_classes + 1
            self.class_embed = get_class_embedding(
                class_embed["name"], class_embed["args"]
            )
            # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        if mask_embed_type == "binary":
            self.mask_embed = MLP(
                hidden_dim, hidden_dim, mask_dim, 3, dropout_rate=self.dropout_mlp
            )
        elif mask_embed_type == "beta":
            self.mask_embed = MLP(
                hidden_dim, hidden_dim, mask_dim * 2, 3, dropout_rate=self.dropout_mlp
            )
        else:
            raise NotImplementedError(
                f"Mask embedding of type: >> {mask_embed_type} << not implemented."
            )
        # Wrap the forward_prediction_heads in a dummy module for hooking
        self._forward_prediction_heads_wrapper = nn.Module()
        self._forward_prediction_heads_wrapper.forward = self.forward_prediction_heads

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
        ret["dim_feedforward"] = cfg["args"]["DIM_FEEDFORWARD"]

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg["args"]["DEC_LAYERS"] >= 1
        ret["dec_layers"] = cfg["args"]["DEC_LAYERS"] - 1
        ret["pre_norm"] = cfg["args"]["PRE_NORM"]
        ret["enforce_input_project"] = cfg["args"]["ENFORCE_INPUT_PROJ"]

        ret["mask_dim"] = cfg["SEM_SEG_HEAD"]["MASK_DIM"]
        ret["class_embed"] = (
            cfg["args"]["class_embed"]
            if "class_embed" in cfg["args"]
            else {"name": "linear", "args": {}}
        )
        if "mask_embed_type" in cfg["args"]:
            ret["mask_embed_type"] = cfg["args"]["mask_embed_type"]
        ret["dropout_mlp"] = (
            cfg["args"]["dropout_mlp"] if "dropout_mlp" in cfg["args"] else 0.0
        )
        ret["attn_threshold"] = (
            cfg["args"]["attn_threshold"] if "attn_threshold" in cfg["args"] else 0.5
        )
        return ret

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features, returns the class predictions, mask predictions
        outputs_class, outputs_mask, attn_mask = self._forward_prediction_heads_wrapper(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output, _ = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output, _ = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = (
                self._forward_prediction_heads_wrapper(
                    output,
                    mask_features,
                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                )
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
            ),
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # (1,100, 256) mask embedding
        outputs_class = self.class_embed(
            decoder_output
        )  # (1(prob. batchsize = 1),100,20) class probabilities for masks
        mask_embed = self.mask_embed(decoder_output)
        if self.mask_embed_type == "binary":
            outputs_mask = torch.einsum(
                "bqc,bchw->bqhw", mask_embed, mask_features
            )  # (1,100 ,256,512)
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(
                outputs_mask,
                size=attn_mask_target_size,
                mode="bilinear",
                align_corners=False,
            )
            STORE_NP = os.getenv("STORE_NP", None)
            if STORE_NP:
                path = STORE_NP
                os.makedirs(path, exist_ok=True)
                num_files = len(os.listdir(path))
                np.save(
                    f"{path}/{num_files}_attn_mask.npy",
                    attn_mask.detach().sigmoid().mean(1).squeeze().cpu().numpy(),
                )
            if VIS:
                import matplotlib.pyplot as plt

                path = "output_attn/binary"
                os.makedirs(path, exist_ok=True)
                num_files = len(os.listdir(path))
                fig, ax = plt.subplots()
                ax.imshow(attn_mask.sigmoid().mean(1).cpu().squeeze())
                ax.axis("off")
                plt.savefig(
                    f"{path}/{num_files}_mean_attn.npy",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close(fig)
                # mask_vis(
                #     attn_mask.detach().sigmoid(),
                #     path=f"{path}/{num_files}.png",
                # )
                # mask_vis(
                #     attn_mask.detach().sigmoid() > 0.5,
                #     path=f"{path}/{num_files}_thresholded.png",
                # )
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (
                attn_mask.sigmoid()
                .flatten(2)
                .unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
                < self.attn_threshold
            ).bool()
            attn_mask = attn_mask.detach()
        elif self.mask_embed_type == "beta":
            outputs_mask_alpha = torch.einsum(
                "bqc,bchw->bqhw",
                mask_embed[:, :, : mask_features.shape[1]],
                mask_features,
            )  # (b,100 ,256,512)
            outputs_mask_beta = torch.einsum(
                "bqc,bchw->bqhw",
                mask_embed[:, :, mask_features.shape[1] :],
                mask_features,
            )  # (b,100 ,256,512)
            outputs_mask_alpha = F.softplus(outputs_mask_alpha) + 1
            outputs_mask_beta = F.softplus(outputs_mask_beta) + 1
            attn_mask = F.interpolate(
                outputs_mask_alpha
                / (outputs_mask_alpha + outputs_mask_beta),  # mean of the beta
                size=attn_mask_target_size,
                mode="bilinear",
                align_corners=False,
            )
            STORE_NP = os.getenv("STORE_NP", None)
            if STORE_NP:
                path = STORE_NP
                os.makedirs(path, exist_ok=True)
                num_files = len(os.listdir(path))
                np.save(
                    f"{path}/{num_files}_attn_mask.npy",
                    attn_mask.detach().mean(1).squeeze().cpu().numpy(),
                )
            if VIS:
                import matplotlib.pyplot as plt

                path = "output_attn/beta"
                os.makedirs(path, exist_ok=True)
                num_files = len(os.listdir(path))
                fig, ax = plt.subplots()
                ax.imshow(attn_mask.mean(1).cpu().squeeze())
                ax.axis("off")
                plt.savefig(
                    f"{path}/{num_files}_mean_attn.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close(fig)
                # mask_vis(
                #     attn_mask.detach().sigmoid(),
                #     path=f"{path}/{num_files}.png",
                # )
                # mask_vis(
                #     attn_mask.detach() > 0.5,
                #     path=f"{path}/{num_files}_thresholded.png",
                # )

            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (
                attn_mask.flatten(2)
                .unsqueeze(1)
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
                < self.attn_threshold
            ).bool()
            attn_mask = attn_mask.detach()
            outputs_mask = torch.cat(
                (outputs_mask_alpha.unsqueeze(1), outputs_mask_beta.unsqueeze(1)), dim=1
            )
        else:
            assert False, "checked above that it should not enter here"

        return outputs_class, outputs_mask, attn_mask

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


class MultiScaleMaskedTransformerDecoder_PN2(MultiScaleMaskedTransformerDecoder):
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        attn_threshold: float = 0.5,
        mask_embed_type: str = "binary",
        class_embed: dict = {"name": "linear", "args": {}},
        dropout_mlp: float = 0.0,
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
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        # super(nn.Module, self).__init__()
        nn.Module.__init__(self)

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.dropout_mlp = dropout_mlp
        self.attn_threshold = attn_threshold
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        self.mask_embed_type = mask_embed_type
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            class_embed["args"]["hidden_dim"] = hidden_dim
            class_embed["args"]["num_classes"] = num_classes # We remove the +1 non object class here
            self.class_embed = get_class_embedding(
                class_embed["name"], class_embed["args"]
            )
            # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        if mask_embed_type == "binary":
            self.mask_embed = MLP(
                hidden_dim, hidden_dim, mask_dim, 3, dropout_rate=self.dropout_mlp
            )
        elif mask_embed_type == "beta":
            self.mask_embed = MLP(
                hidden_dim, hidden_dim, mask_dim * 2, 3, dropout_rate=self.dropout_mlp
            )
        else:
            raise NotImplementedError(
                f"Mask embedding of type: >> {mask_embed_type} << not implemented."
            )
        ###
        # Options to handle non maximum supression
        # 1) Just remove the non object class and see what happens
        # 2) Add an entropy regularization on the dpn
        # 3) Add "saliency" query to find the most salient object as binary classification
        # 4) add specific losses
        # Wrap the forward_prediction_heads in a dummy module for hooking
        self._forward_prediction_heads_wrapper = nn.Module()
        self._forward_prediction_heads_wrapper.forward = self.forward_prediction_heads

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features, returns the class predictions, mask predictions
        outputs_class, outputs_mask, attn_mask = self._forward_prediction_heads_wrapper(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output, _ = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output, _ = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = (
                self._forward_prediction_heads_wrapper(
                    output,
                    mask_features,
                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                )
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
            ),
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # (1,100, 256) mask embedding
        outputs_class = self.class_embed(
            decoder_output
        )  # (1(prob. batchsize = 1),100,20) class probabilities for masks
        mask_embed = self.mask_embed(decoder_output)
        if self.mask_embed_type == "binary":
            outputs_mask = torch.einsum(
                "bqc,bchw->bqhw", mask_embed, mask_features
            )  # (1,100 ,256,512)
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(
                outputs_mask,
                size=attn_mask_target_size,
                mode="bilinear",
                align_corners=False,
            )
            STORE_NP = os.getenv("STORE_NP", None)
            if STORE_NP:
                path = STORE_NP
                os.makedirs(path, exist_ok=True)
                num_files = len(os.listdir(path))
                np.save(
                    f"{path}/{num_files}_attn_mask.npy",
                    attn_mask.detach().sigmoid().mean(1).squeeze().cpu().numpy(),
                )
            if VIS:
                import matplotlib.pyplot as plt

                path = "output_attn/binary"
                os.makedirs(path, exist_ok=True)
                num_files = len(os.listdir(path))
                fig, ax = plt.subplots()
                ax.imshow(attn_mask.sigmoid().mean(1).cpu().squeeze())
                ax.axis("off")
                plt.savefig(
                    f"{path}/{num_files}_mean_attn.npy",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close(fig)
                # mask_vis(
                #     attn_mask.detach().sigmoid(),
                #     path=f"{path}/{num_files}.png",
                # )
                # mask_vis(
                #     attn_mask.detach().sigmoid() > 0.5,
                #     path=f"{path}/{num_files}_thresholded.png",
                # )
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (
                    attn_mask.sigmoid()
                    .flatten(2)
                    .unsqueeze(1)
                    .repeat(1, self.num_heads, 1, 1)
                    .flatten(0, 1)
                    < self.attn_threshold
            ).bool()
            attn_mask = attn_mask.detach()
        elif self.mask_embed_type == "beta":
            outputs_mask_alpha = torch.einsum(
                "bqc,bchw->bqhw",
                mask_embed[:, :, : mask_features.shape[1]],
                mask_features,
            )  # (b,100 ,256,512)
            outputs_mask_beta = torch.einsum(
                "bqc,bchw->bqhw",
                mask_embed[:, :, mask_features.shape[1]:],
                mask_features,
            )  # (b,100 ,256,512)
            outputs_mask_alpha = F.softplus(outputs_mask_alpha) + 1
            outputs_mask_beta = F.softplus(outputs_mask_beta) + 1
            attn_mask = F.interpolate(
                outputs_mask_alpha
                / (outputs_mask_alpha + outputs_mask_beta),  # mean of the beta
                size=attn_mask_target_size,
                mode="bilinear",
                align_corners=False,
            )
            STORE_NP = os.getenv("STORE_NP", None)
            if STORE_NP:
                path = STORE_NP
                os.makedirs(path, exist_ok=True)
                num_files = len(os.listdir(path))
                np.save(
                    f"{path}/{num_files}_attn_mask.npy",
                    attn_mask.detach().mean(1).squeeze().cpu().numpy(),
                )
            if VIS:
                import matplotlib.pyplot as plt

                path = "output_attn/beta"
                os.makedirs(path, exist_ok=True)
                num_files = len(os.listdir(path))
                fig, ax = plt.subplots()
                ax.imshow(attn_mask.mean(1).cpu().squeeze())
                ax.axis("off")
                plt.savefig(
                    f"{path}/{num_files}_mean_attn.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close(fig)
                # mask_vis(
                #     attn_mask.detach().sigmoid(),
                #     path=f"{path}/{num_files}.png",
                # )
                # mask_vis(
                #     attn_mask.detach() > 0.5,
                #     path=f"{path}/{num_files}_thresholded.png",
                # )

            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (
                    attn_mask.flatten(2)
                    .unsqueeze(1)
                    .repeat(1, self.num_heads, 1, 1)
                    .flatten(0, 1)
                    < self.attn_threshold
            ).bool()
            attn_mask = attn_mask.detach()
            outputs_mask = torch.cat(
                (outputs_mask_alpha.unsqueeze(1), outputs_mask_beta.unsqueeze(1)), dim=1
            )
        else:
            assert False, "checked above that it should not enter here"

        return outputs_class, outputs_mask, attn_mask