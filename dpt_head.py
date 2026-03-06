# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

        # ---- FiLM controls & modules (built only when enabled) ----
        self.film_mode = os.getenv("VGGT_FUSE_FILM_MODE", "passive").strip().lower()
        if self.film_mode not in {"off", "passive", "active"}:
            self.film_mode = "passive"
        self.film_enabled = os.getenv("VGGT_FUSE_FILM", "0") == "1" and self.film_mode != "off"
        self.film_blend_on = os.getenv("VGGT_FUSE_FILM_BLEND", "1") == "1"
        film_hidden = int(os.getenv("VGGT_FUSE_FILM_HIDDEN", "512"))
        self.film_test = os.getenv("VGGT_FUSE_FILM_TEST", "0") == "1"

        self.film_cond_proj: Optional[nn.Module] = None
        self.film_mlps: Optional[nn.ModuleList] = None
        self.film_gates: Optional[torch.nn.Parameter] = None
        if self.film_enabled:
            self.film_cond_proj = nn.Conv2d(dim_in, 256, kernel_size=1, stride=1, padding=0)
            self.film_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(256, film_hidden),
                        nn.ReLU(inplace=True),
                        nn.Linear(film_hidden, 2 * oc),
                    )
                    for oc in out_channels
                ]
            )
            self.film_gates = nn.Parameter(torch.zeros(len(out_channels)))
            # Identity initialization for gamma/beta and gate bias.
            with torch.no_grad():
                for mlp, oc in zip(self.film_mlps, out_channels):
                    final = mlp[-1]
                    final.weight.zero_()
                    final.bias.zero_()
                    final.bias[:oc].fill_(1.0)  # gamma=1, beta=0

                alpha0 = float(os.getenv("VGGT_FUSE_FILM_ALPHA", "0.0"))

                def logit(p: float) -> float:
                    eps = 1e-6
                    p = min(max(p, eps), 1 - eps)
                    return math.log(p / (1.0 - p))

                self.film_gates.fill_(logit(alpha0))

                if self.film_test:
                    for mlp, oc in zip(self.film_mlps, out_channels):
                        final = mlp[-1]
                        final.weight.normal_(mean=0.0, std=0.05)
                        final.bias.normal_(mean=0.0, std=0.05)
                        final.bias[:oc] += 0.3  # bias gamma away from 1.0
                    self.film_gates.normal_(mean=0.0, std=0.2)

        # Hold per-level tensors for downstream consumers.
        self.raw_pyramid: Optional[List[torch.Tensor]] = None
        self.film_side_pyramid: Optional[List[torch.Tensor]] = None

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []
        all_conf = []
        all_raw_pyr: List[List[torch.Tensor]] = []
        all_film_pyr: List[Optional[List[Optional[torch.Tensor]]]] = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

            # _forward_impl updates side-channel pyramids for the current chunk.
            # Accumulate them across chunks so downstream consumers (semantic head)
            # see the full temporal window, not only the last chunk.
            if self.raw_pyramid is not None:
                all_raw_pyr.append(self.raw_pyramid)
            all_film_pyr.append(self.film_side_pyramid)

        if all_raw_pyr:
            levels = len(all_raw_pyr[0])
            self.raw_pyramid = [torch.cat([chunk[l] for chunk in all_raw_pyr], dim=1) for l in range(levels)]
        else:
            self.raw_pyramid = None

        if any(chunk is not None for chunk in all_film_pyr):
            # Keep per-level tensors only when every chunk has that level populated.
            level_count = len(next(chunk for chunk in all_film_pyr if chunk is not None))  # type: ignore[arg-type]
            merged_film: List[Optional[torch.Tensor]] = []
            for level_idx in range(level_count):
                level_parts: List[torch.Tensor] = []
                level_ok = True
                for chunk in all_film_pyr:
                    if chunk is None or chunk[level_idx] is None:
                        level_ok = False
                        break
                    level_parts.append(chunk[level_idx])  # type: ignore[arg-type]
                merged_film.append(torch.cat(level_parts, dim=1) if level_ok and level_parts else None)
            self.film_side_pyramid = merged_film if any(x is not None for x in merged_film) else None
        else:
            self.film_side_pyramid = None

        # Concatenate results along the sequence dimension
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out: List[torch.Tensor] = []
        film_side: List[Optional[torch.Tensor]] = []
        cond_vecs: List[Optional[torch.Tensor]] = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)

            cond_vec: Optional[torch.Tensor] = None
            if self.film_enabled:
                cond_tokens = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
                if frames_start_idx is not None and frames_end_idx is not None:
                    cond_tokens = cond_tokens[:, frames_start_idx:frames_end_idx]
                cond_tokens = cond_tokens.contiguous().view(B * S, -1, cond_tokens.shape[-1])
                cond_tokens = self.norm(cond_tokens)
                cond_maps = cond_tokens.permute(0, 2, 1).reshape(B * S, cond_tokens.shape[-1], patch_h, patch_w)
                if self.film_cond_proj is not None:
                    cond_maps = self.film_cond_proj(cond_maps)
                    cond_vec = cond_maps.mean(dim=(-2, -1))

            cond_vecs.append(cond_vec)

            if self.film_enabled and self.film_mode == "passive" and cond_vec is not None:
                x_copy = x.clone()
                gb = self.film_mlps[dpt_idx](cond_vec)
                oc = x_copy.shape[1]
                gamma, beta = gb[:, :oc], gb[:, oc:]
                gamma = gamma.view(B * S, oc, 1, 1)
                beta = beta.view(B * S, oc, 1, 1)
                if self.film_blend_on:
                    alpha = torch.sigmoid(self.film_gates[dpt_idx])
                    x_copy = (1.0 - alpha) * x_copy + alpha * (gamma * x_copy + beta)
                else:
                    x_copy = gamma * x_copy + beta
                if self.film_test:
                    x_copy = 1.3 * x_copy + 0.05 * torch.randn_like(x_copy)
                x_copy = x_copy.contiguous()
                film_side.append(x_copy.view(B, S, *x_copy.shape[1:]))
            else:
                film_side.append(None)

            dpt_idx += 1

        self.raw_pyramid = [lvl.view(B, S, *lvl.shape[1:]).detach().clone() for lvl in out]
        self.film_side_pyramid = (
            [fs for fs in film_side] if any(fs is not None for fs in film_side) else None
        )

        if self.film_enabled and self.film_mode == "active":
            for idx, cond_vec in enumerate(cond_vecs):
                if cond_vec is None:
                    continue
                gb = self.film_mlps[idx](cond_vec)
                oc = out[idx].shape[1]
                gamma, beta = gb[:, :oc], gb[:, oc:]
                gamma = gamma.view(B * S, oc, 1, 1)
                beta = beta.view(B * S, oc, 1, 1)
                if self.film_blend_on:
                    alpha = torch.sigmoid(self.film_gates[idx])
                    out[idx] = (1.0 - alpha) * out[idx] + alpha * (gamma * out[idx] + beta)
                else:
                    out[idx] = gamma * out[idx] + beta
                if self.film_test:
                    out[idx] = 1.3 * out[idx] + 0.05 * torch.randn_like(out[idx])

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
