# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

import torch
from ..decoder import (
    SinglePanopticDeepLabDecoder,
    SinglePanopticDeepLabHead,
)
from torch import nn
from torch.nn import functional as F


class DiscriminativeDeepLabDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_key,
        low_level_channels,
        low_level_key,
        low_level_channels_project,
        decoder_channels,
        atrous_rates,
        num_classes,
        concat_semantic_to_instance=False,
        **kwargs
    ):
        super(DiscriminativeDeepLabDecoder, self).__init__()
        # Build semantic decoder
        self.semantic_decoder = SinglePanopticDeepLabDecoder(
            in_channels,
            feature_key,
            low_level_channels,
            low_level_key,
            low_level_channels_project,
            decoder_channels,
            atrous_rates,
        )
        self.semantic_head = SinglePanopticDeepLabHead(
            decoder_channels, decoder_channels, [num_classes], ["semantic"]
        )
        # Build instance decoder
        self.instance_decoder = None
        self.instance_head = None
        if kwargs.get("has_instance", False):
            self.concat_semantic_to_instance = concat_semantic_to_instance
            instance_decoder_kwargs = dict(
                in_channels=in_channels,
                feature_key=feature_key,
                low_level_channels=low_level_channels,
                low_level_key=low_level_key,
                low_level_channels_project=kwargs[
                    "instance_low_level_channels_project"
                ],
                decoder_channels=kwargs["instance_decoder_channels"],
                atrous_rates=atrous_rates,
                aspp_channels=kwargs["instance_aspp_channels"],
            )
            self.instance_decoder = SinglePanopticDeepLabDecoder(
                **instance_decoder_kwargs
            )
            instance_head_kwargs = dict(
                decoder_channels=kwargs["instance_decoder_channels"]
                + (num_classes if concat_semantic_to_instance else 0),
                head_channels=kwargs["instance_head_channels"],
                num_classes=[kwargs["instance_embedding_dim"]],
                add_position=kwargs["instance_add_position"]
                if "instance_add_position" in kwargs
                else False,
                class_key=["embedding"],
            )
            self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

    def set_image_pooling(self, pool_size):
        self.semantic_decoder.set_image_pooling(pool_size)
        if self.instance_decoder is not None:
            self.instance_decoder.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()

        # Semantic branch
        semantic = self.semantic_decoder(features)
        semantic = self.semantic_head(semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]

        # Instance branch
        if self.instance_decoder is not None:
            instance = self.instance_decoder(features)
            if self.concat_semantic_to_instance:
                instance = torch.cat((instance, semantic["semantic"]), dim=1)
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]

        return pred
