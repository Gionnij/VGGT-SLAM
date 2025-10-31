from collections import OrderedDict
from functools import partial

import torch
from typing import Dict

from models.DeepLabV3Base.deeplabv3.panopticdl.add_pos import AddPosition
from models.DeepLabV3Base.deeplabv3.panopticdl.conv_module import stacked_conv
from models.DeepLabV3Base.deeplabv3.panopticdl.decoder import SinglePanopticDeepLabDecoder, SinglePanopticDeepLabHead
from models.DeepLabV3Base.deeplabv3.u3hs.semantic_heads import get_semantic_head
from models.DeepLabV3Base.deeplabv3.util import apply_nonlinearity

from torch import nn


class SinglePrototypicalDeepLabHead(nn.Module):
    def __init__(
        self,
        decoder_channels,
        head_channels,
        num_classes,
        class_key,
        out_size,
        add_position=False,
    ):
        super(SinglePrototypicalDeepLabHead, self).__init__()
        fuse_conv = partial(
            stacked_conv,
            kernel_size=5,
            num_stack=1,
            padding=2,
            conv_type="depthwise_separable_conv",
        )

        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            ls = [
                fuse_conv(
                    decoder_channels + (2 if add_position == "deep" else 0),
                    head_channels,
                ),
                nn.Conv2d(
                    head_channels + (2 if add_position == True else 0),
                    num_classes[i],
                    1,
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(num_classes[i], out_size),
            ]

            if add_position == "deep":
                ls.insert(0, AddPosition())
            elif add_position == True:
                ls.insert(1, AddPosition())

            classifier[class_key[i]] = nn.Sequential(*ls)
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred


class PrototypicalDeepLabDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_key,
        low_level_channels,
        low_level_key,
        low_level_channels_project,
        decoder_channels,
        atrous_rates,
        stuff_classes,
        thing_classes,
        feature_dim,
        aspp_channels,
        num_classes,
        extra_detection_decoder=True,
        spatial_clustering=False,
        add_position=False,
        predict_sigmas=True,
        sigma_nonlinearity="exp",
        extra_thing_prototypes=True,
        semantic_branch=None,
        rgb_branch: Dict = None,
        per_class_centers=True,
        aspp_args={},
        dm=None,
    ):
        super(PrototypicalDeepLabDecoder, self).__init__()

        self.semantic_branch = semantic_branch
        self.rgb_share_weights = (
            rgb_branch["share_weights"]
            if rgb_branch is not None and "share_weights" in rgb_branch
            else False
        )
        if semantic_branch is not None:
            self.semantic_decoder = SinglePanopticDeepLabDecoder(
                in_channels,
                feature_key,
                low_level_channels,
                low_level_key,
                low_level_channels_project,
                decoder_channels,
                atrous_rates,
                aspp_channels,
                aspp_args=aspp_args,
            )

            kwargs = {
                "decoder_channels": decoder_channels,
                "head_channels": decoder_channels,
                "num_classes": num_classes,
                **semantic_branch["args"],
            }

            self.semantic_head = get_semantic_head(
                tpe=semantic_branch["type"], **kwargs
            )
        else:
            self.semantic_decoder = None
            self.semantic_head = None
        if rgb_branch is not None:
            if not self.rgb_share_weights:
                self.rgb_decoder = SinglePanopticDeepLabDecoder(
                    in_channels,
                    feature_key,
                    low_level_channels,
                    low_level_key,
                    low_level_channels_project,
                    decoder_channels,
                    atrous_rates,
                    aspp_channels,
                    aspp_args=aspp_args,
                )
            self.rgb_head = SinglePanopticDeepLabHead(
                decoder_channels,
                decoder_channels,
                [3],
                ["result"],
            )
        else:
            self.rgb_decoder = None
            self.rgb_head = None
        self.embedding_decoder = SinglePanopticDeepLabDecoder(
            in_channels,
            feature_key,
            low_level_channels,
            low_level_key,
            low_level_channels_project,
            decoder_channels,
            atrous_rates,
            aspp_channels,
            aspp_args=aspp_args,
        )

        self.extra_detection_decoder = extra_detection_decoder
        if extra_detection_decoder:
            self.detection_decoder = SinglePanopticDeepLabDecoder(
                in_channels,
                feature_key,
                low_level_channels,
                low_level_key,
                low_level_channels_project,
                decoder_channels,
                atrous_rates,
                aspp_channels,
                aspp_args=aspp_args,
            )
        else:
            self.detection_decoder = None

        self.feature_dim = feature_dim
        self.stuff_classes = stuff_classes
        self.thing_classes = thing_classes

        self.per_class_centers = per_class_centers

        embedder_channels = decoder_channels
        detector_channels = decoder_channels
        if semantic_branch is not None and semantic_branch["feed_to_embedder"]:
            embedder_channels = embedder_channels + num_classes
        if semantic_branch is not None and semantic_branch["feed_to_detector"]:
            detector_channels = detector_channels + num_classes

        self.detection_head = SinglePanopticDeepLabHead(
            detector_channels,
            decoder_channels,
            [len(thing_classes) if self.per_class_centers else 1],
            ["result"],
        )

        self.spatial_clustering = spatial_clustering
        self.predict_sigmas = predict_sigmas
        self.sigma_nonlinearity = sigma_nonlinearity
        self.num_sigmas = 0 if not predict_sigmas else 2 if spatial_clustering else 1

        if not extra_thing_prototypes and self.predict_sigmas:
            self.embedder = SinglePanopticDeepLabHead(
                embedder_channels,
                decoder_channels,
                [feature_dim + self.num_sigmas],
                ["result"],
                add_position=add_position,
            )
        else:
            self.embedder = SinglePanopticDeepLabHead(
                embedder_channels,
                decoder_channels,
                [feature_dim],
                ["result"],
                add_position=add_position,
            )

        self.extra_thing_prototypes = extra_thing_prototypes
        if self.extra_thing_prototypes:
            self.thing_prototypes = SinglePanopticDeepLabHead(
                embedder_channels,
                decoder_channels,
                [feature_dim + self.num_sigmas],
                ["result"],
                add_position=add_position,
            )

        self.stuff_prototypes = SinglePrototypicalDeepLabHead(
            embedder_channels,
            decoder_channels,
            [decoder_channels],
            ["result"],
            (feature_dim + self.num_sigmas) * len(stuff_classes),
            add_position=add_position,
        )

    def training_step(self, batch, out):
        if self.semantic_head is not None:
            self.semantic_head.training_step(batch, out)

    def set_image_pooling(self, pool_size):
        self.embedding_decoder.set_image_pooling(pool_size)
        self.detection_decoder.set_image_pooling(pool_size)

    def set_dropout_status(self, status):
        self.semantic_decoder.set_dropout_status(status)

    def forward(self, features):
        pred = OrderedDict()

        decode_emb = self.embedding_decoder(features)
        if self.extra_detection_decoder:
            decode_det = self.detection_decoder(features)
        else:
            decode_det = decode_emb

        if (
            self.semantic_decoder is not None
            and self.semantic_head is not None
            and self.semantic_branch is not None
        ):
            decode_semantic = self.semantic_decoder(features)

            semantic = self.semantic_head(decode_semantic)
            forwarded_logits = semantic["logits"]
            semantic_logits = semantic["semantic"]

            pred["semantic"] = semantic_logits
            pred["semantic_additional"] = semantic

            if (
                self.semantic_branch is not None
                and self.semantic_branch["feed_to_embedder"]
            ):
                logits = forwarded_logits
                if self.semantic_branch["detach"]:
                    logits = logits.detach()
                decode_emb = torch.cat((decode_emb, logits), dim=1)

            if (
                self.semantic_branch is not None
                and self.semantic_branch["feed_to_detector"]
            ):
                logits = forwarded_logits
                if self.semantic_branch["detach"]:
                    logits = logits.detach()
                decode_det = torch.cat((decode_det, logits), dim=1)
        if self.rgb_share_weights:
            pred["rgb"] = self.rgb_head(decode_semantic)
        elif self.rgb_decoder is not None:
            rgb_det = self.rgb_decoder(features)
            pred["rgb"] = self.rgb_head(rgb_det)

        pred["detection"] = self.detection_head(decode_det)["result"]
        pred["embedding"] = self.embedder(decode_emb)["result"]
        if self.extra_thing_prototypes:
            pred["thing_prototypes"] = self.thing_prototypes(decode_emb)["result"]
        else:
            pred["thing_prototypes"] = pred["embedding"]
            if self.predict_sigmas:
                pred["embedding"] = pred["embedding"][:, : -self.num_sigmas, :, :]

        pred["thing_prototypes"] = self.transform_sigmas(pred["thing_prototypes"])
        pred["stuff_prototypes"] = self.stuff_prototypes(decode_emb)["result"].reshape(
            (-1, self.feature_dim + self.num_sigmas, len(self.stuff_classes))
        )
        pred["stuff_prototypes"] = self.transform_sigmas(pred["stuff_prototypes"])

        return pred

    def transform_sigmas(self, prototypes):
        new_sigmas = apply_nonlinearity(
            self.sigma_nonlinearity,
            prototypes[:, -self.num_sigmas :],
        )
        return torch.cat((prototypes[:, : -self.num_sigmas], new_sigmas), dim=1)

    def set_output_stride(self, os):
        self.embedding_decoder.set_output_stride(os)
        if self.semantic_decoder is not None:
            self.semantic_decoder.set_output_stride(os)
        if self.detection_decoder is not None:
            self.detection_decoder.set_output_stride(os)
