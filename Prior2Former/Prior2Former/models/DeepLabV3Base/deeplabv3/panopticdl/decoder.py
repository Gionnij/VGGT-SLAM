# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from functools import partial

import torch
from ...deeplabv3.deeplab import ASPP
from .add_pos import AddPosition
from torch import nn
from torch.nn import functional as F

from .conv_module import stacked_conv


__all__ = ["PanopticDeepLabDecoder"]


class SinglePanopticDeepLabDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_key,
        low_level_channels,
        low_level_key,
        low_level_channels_project,
        decoder_channels,
        atrous_rates,
        aspp_channels=None,
        aspp_args={},
    ):
        super(SinglePanopticDeepLabDecoder, self).__init__()
        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp = ASPP(
            in_channels,
            out_channels=aspp_channels,
            atrous_rates=atrous_rates,
            **aspp_args,
        )
        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
        fuse_conv = partial(
            stacked_conv,
            kernel_size=5,
            num_stack=1,
            padding=2,
            conv_type="depthwise_separable_conv",
        )

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(
                        low_level_channels[i],
                        low_level_channels_project[i],
                        1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU(),
                )
            )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_conv(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def set_output_stride(self, os):
        if os == 32:
            rates = [3, 6, 9]
        elif os == 16:
            rates = [6, 12, 18]
        elif os == 8:
            rates = [12, 24, 36]
        elif os == 4:
            rates = [24, 48, 72]
        else:
            raise Exception(f"Output stride of {os} not supported in panoptic deeplab")

        self.aspp.set_rates(rates)

    def set_dropout_status(self, status):
        self.aspp.set_dropout_status(status)

    def forward(self, features):
        x = features[self.feature_key]
        x = self.aspp(x)

        # build decoder
        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode="bilinear", align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)

        return x


class SinglePanopticDeepLabHead(nn.Module):
    def __init__(
        self,
        decoder_channels,
        head_channels,
        num_classes,
        class_key,
        add_position=False,
    ):
        super(SinglePanopticDeepLabHead, self).__init__()
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
            if add_position == "deep":
                classifier[class_key[i]] = nn.Sequential(
                    AddPosition(),
                    fuse_conv(
                        decoder_channels + 2,
                        head_channels,
                    ),
                    nn.Conv2d(head_channels, num_classes[i], 1),
                )
            elif add_position == True:
                classifier[class_key[i]] = nn.Sequential(
                    fuse_conv(
                        decoder_channels,
                        head_channels,
                    ),
                    AddPosition(),
                    nn.Conv2d(head_channels + 2, num_classes[i], 1),
                )
            else:
                classifier[class_key[i]] = nn.Sequential(
                    fuse_conv(
                        decoder_channels,
                        head_channels,
                    ),
                    nn.Conv2d(head_channels, num_classes[i], 1),
                )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred

    def get_uncertainty(self, out, i):
        return 1 - F.softmax(out["semantic"][i].squeeze(), dim=0).max(dim=0).values


class PanopticDeepLabDecoder(nn.Module):
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
        semantic_branch=None,
        rgb_branch=False,
        aspp_args={},
        **kwargs,
    ):
        super(PanopticDeepLabDecoder, self).__init__()
        # Build semantic decoder
        if "rgb_branch" in kwargs:
            rgb_branch = kwargs["rgb_branch"]
            self.rgb_decoder
        self.semantic_decoder = SinglePanopticDeepLabDecoder(
            in_channels,
            feature_key,
            low_level_channels,
            low_level_key,
            low_level_channels_project,
            decoder_channels,
            atrous_rates,
            aspp_args=aspp_args,
        )
        # self.semantic_head = SinglePanopticDeepLabHead(
        #    decoder_channels, decoder_channels, [num_classes], ["semantic"]
        # )
        kwargs2 = {
            "decoder_channels": decoder_channels,
            "head_channels": decoder_channels,
            "num_classes": num_classes,
            **semantic_branch["args"],
        }
        from Prior2Former import (
            get_semantic_head,
        )  # need to import here, else circular import

        self.semantic_head = (
            SinglePanopticDeepLabHead(
                decoder_channels,
                decoder_channels,
                num_classes=[num_classes],
                class_key=["semantic"],
            )
            if semantic_branch["type"] == "SingePanopticDeepLabHead"
            else get_semantic_head(tpe=semantic_branch["type"], **kwargs2)
        )
        if rgb_branch:
            self.rgb_decoder = SinglePanopticDeepLabDecoder(
                in_channels,
                feature_key,
                low_level_channels,
                low_level_key,
                low_level_channels_project,
                decoder_channels,
                atrous_rates,
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
        # Build instance decoder
        self.instance_decoder = None
        self.instance_head = None
        if kwargs.get("has_instance", False):
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
                **instance_decoder_kwargs, aspp_args=aspp_args
            )
            instance_head_kwargs = dict(
                decoder_channels=kwargs["instance_decoder_channels"],
                head_channels=kwargs["instance_head_channels"],
                num_classes=kwargs["instance_num_classes"],
                class_key=kwargs["instance_class_key"],
            )
            self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

    def set_output_stride(self, os):
        self.semantic_decoder.set_output_stride(os)
        if self.instance_decoder is not None:
            self.instance_decoder.set_output_stride(os)

    def set_image_pooling(self, pool_size):
        self.semantic_decoder.set_image_pooling(pool_size)
        if self.instance_decoder is not None:
            self.instance_decoder.set_image_pooling(pool_size)

    def get_uncertainty(self, out, i):
        return 1 - F.softmax(out["semantic"][i].squeeze(), dim=0).max(dim=0).values

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
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]
        if self.rgb_decoder is not None:
            rgb_det = self.rgb_decoder(features)
            pred["rgb"] = self.rgb_head(rgb_det)
        return pred
