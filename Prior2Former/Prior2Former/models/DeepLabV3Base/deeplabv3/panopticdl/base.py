# ------------------------------------------------------------------------------
# Base model for segmentation.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

from torch import nn
from torch.nn import functional as F
import torch


class BaseSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder):
        super(BaseSegmentationModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.run_id = None

    def _init_params(self):
        # Backbone is already initialized (either from pre-trained checkpoint or random init).
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_image_pooling(self, pool_size):
        self.decoder.set_image_pooling(pool_size)

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction.
        Args:
            pred (dict): stores all output of the segmentation model.
            input_shape (tuple): spatial resolution of the desired shape.
        Returns:
            result (OrderedDict): upsampled dictionary.
        """
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(
                pred[key], size=input_shape, mode="bilinear", align_corners=True
            )
            result[key] = out
        return result

    def set_dropout_status(self, status):
        self.decoder.set_dropout_status(status)

    def forward(self, x, targets=None):
        input_shape = x.shape[-2:]

        # contract: features is a dict of tensors
        features = self.backbone(x)
        pred = self.decoder(features)
        results = self._upsample_predictions(pred, input_shape)
        if "res5" in features:
            return {**results, "res5": features["res5"]}
        return results

    def forward_features(self, x):

        # contract: features is a dict of tensors
        features = self.backbone(x)
        flat_features = torch.flatten(features["res5"], 1)
        return flat_features

    def loss(self, results, targets=None):
        raise NotImplementedError

    def get_backbone_params(self):
        return [
            (f"backbone.{name}", param)
            for name, param in self.backbone.named_parameters()
        ]

    def get_backbone_modules(self):
        return [
            (f"backbone.{name}", module)
            for name, module in self.backbone.named_modules()
        ]
