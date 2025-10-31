from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DeepLabV3Base.deeplabv3.deeplab import ASPP



class DeepLabHeadV3PlusSoftplus(nn.Module):
    def __init__(
        self,
        in_channels,
        low_level_channels,
        num_classes=21,
        aspp_dilate=[12, 24, 36],
        add_one=False,
    ):
        super(DeepLabHeadV3PlusSoftplus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

        self.num_classes = num_classes
        self.add_one = add_one

        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature["low_level"])
        output_feature = self.aspp(feature["out"])
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        logits = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        positive_logits = F.softplus(logits)
        if self.add_one:
            positive_logits = positive_logits + 1
        logit_sum = positive_logits.sum(dim=1, keepdim=True)

        probs = positive_logits / logit_sum

        log_probs = probs.log()

        return log_probs, probs, positive_logits, logits

    def get_probabilities(
        self, out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return out[1]

    def get_certainties(
        self,
        out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        probs: torch.Tensor,
        x: torch.Tensor,
        module,
    ) -> torch.Tensor:
        return out[1].max(dim=1).values

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
