import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from ..deeplab import ASPP

from torch import tensor


class DeepLabHeadV3PlusEvidential(nn.Module):
    """Implements https://arxiv.org/abs/1806.01768"""

    def __init__(
        self,
        in_channels,
        low_level_channels,
        feature_dim=256,
        num_classes=21,
        aspp_dilate=[12, 24, 36],
        logit_to_evd="relu",
        loss_type="mse",
        loss_top_k=1.0,
        kl_weight=1.0,
        kl_annealing=10,
        uncertainty_mode="dir_strength",
        uncertainty_scale=1,
        softmax_branch=False,
        use_softmax_certainty=False,
        prior=None,
    ):
        super(DeepLabHeadV3PlusEvidential, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.feature_layer = nn.Sequential(
            nn.Conv2d(304, feature_dim, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

        self.softmax_branch = softmax_branch
        self.use_softmax_certainty = use_softmax_certainty
        if self.softmax_branch:
            self.feature_layer2 = nn.Sequential(
                nn.Conv2d(304, feature_dim, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1),
            )
        else:
            self.feature_layer2 = None

        self.num_classes = num_classes

        self.logit_to_evd = logit_to_evd
        self.loss_type = loss_type

        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing

        self.uncertainty_mode = uncertainty_mode

        self.uncertainty_scale = uncertainty_scale

        self.loss_top_k = loss_top_k

        self.prior = prior

        self._init_weight()

    def logit_to_evidence(self, logits: torch.Tensor):
        if self.logit_to_evd == "relu":
            return logits.relu()
        elif self.logit_to_evd == "exp":
            return logits.exp()
        elif self.logit_to_evd == "softplus":
            return torch.nn.Softplus()(logits)
        else:
            raise Exception(f"Logit to evidence function {self.logit_to_evd} not known")

    def forward(self, feature):
        low_level_feature = self.project(feature["low_level"])
        output_feature = self.aspp(feature["out"])
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        logits = self.feature_layer(
            torch.cat([low_level_feature, output_feature], dim=1)
        )

        if self.feature_layer2 is not None:
            logits_softmax = self.feature_layer2(
                torch.cat([low_level_feature, output_feature], dim=1)
            )
        else:
            logits_softmax = None

        evidence = self.logit_to_evidence(logits)
        alpha = evidence + 1

        alpha_sum = alpha.sum(dim=1, keepdim=True)

        if self.uncertainty_mode == "dir_strength":
            uncertainty = self.num_classes / alpha_sum
        elif self.uncertainty_mode == "entropy":
            d = distributions.dirichlet.Dirichlet(
                alpha.permute(0, 2, 3, 1).reshape((-1, alpha.shape[1]))
            )

            uncertainty = torch.exp(
                d.entropy().reshape((alpha.shape[0], 1, alpha.shape[2], alpha.shape[3]))
                / self.uncertainty_scale
            )
        else:
            raise Exception(f"Uncertainty mode {self.uncertainty_mode} not known")
        probability = alpha / alpha_sum

        if self.use_softmax_certainty and logits_softmax is not None:
            softmax_prob = F.softmax(logits_softmax, dim=1)
            max_prob = softmax_prob.max(dim=1).values

            # We want to get the epistemic uncertainty
            # The max_prob is (1-aleatoric uncertainty), while uncertainty until here
            # consists of aleatoric and epistemic uncertainty

            # Hence when max prob high -> aleatoric uncertainty low
            # Then uncertainty is exactly epistemic uncertainty
            # When max prob low -> aleatoric uncertainty high
            # The uncertainty is mostly aleatoric uncertainty and low epistemic uncertainty
            uncertainty = uncertainty * max_prob.unsqueeze(1)

        return {
            "prediction": alpha,
            "uncertainty": uncertainty,
            "probability": probability,
            "logits_softmax": logits_softmax,
        }

    def get_probabilities(self, out: Dict) -> torch.Tensor:
        return out["probability"]

    def get_certainties(
        self,
        out: Dict,
        probs: torch.Tensor,
        x: torch.Tensor,
        module,
    ) -> torch.Tensor:
        return 1 - out["uncertainty"].squeeze(1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def kl_div(self, alpha):
        beta = torch.ones(
            (1, self.num_classes), device=alpha.device, dtype=torch.float32
        )

        S_alpha = alpha.sum(dim=1, keepdim=True)
        S_beta = beta.sum(dim=1, keepdim=True)

        lnB = (
            torch.lgamma(S_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            - torch.lgamma(S_beta)
        )

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = ((alpha - 1) * (dg1 - dg0)).sum(dim=1, keepdim=True) + lnB
        return kl

    def loss_f(
        self, y, alpha: torch.Tensor, module, logits_softmax, func=torch.digamma
    ):
        S = alpha.sum(dim=1, keepdim=True)
        evidence = alpha - 1

        yc = y.clone()
        yc[yc == 255] = self.num_classes
        yc = F.one_hot(yc, self.num_classes + 1)
        yc = yc[:, :, :, : self.num_classes]
        yc = yc.permute(0, 3, 1, 2)

        A = (yc * (func(S) - func(alpha))).sum(dim=1, keepdim=True)

        alpha_tilde = alpha * (1 - yc) + yc
        B = self.kl_div(alpha_tilde) * min(
            1, module.trainer.current_epoch / self.kl_annealing
        )

        if logits_softmax is not None:
            ce = F.cross_entropy(logits_softmax, y, ignore_index=255)
        else:
            ce = 0.0

        return (A + self.kl_weight * B).mean() + ce

    def loss(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        out: Dict,
        module,
    ):
        if self.loss_type == "mse":
            alpha = out["prediction"]
            S = alpha.sum(dim=1, keepdim=True)
            evidence = alpha - 1

            m = alpha / S

            yc = y.clone()
            yc[yc == 255] = self.num_classes
            yc = F.one_hot(yc, self.num_classes + 1)
            yc = yc[:, :, :, : self.num_classes]
            yc = yc.permute(0, 3, 1, 2)

            A = ((yc - m) ** 2).sum(dim=1, keepdim=True)
            B = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1, keepdim=True)

            if self.kl_weight > 0:
                alpha_tilde = alpha * (1 - yc) + yc
                C = self.kl_div(alpha_tilde) * min(
                    1, module.trainer.current_epoch / self.kl_annealing
                )

                return (A + B + self.kl_weight * C).mean()
            else:
                return (A + B).mean()
        elif self.loss_type == "ece":
            alpha = out["prediction"]
            logits_softmax = out["logits_softmax"]
            return self.loss_f(y, alpha, module, logits_softmax)
        elif self.loss_type == "nll":
            alpha = out["prediction"]
            logits_softmax = out["logits_softmax"]
            return self.loss_f(y, alpha, module, logits_softmax, func=torch.log)
        elif self.loss_type == "ce":
            alpha = out["prediction"]
            dist = distributions.Dirichlet(
                alpha.permute((0, 2, 3, 1)).reshape((-1, alpha.shape[1]))
            )
            cat_dist = dist.rsample().reshape(
                (alpha.shape[0], alpha.shape[2], alpha.shape[3], alpha.shape[1])
            )[y != 255, :]
            # We could also optimize pretty much every other segmentation loss here!
            loss = F.nll_loss(torch.log(cat_dist), y[y != 255], reduction="none")

            if self.loss_top_k < 1:
                loss, _ = torch.topk(loss, int(loss.shape[0] * self.loss_top_k))

            return loss.mean()
        elif self.loss_type == "dice":
            alpha = out["prediction"]
            dist = distributions.Dirichlet(
                alpha.permute((0, 2, 3, 1)).reshape((-1, alpha.shape[1]))
            )
            cat_dist = dist.rsample().reshape(
                (alpha.shape[0], alpha.shape[2], alpha.shape[3], alpha.shape[1])
            )[y != 255, :]

            y = y[y != 255].reshape((-1))
            oh = F.one_hot(y, cat_dist.shape[1])

            intersection = (oh * cat_dist).sum(dim=0)
            union = (oh + cat_dist).sum(dim=0)

            eps = 0.0001
            dice = (2 * intersection + eps) / (union + eps)

            loss = (1 - dice).mean()
            return loss
        elif self.loss_type == "variational":
            # Based on https://openreview.net/pdf?id=ByxmXnA9FQ
            alpha = out["prediction"]

            mask = y != 255

            y = y[mask]
            alpha = alpha.permute(1, 0, 2, 3)[:, mask]

            S = alpha.sum(dim=0)
            alpha_y = alpha.gather(dim=0, index=y.unsqueeze(0)).squeeze(0)

            prior = torch.ones_like(alpha)
            if self.prior == "gt":
                prior = prior.scatter(dim=0, index=y.unsqueeze(0), src=alpha.detach())
            else:
                raise Exception(f"Prior {self.prior} not known")

            A = (
                torch.digamma(alpha_y)
                - torch.digamma(S)
                + log_beta(alpha)
                - log_beta(prior)
            )
            B = (
                (alpha - prior) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(0)))
            ).sum(dim=0)

            loss = -(A - B).mean()
            return loss
        else:
            raise Exception(f"Loss type {self.loss_type} is unknown")


def log_beta(alpha: torch.Tensor):
    return alpha.lgamma().sum(dim=0) - torch.lgamma(alpha.sum(dim=0))
