import torch
import torch.nn as nn
from torch.nn import functional as F

from Prior2Former import Dirichlet


class RegularCE(nn.Module):
    """
    Regular cross entropy loss for semantic segmentation, support pixel-wise loss weight.
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, weight=None):
        super(RegularCE, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, **kwargs):
        if "semantic_weights" in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs["semantic_weights"]
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask = labels.contiguous().view(-1) != self.ignore_label

        pixel_losses = pixel_losses[mask]
        return pixel_losses.mean()


class OhemCE(nn.Module):
    """
    Online hard example mining with cross entropy loss, for semantic segmentation.
    This is widely used in PyTorch semantic segmentation frameworks.
    Reference: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/1b3ae72f6025bde4ea404305d502abea3c2f5266/lib/core/criterion.py#L29
    Arguments:
        ignore_label: Integer, label to ignore.
        threshold: Float, threshold for softmax score (of gt class), only predictions with softmax score
            below this threshold will be kept.
        min_kept: Integer, minimum number of pixels to be kept, it is used to adjust the
            threshold value to avoid number of examples being too small.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, threshold=0.7, min_kept=100000, weight=None):
        super(OhemCE, self).__init__()
        self.threshold = threshold
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, **kwargs):
        predictions = F.softmax(logits, dim=1)
        if "semantic_weights" in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs["semantic_weights"]
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask = labels.contiguous().view(-1) != self.ignore_label

        tmp_labels = labels.clone()
        tmp_labels[tmp_labels == self.ignore_label] = 0
        # Get the score for gt class at each pixel location.
        predictions = predictions.gather(1, tmp_labels.unsqueeze(1))
        predictions, indices = (
            predictions.contiguous()
            .view(
                -1,
            )[mask]
            .contiguous()
            .sort()
        )
        min_value = predictions[min(self.min_kept, predictions.numel() - 1)]
        threshold = max(min_value, self.threshold)

        pixel_losses = pixel_losses[mask][indices]
        pixel_losses = pixel_losses[predictions < threshold]
        return pixel_losses.mean()


class DeepLabCE(nn.Module):
    """
    Hard pixel mining mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, top_k_percent=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent = top_k_percent
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, **kwargs):
        if isinstance(labels, dict):
            labels = labels["semantic"]

        if "semantic_weights" in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs["semantic_weights"]
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        if self.top_k_percent == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()


class DirichletPrior(nn.Module):
    def __init__(
        self,
        num_classes,
        ignore_label=-1,
        top_k_percent=1.0,
        loss_type="nll",
        kl_weight=1.0,
        kl_annealing=10,
        mask_loss_at_ignore=False,
        hierarchical=False,
        dm=None,
        entropy_regularization=0,
    ):
        super(DirichletPrior, self).__init__()
        self.top_k_percent = top_k_percent
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.kl_weight = kl_weight
        self.mask_loss_at_ignore = mask_loss_at_ignore
        self.hierarchical = hierarchical
        self.kl_annealing = kl_annealing
        if self.hierarchical:
            if dm is None:
                raise Exception("Need datamanager when doing hierarchical training")
            self.class_categories = dm.class_categories
        self.entropy_regularization = entropy_regularization

    def hierarchical_loss(self, logits, targets, epoch, **kwargs):
        category_logits = logits["category"]

        category_gt = targets["category"]

        category_loss = self.forward(
            category_logits,
            {"semantic": category_gt},
            epoch,
            num_classes=len(self.class_categories),
            **kwargs,
        )

        semantics = targets["semantic"]

        loss = category_loss

        for category in self.class_categories:
            gt = torch.ones_like(semantics) * 255
            mask = category_gt == category["id"]
            gt[mask] = semantics[mask] - category["min_ix"]

            loss += self.forward(
                logits[category["name"]],
                {"semantic": gt},
                epoch,
                num_classes=category["num_classes"],
                **kwargs,
            )
        return loss

    def forward(self, logits, targets, epoch, num_classes=None, **kwargs):
        if isinstance(logits, dict) and self.hierarchical:
            return self.hierarchical_loss(logits, targets, epoch, **kwargs)

        if num_classes is None:
            num_classes = self.num_classes

        y = targets["semantic"]

        evidence = logits
        alpha = evidence + 1
        loss = 0
        if self.entropy_regularization > 0:
            a = alpha.permute(0, 2, 3, 1)
            loss += Dirichlet(a).entropy().mean() * self.entropy_regularization
        if self.loss_type == "mse":
            S = alpha.sum(dim=1, keepdim=True)
            evidence = alpha - 1

            m = alpha / S

            yc = y.clone()
            yc[yc == self.ignore_label] = num_classes
            yc = F.one_hot(yc, num_classes + 1)
            yc = yc[:, :, :, :num_classes]
            yc = yc.permute(0, 3, 1, 2)

            A = ((yc - m) ** 2).sum(dim=1, keepdim=True)
            B = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1, keepdim=True)

            alpha_tilde = evidence * (1 - yc) + 1
            C = self.kl_div(alpha_tilde) * min(1, epoch / self.kl_annealing)

            loss += (A + B + self.kl_weight * C).mean()
        elif self.loss_type == "ece":
            loss += self.loss_f(y, alpha, evidence, epoch, num_classes=num_classes)
        elif self.loss_type == "nll":
            loss += self.loss_f(
                y, alpha, evidence, epoch, func=torch.log, num_classes=num_classes
            )
        else:
            raise Exception(f"Loss type {self.loss_type} is unknown")
        return loss

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
        self,
        y,
        alpha: torch.Tensor,
        evidence: torch.Tensor,
        epoch: int,
        func=torch.digamma,
        num_classes=None,
    ):
        if num_classes is None:
            num_classes = self.num_classes
        S = alpha.sum(dim=1, keepdim=True)

        yc = y.clone()
        yc[yc == self.ignore_label] = num_classes
        yc = F.one_hot(yc, num_classes + 1)
        yc = yc[:, :, :, :num_classes]
        yc = yc.permute(0, 3, 1, 2)

        A = (yc * (func(S) - func(alpha))).sum(dim=1, keepdim=True)

        loss = A

        if self.kl_weight > 0:
            alpha_tilde = alpha * (1 - yc) + yc

            B = self.kl_div(alpha_tilde) * min(1, epoch / self.kl_annealing)

            loss = loss + self.kl_weight * B

        if self.mask_loss_at_ignore:
            loss = loss.squeeze(1)[y != self.ignore_label]

        if self.top_k_percent < 1:
            top_k_pixels = int(self.top_k_percent * loss.numel())
            loss, _ = torch.topk(loss.reshape((-1)), top_k_pixels)

        if loss.shape[0] == 0:
            return 0
        return loss.mean()
