# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta, Dirichlet
from tqdm import tqdm

from Prior2Former import (
    apply_nonlinearity,
)
from .matcher import HungarianMatcher
import matplotlib.pyplot as plt
import seaborn as sns


# from detectron2.utils.comm import get_world_size
from models.mask2former.detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness_pos_select,
    point_sample,
)
from ..utils.misc import nested_tensor_from_tensor_list

VIS = False


def beta_likelihood_loss(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
    kwargs: dict = {},
    cls_label_weights: torch.Tensor = torch.tensor(1),
):
    dist = Beta(alpha, beta)
    loss = -dist.log_prob(target)  # * cls_label_weights
    if "balancing" in kwargs:
        if kwargs["balancing"] == "linear":
            # balance loss according to number of 0 and 1 labels
            o = target.shape[1] * target.shape[0] / ((target > 0.7).sum() * 2 + 1e-5)
            z = target.shape[1] * target.shape[0] / ((target < 0.3).sum() * 2 + 1e-5)
            loss[target > 0.7] *= o
            loss[target < 0.3] *= z
    if kwargs.get("reg_likelihood", 0) > 0:
        loss -= kwargs.get("reg_likelihood", 0) * dist.entropy()
    return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    cls_label_weights: torch.Tensor = torch.ones(1),
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # return (loss).sum() / num_masks
    return (loss).sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cls_label_weights: torch.Tensor = torch.ones(1),
    logits: bool = True,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    if logits:
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    else:
        loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    # if loss_reduction == None:
    return loss
    # return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def calculate_uncertainty_beta(logits, type="evidence"):
    if type == "evidence":
        return 1 / logits
    elif type == "prob":
        return 1 - torch.abs(logits - 0.5)

def calculate_uncertainty_beta_dir(mask_logits, class_logits,type="evidence"):
    """

    :param mask_logits:  bs x evidence x queries x h x w
    :param class_logits: bs x queries x cls
    :param type:
    :return:
    """
    mask_logits


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        eos_coef,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        class_weight,
        mask_weight,
        dice_weight,
        dec_layers,
        deep_supervision,
        dm,
        class_kl_weight=0,
        weighted_kl_div=False,
        weight_dict_method: str = None,
        losses=["labels", "masks"],
        mask_embed_type: str = "binary",
        mask_loss: str = "ce",
        positive_label_sample_ratio: float = 0.0,
        min_positiv_points: float = -1.0,
        u_type: str = "evidence",
        subsample_size: int =-1,
        cls_focal_loss_weight =0,
        gamma=2,
        entropy_reg_weight=0.0,
        mask_kl_divergence_weight=0.0,
        kl_weight_non_mask=0.0,
        kl_weight_nno=0.0,
        kl_nno_weighed=True,
        kl_non_mask_weighed=True,
        cls_func="log",
        binary_weight=0.1,
        binary_loss_mode="sample",
        binary_loss_negative_weight=None,
        **kwargs,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            mask_embed_type: how the mask MLP returns the mask vectors
            mask_loss: which type of loss should be used for the mask loss, next to the dice loss, only relevant if mask_embed_type is >>beta<<
            subsample_size: subsamples the class to distribte the classes more evenly
            binary_weight: weight for the binary classification loss, 
            binary_loss_mode: how to handle the binary loss, either "sample" or "all". "Sample" will sample the same number of positive and negative samples, "all" will use all negative samples.
            binary_loss_negative_weight: if not None, will use this weight for the negative samples in the binary loss
            kwargs:
                reg_likelihood -> can regulize the log likelihood of the maskloss
        """
        super().__init__()
        self.u_type = u_type
        self.num_classes = num_classes
        # self.matcher = matcher
        self.matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=num_points,
            mask_embed_type=mask_embed_type,
            # weight_functions=mask_losses,
        )
        self.interpolation = "bilinear"
        self.padding_mode = "border"
        self.align_corners = True
        # self.weight_dict = weight_dict
        self.weight_dict = {
            "loss_ce": class_weight,
            "loss_dpn": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_binary": binary_weight,
        }
        self.label_weight_dict = dm.get_masks_weight_dict(
            method=weight_dict_method, **kwargs
        )
        # if weighted_kl_div:
        #     if eos_coef > 0:
        #         empty_counts= torch.ones(self.num_classes + 1)
        #     else:
        #         empty_counts = torch.ones(self.num_classes)
        #     label_counts_dict= dm.get_instance_counts()
        #     if isinstance(label_counts_dict, dict):
        #         for k, v in label_counts_dict.items():
        #             empty_counts[k] = v
        #     else:
        #         if eos_coef > 0:
        #             empty_counts[:-1] =label_counts_dict
        #         else:
        #             empty_counts = label_counts_dict
        #     self.label_counts_dict=empty_counts
        if cls_func=="digamma":
            self.cls_func=torch.digamma
        elif cls_func=="log":
            self.cls_func = torch.log
        else:
            self.cls_func = torch.log
        self.cls_focal_loss_weight = cls_focal_loss_weight
        self.subsample_size = subsample_size
        self.weighted_kl_div=weighted_kl_div
        self.mask_kl_divergence_weight=mask_kl_divergence_weight
        self.kl_weight_nno=kl_weight_nno
        self.kl_weight_non_mask = kl_weight_non_mask
        self.label_weight_dict_method = weight_dict_method
        self.kl_weight = class_kl_weight
        self.kl_nno_weighed=kl_nno_weighed
        self.kl_non_mask_weighed=kl_non_mask_weighed
        self.gamma = gamma
        self.entropy_reg_weight=entropy_reg_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )
            self.weight_dict.update(aux_weight_dict)
        self.eos_coef = eos_coef
        self.losses = losses
        if eos_coef >0:
            empty_weight = torch.ones(self.num_classes + 1)
        else:
            empty_weight = torch.ones(self.num_classes)
        if self.label_weight_dict_method is not None :
            if isinstance(self.label_weight_dict, dict):
                for k, v in self.label_weight_dict.items():
                    empty_weight[k] = v
            else:
                if eos_coef > 0:
                    empty_weight[:-1] = self.label_weight_dict
                else:
                    empty_weight = self.label_weight_dict
        if eos_coef > 0:
            empty_weight[-1] = self.eos_coef
        self.empty_weight = empty_weight

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.positive_label_sample_ratio = positive_label_sample_ratio
        self.min_positiv_points = min_positiv_points
        self.binary_loss_mode = binary_loss_mode
        self.binary_loss_negative_weight = binary_loss_negative_weight
        self.mask_embed_type = mask_embed_type
        self.mask_loss = mask_loss
        self.kwargs = kwargs

    def loss_labels_dpn_nno(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        Without non object class
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        src_logits = torch.nn.Softplus()(src_logits)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["labels"], indices)]
        ).to(torch.int64)
        evidence = src_logits[idx]
        if self.subsample_size>0:
            subsample_idx=sample_indices_by_inverse_frequency(target_classes_o,self.subsample_size)
            target_classes_o=target_classes_o[subsample_idx]
            evidence=evidence[subsample_idx]
        y = target_classes_o

        alpha = evidence + 1

        num_classes = self.num_classes

        return self.loss_f_nno(
            y, alpha, evidence, epoch=500, func=torch.log, num_classes=num_classes
        )

    def loss_labels_dpn_focal_wno(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        Without non object class
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        src_logits = torch.nn.Softplus()(src_logits)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["labels"], indices)]
        ).to(torch.int64)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        evidence = src_logits.permute(0, 2, 1)
        if self.subsample_size>0:
            subsample_idx=sample_indices_by_inverse_frequency(target_classes_o,self.subsample_size)
            target_classes=target_classes[subsample_idx]
            evidence=evidence[subsample_idx]
        y = target_classes
        alpha = evidence + 1

        num_classes = self.num_classes

        return self.loss_f_focal_wno(
            y, alpha, evidence, epoch=500, func=torch.log, num_classes=num_classes
        )

    def loss_labels_dpn_focal(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        Without non object class
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        src_logits = torch.nn.Softplus()(src_logits)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["labels"], indices)]
        ).to(torch.int64)

        evidence = src_logits[idx]
        if self.subsample_size>0:
            subsample_idx=sample_indices_by_inverse_frequency(target_classes_o,self.subsample_size)
            target_classes_o=target_classes_o[subsample_idx]
            evidence=evidence[subsample_idx]
        y = target_classes_o
        alpha = evidence + 1

        num_classes = self.num_classes

        return self.loss_f_focal(
            y, alpha, evidence, epoch=500, func=torch.log, num_classes=num_classes
        )

    def loss_labels_dpn_nno_reg(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
               targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
               Without non object class
               """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        src_logits = torch.nn.Softplus()(src_logits)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["labels"], indices)]
        ).to(torch.int64)

        evidence = src_logits[idx]
        mask = torch.ones_like(src_logits[:, :, 0], dtype=torch.bool)
        mask[idx] = False
        evidence_nno = src_logits[mask]
        if self.subsample_size > 0:
            subsample_idx = sample_indices_by_inverse_frequency(target_classes_o, self.subsample_size)
            target_classes_o = target_classes_o[subsample_idx]
            evidence = evidence[subsample_idx]
        y = target_classes_o
        alpha = evidence + 1
        alpha_nno = evidence_nno + 1

        num_classes = self.num_classes

        return self.loss_f_nno_clr_reg(
            y, alpha, alpha_nno, evidence, epoch=500, func=torch.log, num_classes=num_classes
        )

    def loss_labels_dpn_focal_nno_reg(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
               targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
               Without non object class
               """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        src_logits = torch.nn.Softplus()(src_logits)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["labels"], indices)]
        ).to(torch.int64)

        evidence = src_logits[idx]
        mask = torch.ones_like(src_logits[:, :, 0], dtype=torch.bool)
        mask[idx] = False
        evidence_nno = src_logits[mask]
        if self.subsample_size > 0:
            subsample_idx = sample_indices_by_inverse_frequency(target_classes_o, self.subsample_size)
            target_classes_o = target_classes_o[subsample_idx]
            evidence = evidence[subsample_idx]
        y = target_classes_o
        alpha = evidence + 1
        alpha_nno = evidence_nno + 1

        num_classes = self.num_classes

        return self.loss_f_nno_clr_reg(
            y, alpha, alpha_nno, evidence, epoch=500, func=torch.log, num_classes=num_classes,focal=True,
        )

    def loss_labels_dpn(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        src_logits = torch.nn.Softplus()(src_logits)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["labels"], indices)]
        ).to(torch.int64)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        y = target_classes
        evidence = src_logits.permute(0, 2, 1)
        alpha = evidence + 1

        num_classes = self.num_classes

        return self.loss_f(
            y, alpha, evidence, epoch=500, func=torch.log, num_classes=num_classes
        )

    def loss_labels_dpn_binary(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        binary_logit = src_logits[:, :, self.num_classes]
        src_logits = torch.nn.Softplus()(src_logits[:,:,:self.num_classes])
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["labels"], indices)]
        ).to(torch.int64)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        evidence = src_logits[idx]
        mask = torch.ones_like(src_logits[:, :, 0], dtype=torch.bool)
        mask[idx] = False
        false_mask = binary_logit[mask]
        true_mask=binary_logit[idx]
        if self.binary_loss_mode == "sample":
            rand_idx = torch.randperm(binary_logit[mask].shape[0])[:binary_logit[idx].shape[0]]
            false_mask=false_mask[rand_idx]
        elif self.binary_loss_mode == "all":
            "the false mask is all predictions that are not matched already"
            pass
        else:
            raise ValueError(f"Unknown binary loss mode: {self.binary_loss_model}")
        evidence_nno = src_logits[mask]
        if self.subsample_size > 0:
            subsample_idx = sample_indices_by_inverse_frequency(target_classes_o, self.subsample_size)
            target_classes_o = target_classes_o[subsample_idx]
            evidence = evidence[subsample_idx]
        y = target_classes_o
        alpha = evidence + 1
        alpha_nno = evidence_nno + 1

        num_classes = self.num_classes

        return self.loss_f_nno_clr_reg_binary(
            y, alpha,alpha_nno, evidence, true_mask, false_mask, epoch=500, func=torch.log, num_classes=num_classes
        )

    def loss_labels_dpn_focal_binary(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        binary_logit = src_logits[:, :, self.num_classes]
        src_logits = torch.nn.Softplus()(src_logits[:,:,:self.num_classes])

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["labels"], indices)]
        ).to(torch.int64)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        evidence = src_logits[idx]
        mask = torch.ones_like(src_logits[:, :, 0], dtype=torch.bool)
        mask[idx] = False
        false_mask = binary_logit[mask]
        true_mask=binary_logit[idx]
        rand_idx = torch.randperm(binary_logit[mask].shape[0])[:binary_logit[idx].shape[0]]
        false_mask=false_mask[rand_idx]
        evidence_nno = src_logits[mask]
        if self.subsample_size > 0:
            subsample_idx = sample_indices_by_inverse_frequency(target_classes_o, self.subsample_size)
            target_classes_o = target_classes_o[subsample_idx]
            evidence = evidence[subsample_idx]
        y = target_classes_o
        alpha = evidence + 1
        alpha_nno = evidence_nno + 1

        num_classes = self.num_classes

        return self.loss_f_nno_clr_reg_binary(
            y, alpha,alpha_nno, evidence, true_mask, false_mask, epoch=500, func=torch.log, num_classes=num_classes, focal=True
        )

    def kl_div(self, alpha):
        alpha = alpha[:, :-1, ...]  # not for no_object_class in any case
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

    def kl_div_nno(self, alpha):
        #alpha = alpha[:, :-1, ...]  # not for no_object_class in any case
        alpha_zero = torch.ones(
            (1, self.num_classes), device=alpha.device, dtype=torch.float32
        )
        if self.weighted_kl_div:
            #alpha_zero = self.label_counts_dict*alpha_zero
            prior = Dirichlet(self.empty_weight.to(alpha.device))
            pred = Dirichlet(alpha)
            kl_div5 = torch.distributions.kl.kl_divergence(pred, prior)
            return kl_div5.mean()

        S_alpha = alpha.sum(dim=1, keepdim=True)
        S_beta = alpha_zero.sum(dim=1, keepdim=True)

        lnB = (
            torch.lgamma(S_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            - torch.lgamma(S_beta)
        )

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = ((alpha - 1) * (dg1 - dg0)).sum(dim=1, keepdim=True) + lnB


        return kl

    def kl_div_reg(self, alpha):
        #alpha = alpha[:, :-1, ...]  # not for no_object_class in any case
        alpha_zero = torch.ones(
            (1, self.num_classes), device=alpha.device, dtype=torch.float32
        )
        S_alpha = alpha.sum(dim=1, keepdim=True)
        S_beta = alpha_zero.sum(dim=1, keepdim=True)

        lnB = (
            torch.lgamma(S_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            - torch.lgamma(S_beta)
        )

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = ((alpha - 1) * (dg1 - dg0)).sum(dim=1, keepdim=True) + lnB
        return kl

    def kl_div_weighed(self,alpha):
        prior = Dirichlet(self.empty_weight.to(alpha.device))
        pred = Dirichlet(alpha)
        kl_div5 = torch.distributions.kl.kl_divergence(pred, prior)
        return kl_div5.mean()

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
        yc = F.one_hot(yc, num_classes + 1)
        # yc = yc[:, :, :num_classes]
        yc = yc.permute(0, 2, 1)
        A = (
            (yc * (func(S) - func(alpha)))
            * self.empty_weight.to(alpha.device).unsqueeze(0).unsqueeze(-1)
        ).sum(dim=1, keepdim=True)

        loss = A

        if self.kl_weight > 0:
            alpha_tilde = alpha * (1 - yc) + yc

            B = self.kl_div(alpha_tilde)  # * min(1, epoch / self.kl_annealing)

            loss = loss + self.kl_weight * B

        if loss.shape[0] == 0:
            return {"loss_dpn": 0}
        return {"loss_dpn": loss.mean()}

    def loss_f_nno_clr_reg(self,y,
        alpha: torch.Tensor,
        alpha_nno: torch.Tensor,
        evidence: torch.Tensor,
        epoch: int,
        func=torch.digamma,
        num_classes=None,
        focal=False):

        if num_classes is None:
            num_classes = self.num_classes
        S = alpha.sum(dim=1, keepdim=True)
        yc = y.clone()
        yc = F.one_hot(yc, num_classes)
        # yc = yc[:, :, :num_classes]
        if focal:
            A = focal_loss(alpha, y, self.gamma, self.empty_weight)
            loss = A
        else:
            A = (
                    (yc * (func(S) - func(alpha)))
                    * self.empty_weight.to(alpha.device).unsqueeze(0).unsqueeze(-1)
            ).sum(dim=1, keepdim=True)

            loss = A

        if self.kl_weight > 0:
            # https://proceedings.neurips.cc/paper_files/paper/2018/file/3ea2db50e62ceefceaf70a9d9a56a6f4-Paper.pdf
            if self.kl_weight > 0:
                if self.weighted_kl_div:
                    kl_div_ind = self.kl_div_weighed(alpha)
                else:
                    alpha_tilde = alpha * (1 - yc) + yc
                    kl_div_ind = self.kl_div_reg(alpha_tilde)
            loss = loss + self.kl_weight * kl_div_ind# * min(1, epoch / self.kl_annealing)

        if loss.shape[0] == 0:
            return {"loss_dpn": 0}
        loss=loss.mean()
        if self.kl_weight_nno > 0:
            alpha_zero = torch.ones(
                (alpha_nno.shape[0], self.num_classes), device=alpha.device, dtype=torch.float32
            )
            if self.kl_nno_weighed:
                S_alpha = alpha_nno.sum(dim=1, keepdim=True)
                alpha_zero*=(S_alpha/num_classes)
            prior = Dirichlet(alpha_zero)
            pred = Dirichlet(alpha_nno)
            kl_div_nno = torch.distributions.kl.kl_divergence(pred, prior)
            # B = self.kl_div_nno(alpha)  # * min(1, epoch / self.kl_annealing)
            loss = loss+ self.kl_weight_nno * kl_div_nno.mean()

        if self.entropy_reg_weight >0:
            entropy_reg = Dirichlet(alpha).entropy().mean()
            if self.kl_weight_nno > 0:
                entropy_reg += Dirichlet(alpha_nno).entropy().mean()
                entropy_reg/=2
            loss -= self.entropy_reg_weight * entropy_reg.mean()
        return {"loss_dpn": loss}

    def loss_f_nno_clr_reg_binary(self,y,
        alpha: torch.Tensor,
        alpha_nno: torch.Tensor,
        evidence: torch.Tensor,
        true_mask: torch.Tensor,
        false_mask: torch.Tensor,
        epoch: int,
        func=torch.digamma,
        num_classes=None,
        focal=False):

        if num_classes is None:
            num_classes = self.num_classes
        S = alpha.sum(dim=1, keepdim=True)
        yc = y.clone()
        yc = F.one_hot(yc, num_classes)
        # yc = yc[:, :, :num_classes]
        if focal:
            A = focal_loss(alpha, y, self.gamma, self.empty_weight)
            loss = A
        else:
            #there was before  * self.empty_weight.to(alpha.device).unsqueeze(0).unsqueeze(-1)
            # but this throws an error since empty_weight cannot be broadcasted to the shape of alpha
            A = (
                    (yc * (func(S) - func(alpha)))
                    * self.empty_weight.to(alpha.device).unsqueeze(0)
            ).sum(dim=1, keepdim=True)

            loss = A
        #binary_loss
        combined_mask = torch.cat([true_mask, false_mask])
        mask_labels = torch.cat([torch.ones_like(true_mask), torch.zeros_like(false_mask)]).float()

        #true_mask has weight 1 and false_mask has weight binary_loss_negative_weight
        if self.binary_loss_negative_weight != None:
            weight = (1-mask_labels) * self.binary_loss_negative_weight+mask_labels 
        else:
            weight = None
        binary_loss = F.binary_cross_entropy_with_logits(combined_mask, mask_labels, weight=weight )
        if self.kl_weight > 0:
            # https://proceedings.neurips.cc/paper_files/paper/2018/file/3ea2db50e62ceefceaf70a9d9a56a6f4-Paper.pdf
            if self.kl_weight > 0:
                if self.weighted_kl_div:
                    kl_div_ind = self.kl_div_weighed(alpha)
                else:
                    alpha_tilde = alpha * (1 - yc) + yc
                    kl_div_ind = self.kl_div_reg(alpha_tilde)
            loss = loss + self.kl_weight * kl_div_ind# * min(1, epoch / self.kl_annealing)

        if loss.shape[0] == 0:
            return {"loss_dpn": 0}
        loss=loss.mean()
        if self.kl_weight_nno > 0:
            alpha_zero = torch.ones(
                (alpha_nno.shape[0], self.num_classes), device=alpha.device, dtype=torch.float32
            )
            if self.kl_nno_weighed:
                S_alpha = alpha_nno.sum(dim=1, keepdim=True)
                alpha_zero*=(S_alpha/num_classes)
            prior = Dirichlet(alpha_zero)
            pred = Dirichlet(alpha_nno)
            kl_div_nno = torch.distributions.kl.kl_divergence(pred, prior)
            # B = self.kl_div_nno(alpha)  # * min(1, epoch / self.kl_annealing)
            loss = loss+ self.kl_weight_nno * kl_div_nno.mean()

        if self.entropy_reg_weight >0:
            entropy_reg = Dirichlet(alpha).entropy().mean()
            if self.kl_weight_nno > 0:
                entropy_reg += Dirichlet(alpha_nno).entropy().mean()
                entropy_reg/=2
            loss -= self.entropy_reg_weight * entropy_reg.mean()
        return {"loss_dpn": loss, "loss_binary":binary_loss}

    def loss_f_nno(
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
        yc = F.one_hot(yc, num_classes)
        # yc = yc[:, :, :num_classes]
        #yc = yc.permute(0, 2, 1)
        A = (
            (yc * (func(S) - func(alpha)))
            * self.empty_weight.to(alpha.device).unsqueeze(0)
        ).sum(dim=1, keepdim=True)

        loss = A

        if self.cls_focal_loss_weight > 0:
            alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
            p = alpha / alpha_sum
            p_t = p.gather(1, y.unsqueeze(1)).squeeze(1)
            focal_loss = -((1 - p_t) ** self.gamma) * torch.log(p_t) * self.empty_weight.to(alpha.device)[y]
            loss = loss + self.cls_focal_loss_weight * focal_loss
        if self.kl_weight > 0:
            if self.weighted_kl_div:
                B = self.kl_div_weighed(alpha)
            else:
                alpha_tilde = alpha * (1 - yc) + yc
                B = self.kl_div_reg(alpha_tilde)
            loss = loss + self.kl_weight * B # * min(1, epoch / self.kl_annealing)

        if self.entropy_reg_weight >0:
            entropy_reg = Dirichlet(alpha).entropy()
            loss = loss + self.entropy_reg_weight * entropy_reg.mean()
        if loss.shape[0] == 0:
            return {"loss_dpn": 0}
        return {"loss_dpn": loss.mean()}

    def loss_f_focal(
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

        yc = y.clone()
        yc = F.one_hot(yc, num_classes)

        A=focal_loss(alpha,y,self.gamma,self.empty_weight)
        loss = A
        if self.kl_weight > 0:
            if self.weighted_kl_div:
                B = self.kl_div_weighed(alpha)
            else:
                alpha_tilde = alpha * (1 - yc) + yc
                B = self.kl_div_reg(alpha_tilde)  # * min(1, epoch / self.kl_annealing)

            loss = loss + self.kl_weight * B

        if loss.shape[0] == 0:
            return {"loss_dpn": 0}
        return {"loss_dpn": loss.mean()}

    def loss_f_focal_wno(
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

        yc = y.clone()
        yc = F.one_hot(yc, num_classes+1)

        A = focal_loss(alpha,y,self.gamma,self.empty_weight)
        loss = A
        if self.kl_weight > 0:
            if self.weighted_kl_div:
                B = self.kl_div_weighed(alpha[:, :-1, ...])
            else:
                alpha_tilde = alpha * (1 - yc) + yc
                B = self.kl_div(alpha_tilde)  # * min(1, epoch / self.kl_annealing)

            loss = loss + self.kl_weight * B

        if loss.shape[0] == 0:
            return {"loss_dpn": 0}
        return {"loss_dpn": loss.mean()}

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["labels"], indices)]
        ).to(torch.int64)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight.to(src_logits.device),
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]

        masks = targets["masks"]  # [t for t in targets["masks"]]
        # use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            masks
        ).decompose()  # masks contains different number of masks, pad to greater number and put in one tensor
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # cls_label_weights = (
        #     torch.tensor(
        #         [
        #             self.label_weight_dict[id.item()]
        #             for id in torch.cat(targets["labels"])
        #         ],
        #         device=target_masks.device,
        #     )
        #     if self.label_weight_dict_method is not None
        #     else None
        # )
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        target_masks = target_masks[:, None]
        if self.mask_embed_type == "binary":
            src_masks = src_masks[src_idx]
            src_masks = src_masks[:, None]

            with torch.no_grad():

                # start_n = time.time()
                # point_coords = get_uncertain_point_coords_with_randomness(
                #     src_masks,
                #     lambda logits: calculate_uncertainty(logits),
                #     self.num_points,
                #     self.oversample_ratio,
                #     self.importance_sample_ratio,
                # )
                # end_n = time.time()
                # start_p = time.time()
                point_coords = get_uncertain_point_coords_with_randomness_pos_select(
                    src_masks,
                    target_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                    self.positive_label_sample_ratio,
                    self.min_positiv_points,
                )
                # end_p = time.time()
                # # Write the time differences to a CSV file
                # csv_file = "timing2_results.csv"
                # with open(csv_file, mode="a", newline="") as file:
                #     writer = csv.writer(file)
                #     writer.writerow([end_n - start_n, end_p - start_p])

                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)
            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)
            mask_loss = sigmoid_ce_loss_jit(
                point_logits,
                point_labels,
                self.empty_weight.unsqueeze(1).to(point_logits.device),
                logits=True,
            ).mean()
            dice_loss = dice_loss_jit(point_logits.sigmoid(), point_labels, num_masks)
        elif self.mask_embed_type == "beta":
            alpha = src_masks[:, 0][src_idx]
            beta = src_masks[:, 1][src_idx]
            alpha = alpha[:, None]
            beta = beta[:, None]

            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness_pos_select(
                    (
                        alpha + beta
                        if self.u_type == "evidence"
                        else alpha / (alpha + beta)
                    ),
                    target_masks,
                    lambda logits: calculate_uncertainty_beta(logits, type=self.u_type),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                    self.positive_label_sample_ratio,
                    self.min_positiv_points,
                    align_corners=self.align_corners
                )

                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=self.align_corners,
                    padding_mode=self.padding_mode,
                ).squeeze(1)
                if VIS:
                    vis_sampling(
                        target_masks,
                        point_coords,
                        original=targets["original_image"],
                    )
            point_alpha = point_sample(
                alpha,
                point_coords,
                align_corners=self.align_corners,
                mode=self.interpolation,
                padding_mode=self.padding_mode,
            ).squeeze(1)
            point_beta = point_sample(
                beta,
                point_coords,
                align_corners=self.align_corners,
                mode=self.interpolation,
                padding_mode=self.padding_mode,
            ).squeeze(1)
            if self.mask_loss == "ce":

                if self.kwargs.get("double_ce", False):
                    mask_loss_alpha = sigmoid_ce_loss_jit(
                        point_alpha / (point_alpha + point_beta),
                        point_labels,
                        logits=False,
                    )
                    mask_loss_beta = sigmoid_ce_loss_jit(
                        point_beta / (point_alpha + point_beta),
                        1 - point_labels,
                        logits=False,
                    )
                    mask_loss = (
                        mask_loss_alpha * (point_labels > 0.7).float()
                        + mask_loss_beta * (point_labels < 0.3).float()
                    ).mean()
                elif self.kwargs.get("double_ce_reversed", False):
                    mask_loss_alpha = sigmoid_ce_loss_jit(
                        point_alpha / (point_alpha + point_beta),
                        point_labels,
                        logits=False,
                    )
                    mask_loss_beta = sigmoid_ce_loss_jit(
                        point_beta / (point_alpha + point_beta),
                        1 - point_labels,
                        logits=False,
                    )
                    mask_loss = (
                        mask_loss_alpha * (point_labels < 0.3).float()
                        + mask_loss_beta * (point_labels > 0.7).float()
                    ).mean()
                else:
                    mask_loss = sigmoid_ce_loss_jit(
                        point_alpha / (point_alpha + point_beta),
                        point_labels,
                        logits=False,
                    ).mean()
            elif self.mask_loss == "likelihood":
                mask_loss = beta_likelihood_loss(
                    point_alpha,
                    point_beta,
                    point_labels * (1 - self.kwargs.get("offset", 0.01))
                    + (1 - point_labels) * self.kwargs.get("offset", 0.01),
                    self.kwargs,
                )
                if self.kwargs.get("double_likelihood", False):
                    # change role of alpha and beta and switch pointlabels accordingly
                    mask_loss_reversed = beta_likelihood_loss(
                        point_beta,
                        point_alpha,
                        point_labels * (self.kwargs.get("offset", 0.01))
                        + (1 - point_labels) * (1 - self.kwargs.get("offset", 0.01)),
                        self.kwargs,
                    )
                    mask_loss = (
                        mask_loss * (point_labels > 0.7).float()
                        + mask_loss_reversed * (point_labels < 0.3).float()
                    ).mean()
                elif self.kwargs.get("double_likelihood_reversed", False):
                    # change role of alpha and beta and switch pointlabels accordingly
                    mask_loss_reversed = beta_likelihood_loss(
                        point_beta,
                        point_alpha,
                        point_labels * (self.kwargs.get("offset", 0.01))
                        + (1 - point_labels) * (1 - self.kwargs.get("offset", 0.01)),
                        self.kwargs,
                    )
                    mask_loss = (
                        mask_loss * (point_labels < 0.3).float()
                        + mask_loss_reversed * (point_labels > 0.7).float()
                    ).mean()
                else:
                    mask_loss = mask_loss.mean()
            else:
                raise NotImplementedError(
                    f"mask loss >> {self.mask_loss} << is not implemented"
                )
            dice_loss = dice_loss_jit(
                point_alpha / (point_alpha + point_beta),
                point_labels,
                num_masks,
            )
            if self.kwargs.get("double_dice", False):
                dice_loss += dice_loss_jit(
                    point_beta / (point_alpha + point_beta),
                    1 - point_labels,
                    num_masks,
                )
                dice_loss /= 2
            if self.mask_kl_divergence_weight:
                # kl_div
                pass
            if self.kl_weight_non_mask:
                unmatched_mask = torch.ones_like(src_masks[:, 0, :, 0, 0], dtype=torch.bool)
                unmatched_mask[src_idx] = False
                alpha_nm = src_masks[:, 0][unmatched_mask]
                beta_nm = src_masks[:, 1][unmatched_mask]
                alpha_nm = alpha_nm[:, None]
                beta_nm = beta_nm[:, None]
                point_coords_nm=torch.rand(
                    alpha_nm.shape[0], self.num_points, 2, device=alpha_nm.device
                )
                point_alpha_nm = point_sample(
                    alpha_nm,
                    point_coords_nm,
                    align_corners=False,
                ).squeeze(1)
                point_beta_nm = point_sample(
                    beta_nm,
                    point_coords_nm,
                    align_corners=False,
                ).squeeze(1)
                # alpha_beta_zero = torch.ones(
                #     (1, 1), device=alpha.device, dtype=torch.float32
                # )
                alpha_beta_zero = torch.ones(
                    point_alpha_nm.shape, device=alpha.device, dtype=torch.float32
                )
                if self.kl_non_mask_weighed:
                    S_alpha_beta = point_alpha_nm+point_beta_nm
                    alpha_beta_zero *= (S_alpha_beta / 2)
                prior = Beta(alpha_beta_zero,alpha_beta_zero)
                pred = Beta(point_alpha_nm,point_beta_nm)
                kl_div_nno = torch.distributions.kl.kl_divergence(pred, prior)
                mask_loss+=kl_div_nno.mean()*self.kl_weight_non_mask
            # if self.mask_entropy_weight: ### ALREADY present as reg_likelihood
            #     entropy_mask = Beta(point_alpha,point_beta).entropy()
            #     mask_loss = mask_loss + self.mask_entropy_weight * entropy_mask.mean()
            #     pass

        losses = {"loss_mask": mask_loss, "loss_dice": dice_loss}

        del src_masks
        del target_masks
        return losses

    def loss_masks_bdpn(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]

        --> Create a joint loss?
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_classification = outputs["pred_logits"]
        masks = targets["masks"]  # [t for t in targets["masks"]]
        # use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            masks
        ).decompose()  # masks contains different number of masks, pad to greater number and put in one tensor
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]


        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        target_masks = target_masks[:, None]
        if self.mask_embed_type == "binary":
            raise NotImplementedError("binary Not supported in BetaDirPN")
        elif self.mask_embed_type == "beta":
            alpha = src_masks[:, 0][src_idx]
            beta = src_masks[:, 1][src_idx]
            alpha = alpha[:, None]
            beta = beta[:, None]

            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness_pos_select(
                    (
                        alpha + beta
                        if self.u_type == "evidence"
                        else alpha / (alpha + beta)
                    ),
                    target_masks,
                    lambda logits: calculate_uncertainty_beta(logits, type=self.u_type),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                    self.positive_label_sample_ratio,
                    self.min_positiv_points,
                )

                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)
                if VIS:
                    vis_sampling(
                        target_masks,
                        point_coords,
                        original=targets["original_image"],
                    )

            point_alpha = point_sample(
                alpha,
                point_coords,
                align_corners=False,
            ).squeeze(1)
            point_beta = point_sample(
                beta,
                point_coords,
                align_corners=False,
            ).squeeze(1)
            if self.mask_loss == "ce":

                if self.kwargs.get("double_ce", False):
                    mask_loss_alpha = sigmoid_ce_loss_jit(
                        point_alpha / (point_alpha + point_beta),
                        point_labels,
                        logits=False,
                    )
                    mask_loss_beta = sigmoid_ce_loss_jit(
                        point_beta / (point_alpha + point_beta),
                        1 - point_labels,
                        logits=False,
                    )
                    mask_loss = (
                        mask_loss_alpha * (point_labels > 0.7).float()
                        + mask_loss_beta * (point_labels < 0.3).float()
                    ).mean()
                elif self.kwargs.get("double_ce_reversed", False):
                    mask_loss_alpha = sigmoid_ce_loss_jit(
                        point_alpha / (point_alpha + point_beta),
                        point_labels,
                        logits=False,
                    )
                    mask_loss_beta = sigmoid_ce_loss_jit(
                        point_beta / (point_alpha + point_beta),
                        1 - point_labels,
                        logits=False,
                    )
                    mask_loss = (
                        mask_loss_alpha * (point_labels < 0.3).float()
                        + mask_loss_beta * (point_labels > 0.7).float()
                    ).mean()
                else:
                    mask_loss = sigmoid_ce_loss_jit(
                        point_alpha / (point_alpha + point_beta),
                        point_labels,
                        logits=False,
                    ).mean()
            elif self.mask_loss == "likelihood":
                mask_loss = beta_likelihood_loss(
                    point_alpha,
                    point_beta,
                    point_labels * (1 - self.kwargs.get("offset", 0.01))
                    + (1 - point_labels) * self.kwargs.get("offset", 0.01),
                    self.kwargs,
                )
                if self.kwargs.get("double_likelihood", False):
                    # change role of alpha and beta and switch pointlabels accordingly
                    mask_loss_reversed = beta_likelihood_loss(
                        point_beta,
                        point_alpha,
                        point_labels * (self.kwargs.get("offset", 0.01))
                        + (1 - point_labels) * (1 - self.kwargs.get("offset", 0.01)),
                        self.kwargs,
                    )
                    mask_loss = (
                        mask_loss * (point_labels > 0.7).float()
                        + mask_loss_reversed * (point_labels < 0.3).float()
                    ).mean()
                elif self.kwargs.get("double_likelihood_reversed", False):
                    # change role of alpha and beta and switch pointlabels accordingly
                    mask_loss_reversed = beta_likelihood_loss(
                        point_beta,
                        point_alpha,
                        point_labels * (self.kwargs.get("offset", 0.01))
                        + (1 - point_labels) * (1 - self.kwargs.get("offset", 0.01)),
                        self.kwargs,
                    )
                    mask_loss = (
                        mask_loss * (point_labels < 0.3).float()
                        + mask_loss_reversed * (point_labels > 0.7).float()
                    ).mean()
                else:
                    mask_loss = mask_loss.mean()
            else:
                raise NotImplementedError(
                    f"mask loss >> {self.mask_loss} << is not implemented"
                )
            dice_loss = dice_loss_jit(
                point_alpha / (point_alpha + point_beta),
                point_labels,
                num_masks,
            )
            if self.kwargs.get("double_dice", False):
                dice_loss += dice_loss_jit(
                    point_beta / (point_alpha + point_beta),
                    1 - point_labels,
                    num_masks,
                )
                dice_loss /= 2

        losses = {"loss_mask": mask_loss, "loss_dice": dice_loss}

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        if num_masks == 0:
            loss_map = {
                "labels": "loss_ce",
                "labels_dpn": "loss_dpn",
                "labels_dpn_nno": "loss_dpn",
                "labels_dpn_focal":"loss_dpn",
                "labels_dpn_nno_reg": "loss_dpn",
                "labels_dpn_focal_nno_reg" : "loss_dpn",
                "labels_dpn_focal_wno": "loss_dpn",
                "loss_labels_dpn_binary": "loss_dpn",
                "loss_labels_dpn_focal_binary": "loss_dpn",
                "masks": "loss_mask",
            }
            assert loss in loss_map, f"do you really want to compute {loss} loss?"
            return {
                loss_map[loss]: torch.tensor(0.0, requires_grad=True).to(torch.float)
            }
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "labels_dpn": self.loss_labels_dpn,
            "labels_dpn_nno": self.loss_labels_dpn_nno,
            "labels_dpn_focal": self.loss_labels_dpn_focal,
            "labels_dpn_focal_wno": self.loss_labels_dpn_focal_wno,
            "labels_dpn_nno_reg": self.loss_labels_dpn_nno_reg,
            "labels_dpn_focal_nno_reg": self.loss_labels_dpn_focal_nno_reg,
            "loss_labels_dpn_binary": self.loss_labels_dpn_binary,
            "loss_labels_dpn_focal_binary": self.loss_labels_dpn_focal_binary,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # need to combine beta parameter alpha,beta
        # It is done in the loss function and not in the forward pass, since the values are needed for uncertainty estimation for inference after the forward pass
        # if self.mask_embed_type == "beta":
        #     mask_pred = outputs["pred_masks"]
        #     mask_pred = mask_pred[:, 0] / (
        #         mask_pred[:, 0] + mask_pred[:, 1]
        #     )  # mean of beta distribution, mask_pred has already softplus applied
        #     outputs["pred_masks"] = mask_pred
        #     if "aux_outputs" in outputs:
        #         aux_outputs = []
        #         for aux in outputs["aux_outputs"]:
        #             mask_pred = aux["pred_masks"]
        #             mask_pred = mask_pred[:, 0] / (
        #                 mask_pred[:, 0] + mask_pred[:, 1]
        #             )  # mean of beta distribution, mask_pred has already softplus applied
        #             aux["pred_masks"] = mask_pred
        #             aux_outputs.append(aux)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # labels_list = []
        # masks_list = []
        # for l in labels:
        #     labels_list.append(l[l != 255])
        # for m in masks:
        #     void = m == 255
        #     m[void] = 0
        #     mask = F.one_hot(m).permute(2, 0, 1)
        #     mask[0][void] = 0
        #     masks_list.append(mask)
        # targets["masks"] = masks_list
        # targets["labels"] = labels_list
        # Retrieve the matching between the outputs of the last layer and the targets

        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t) for t in targets["labels"])
        # num_masks = torch.as_tensor(
        #     [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        # )
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_masks)
        # num_masks = torch.clamp(num_masks, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            ret = self.get_loss(loss, outputs, targets, indices, num_masks)
            losses.update(ret)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] = losses[k] * self.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

    def prepare_targets(self, targets):
        labels = []
        gt_masks = []
        for b in range(len(targets["image"])):
            unique_classes = torch.unique(targets["semantic"][b])
            unique_classes = unique_classes[unique_classes != 255]  # hard coded ignore
            labels.append(unique_classes)
            masks = []
            for cls in unique_classes:
                masks.append(targets["semantic"][b] == cls)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                gt_masks.append(
                    torch.zeros(
                        (
                            0,
                            targets["semantic"][b].shape[-2],
                            targets["semantic"][b].shape[-1],
                        )
                    )
                )
            else:
                gt_masks.append(torch.stack([x.clone() for x in masks]))
        targets["labels"] = labels
        targets["masks"] = gt_masks
        return targets
        # # gets instance class
        # h_pad, w_pad = images.tensor.shape[-2:]
        # new_targets = []
        # for targets_per_image in targets:
        #     # pad gt
        #     gt_masks = targets_per_image.gt_masks  # one hot encoded instances
        #     padded_masks = torch.zeros(
        #         (gt_masks.shape[0], h_pad, w_pad),
        #         dtype=gt_masks.dtype,
        #         device=gt_masks.device,
        #     )
        #     padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
        #     new_targets.append(
        #         {
        #             "labels": targets_per_image.gt_classes,  # classes for the one hot encoding
        #             "masks": padded_masks,
        #         }
        #     )
        # return new_targets

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def vis_sampling(target_masks, point_coords, alpha=0.3, stop=-1, original=None):
    num_images = len(target_masks) + (len(original) if original is not None else 0)
    rows = int(num_images / 4)
    print(rows)
    h, w = target_masks.shape[-2:]
    ratio = w // h
    # Step 3: Create a grid plot
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(15 * ratio - 3, 15 - 3),
        gridspec_kw={"wspace": 0.02, "hspace": 0.1},
    )

    for idx, (coord, tm, ax) in tqdm(
        enumerate(zip(point_coords, target_masks, axes.flatten()[: len(point_coords)]))
    ):
        if idx == stop:
            break
        # ax = axes[idx // 4, idx % rows]  # Get the correct subplot
        coord = coord.squeeze() * torch.tensor(
            target_masks.shape[-2:], device=coord.device
        ).unsqueeze(0)
        coord = coord.int()
        tm = tm.squeeze()

        # Extract x and y coordinates
        x_coords = coord[:, 1].cpu().numpy()
        y_coords = coord[:, 0].cpu().numpy()

        # Plot the target mask and mark the points
        ax.imshow(tm.cpu(), cmap="gray")
        sns.kdeplot(
            x=x_coords,
            y=y_coords,
            ax=ax,
            cmap="Reds",
            fill=True,
            alpha=alpha,
            bw_adjust=0.5,
        )

        # ax.set_title(f"Mask {idx + 1}")
        ax.axis("off")

    if original is not None:
        for idx, img in enumerate(original):
            ax = axes[
                (idx + len(target_masks)) // 4,
                (idx + len(target_masks)) % rows,
            ]
            ax.imshow(img.cpu().numpy())
            ax.axis("off")
    # Turn off axes for any empty subplots
    for ax in axes.flatten()[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("tmp.png")


class DirichletPriorNetworkLoss(nn.Module):
    def __init__(self, alpha_0, gamma=2.0, lambda_reg=1.0):
        super(DirichletPriorNetworkLoss, self).__init__()
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.lambda_reg = lambda_reg

    def forward(self, alpha, target):
        # Compute the focal loss component
        alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / alpha_sum
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_loss = -((1 - p_t) ** self.gamma) * torch.log(p_t)
        focal_loss = focal_loss.mean()

        # Compute the regularization loss component (KL-divergence)
        prior = Dirichlet(self.alpha_0)
        pred = Dirichlet(alpha)
        kl_div = torch.distributions.kl.kl_divergence(pred, prior)
        reg_loss = kl_div.mean()

        # Combine the focal loss and the regularization loss
        total_loss = focal_loss + self.lambda_reg * reg_loss
        return total_loss

def focal_loss(alpha,y,gamma,class_weights):
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / alpha_sum
    p_t = p.gather(1, y.unsqueeze(1)).squeeze(1)
    focal_loss = -((1 - p_t) ** gamma) * torch.log(p_t)*class_weights.to(alpha.device)[y]
    return focal_loss

def focal_loss2(alpha,y,gamma,class_weights,func,num_classes):
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
    yc = y.clone()
    yc = F.one_hot(yc, num_classes)
    # yc = yc[:, :, :num_classes]
    # yc = yc.permute(0, 2, 1)
    p = alpha / alpha_sum
    p_t = p.gather(1, y.unsqueeze(1)).squeeze(1)
    focal_loss = -((1 - p_t) ** gamma) * (yc * (func(alpha_sum) - func(alpha)))*class_weights.to(alpha.device)
    return focal_loss

def kl_divergence_dirichlet(alpha, alpha_0):
    # Compute the sum of the alpha parameters
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
    alpha_0_sum = torch.sum(alpha_0)

    # Compute the log term involving the Gamma function
    log_gamma_alpha_sum = torch.lgamma(alpha_sum)
    log_gamma_alpha_0_sum = torch.lgamma(alpha_0_sum)

    # Compute the sum of the log Gamma terms for individual alpha parameters
    log_gamma_alpha = torch.lgamma(alpha)
    log_gamma_alpha_0 = torch.lgamma(alpha_0)

    # Compute the digamma terms
    digamma_alpha = torch.digamma(alpha)
    digamma_alpha_sum = torch.digamma(alpha_sum)

    # Compute the KL divergence
    kl_div = log_gamma_alpha_0_sum - log_gamma_alpha_sum + \
             torch.sum(log_gamma_alpha - log_gamma_alpha_0, dim=1) + \
             torch.sum((alpha - alpha_0) * (digamma_alpha - digamma_alpha_sum), dim=1)

    return kl_div.mean()


def sample_indices_by_inverse_frequency(tensor, n):
    """
    Sample n unique indices from the given tensor with probability proportional to the inverse frequency of the elements.

    Args:
        tensor (torch.Tensor): The input tensor.
        n (int): The number of indices to sample.

    Returns:
        torch.Tensor: The sampled indices.
    """
    if n > len(tensor):
        return torch.arange(len(tensor))

    # Calculate the frequency of each element
    unique_elements, counts = torch.unique(tensor, return_counts=True)

    # Calculate the inverse frequency of each element
    inverse_freq = 1.0 / counts.float()

    # Create a probability vector with the inverse frequency of each element
    probs = torch.zeros_like(tensor, dtype=torch.float)
    for i, elem in enumerate(unique_elements):
        probs[tensor == elem] = inverse_freq[i]

    # Normalize the probabilities
    probs = probs / probs.sum()

    # Sample the indices based on the probabilities
    sampled_indices = torch.multinomial(probs, num_samples=n, replacement=False)

    return sampled_indices