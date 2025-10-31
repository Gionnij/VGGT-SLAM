# ------------------------------------------------------------------------------
# Panoptic-DeepLab meta architecture.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from typing import Dict

import torch

from Visualizations.util import plot_panoptic_prediction
from mlflow.tracking import MlflowClient

from ....losses import get_loss
from .postprocess import (
    get_panoptic_segmentation,
    get_semantic_segmentation,
)
from torch import nn
from torch.nn import functional as F

from .base import BaseSegmentationModel
from .decoder import PanopticDeepLabDecoder
from PIL import Image
import os
import numpy as np


class PanopticDeepLab(BaseSegmentationModel):
    """
    Implements Panoptic-DeepLab model from
    `"Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation"
    <https://arxiv.org/abs/1911.10194>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        in_channels (int): number of input channels from the backbone
        feature_key (str): names of input feature from backbone
        low_level_channels (list): a list of channels of low-level features
        low_level_key (list): a list of name of low-level features used in decoder
        low_level_channels_project (list): a list of channels of low-level features after projection in decoder
        decoder_channels (int): number of channels in decoder
        atrous_rates (tuple): atrous rates for ASPP
        num_classes (int): number of classes
        semantic_loss (nn.Module): loss function
        semantic_loss_weight (float): loss weight
        center_loss (nn.Module): loss function
        center_loss_weight (float): loss weight
        offset_loss (nn.Module): loss function
        offset_loss_weight (float): loss weight
        **kwargs: arguments for instance head
    """

    def __init__(
        self,
        backbone,
        in_channels,
        feature_key,
        low_level_channels,
        low_level_key,
        low_level_channels_project,
        decoder_channels,
        atrous_rates,
        num_classes,
        losses: Dict,
        **kwargs,
    ):
        decoder = PanopticDeepLabDecoder(
            in_channels,
            feature_key,
            low_level_channels,
            low_level_key,
            low_level_channels_project,
            decoder_channels,
            atrous_rates,
            num_classes,
            **kwargs,
        )
        super(PanopticDeepLab, self).__init__(backbone, decoder)

        self.losses = {}
        for loss_name in losses:
            self.losses[loss_name] = {
                "weight": losses[loss_name]["weight"],
                "loss": get_loss(**losses[loss_name]),
            }

        # Initialize parameters.
        self._init_params()

    def training_step(self, x, out):
        pass

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
        Args:
            pred (dict): stores all output of the segmentation model.
            input_shape (tuple): spatial resolution of the desired shape.
        Returns:
            result (OrderedDict): upsampled dictionary.
        """
        # Override upsample method to correctly handle `offset`
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(
                pred[key], size=input_shape, mode="bilinear", align_corners=True
            )
            if "offset" in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def loss(self, results, targets=None, **kwargs):
        batch_size = results["semantic"].size(0)
        losses = {"total": 0}
        if targets is not None:
            if "semantic_weights" in targets.keys():
                semantic_loss = (
                    self.losses["semantic"]["loss"](
                        results["semantic"],
                        targets,
                        semantic_weights=targets["semantic_weights"],
                        epoch=kwargs["epoch"],
                    )
                    * self.losses["semantic"]["weight"]
                )
            else:
                semantic_loss = (
                    self.losses["semantic"]["loss"](
                        results["semantic"], targets["semantic"]
                    )
                    * self.losses["semantic"]["weight"]
                )
            losses["semantic"] = semantic_loss
            losses["total"] += semantic_loss
            if "center" in self.losses:
                # Pixel-wise loss weight
                center_loss_weights = targets["center_weights"][
                    :, None, :, :
                ].expand_as(results["center"])
                center_loss = (
                    self.losses["center"]["loss"](results["center"], targets["center"])
                    * center_loss_weights
                )
                # safe division
                if center_loss_weights.sum() > 0:
                    center_loss = (
                        center_loss.sum()
                        / center_loss_weights.sum()
                        * self.losses["center"]["weight"]
                    )
                else:
                    center_loss = center_loss.sum() * 0
                losses["center"] = center_loss
                losses["total"] += center_loss
            if "offset" in self.losses:
                # Pixel-wise loss weight
                offset_loss_weights = targets["offset_weights"][
                    :, None, :, :
                ].expand_as(results["offset"])
                offset_loss = (
                    self.losses["offset"]["loss"](results["offset"], targets["offset"])
                    * offset_loss_weights
                )
                # safe division
                if offset_loss_weights.sum() > 0:
                    offset_loss = (
                        offset_loss.sum()
                        / offset_loss_weights.sum()
                        * self.losses["offset"]["weight"]
                    )
                else:
                    offset_loss = offset_loss.sum() * 0
                losses["offset"] = offset_loss
                losses["total"] += offset_loss
            if "rgb" in self.losses:
                rgb_loss = (
                    self.losses["rgb"]["loss"](
                        results["rgb"]["result"], targets["image"]
                    )
                    * self.losses["rgb"]["weight"]
                )
                losses["rgb"] = rgb_loss.sum() * self.losses["rgb"]["weight"]
                losses["total"] += rgb_loss.sum() * self.losses["rgb"]["weight"]
        return losses

    def set_output_stride(self, os):
        self.backbone.set_output_stride(os)
        self.decoder.set_output_stride(os)

    def postprocess(self, out, dm, post_process_conf):
        batch_size = out["semantic"].shape[0]

        semantic_preds = []
        panoptic_preds = []
        center_preds = []
        uncertainties = []

        for i in range(batch_size):
            semantic = out["semantic"][i].unsqueeze(0)
            center = out["center"][i].unsqueeze(0)
            offset = out["offset"][i].unsqueeze(0)
            # semantic_softmax = F.softmax(semantic.squeeze(), dim=0)
            # uncertainty = (torch.max(semantic_softmax, dim=0)[0] * 255).type(torch.int8)
            uncertainty = self.decoder.semantic_head.get_uncertainty(out, i)

            semantic_pred = get_semantic_segmentation(semantic).unsqueeze(0)

            panoptic_pred, center_pred = get_panoptic_segmentation(
                semantic,
                center,
                offset,
                thing_list=dm.mapped_thing_list,
                label_divisor=dm.label_divisor,
                stuff_area=post_process_conf["stuff_area"],
                void_label=(dm.label_divisor * 255),
                threshold=post_process_conf["center_threshold"],
                nms_kernel=post_process_conf["nms_kernel"],
                top_k=post_process_conf["top_k_instance"],
                foreground_mask=None,
            )

            semantic_preds.append(semantic_pred.squeeze())
            panoptic_preds.append(panoptic_pred.squeeze())
            center_preds.append(center_pred)
            uncertainties.append(uncertainty)

        semantic_preds = torch.stack(semantic_preds)
        panoptic_preds = torch.stack(panoptic_preds)
        uncertainties = torch.stack(uncertainties)

        return {
            "semantic": semantic_preds,
            "panoptic": panoptic_preds,
            "center": center_preds,
            "uncertainty": uncertainties,
            "certainties": -uncertainties,
        }

    def plot_prediction(
        self,
        x,
        batch,
        postprocess,
        out,
        dm,
        experiment,
        batch_idx,
        global_step,
        log_folder, 
        **kwargs,
    ):
        grid, panoptic_gt_img, _ = plot_panoptic_prediction(
            x,
            batch["semantic"],
            batch["instance"],
            postprocess["semantic"],
            postprocess["panoptic"],
            out["center"].cpu(),
            out["offset"].cpu(),
            datamodule=dm,
            extra_imgs=postprocess["uncertainty"],
        )

        if isinstance(experiment, MlflowClient):

            img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img.save(os.path.join(log_folder, f"current/predictions-{batch_idx}.png"))
            # Optionally log image artifact via MLflow:
            # experiment.log_artifact(run_id=self.run_id, local_path=..., artifact_path=...)
        else:
            experiment.add_image(f"predictions-{batch_idx}", grid, global_step)

            def collect_set_data(self, batch, out, postprocess, dm):
                # used in Prototypicall Deeplab for returning mean embeddings per class, i.e providing additional validition and interpretation
                return None

            def process_set_data(self, valset_data, experiment, dm):
                # used in prototypicall deeplab for visualization
                return None

    def collect_set_data(self, batch, out, postprocess, dm):
        pass