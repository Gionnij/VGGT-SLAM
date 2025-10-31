# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import os
import traceback
from typing import Any, Dict, List, Set

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from PIL import Image
from mlflow.tracking import MlflowClient
from utils.logging_utils.log_writers import (
    gl_error,
)


# from detectron2.config import configurable
# from detectron2.data import MetadataCatalog
# from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head

# from detectron2.modeling.backbone import Backbone #use get_backbone function
# from detectron2.modeling.postprocessing import sem_seg_postprocess

from Benchmarks.Open_World_Benchmark.estimators import get_estimator
from models.mask2former.detectron2.layers.shape_spec import ShapeSpec
from Visualizations.segmentation import (
    segmentation_to_img,
    get_color_range,
)
from Visualizations.util import (
    plot_panoptic_prediction,
    tsne_embedding,
    plot_panoptic_img,
)

from models.mask2former.detectron2.structures import (
    Boxes,
    Instances,
)
from Data_Loaders.lightning_data_modules.transforms.panoptic import (
    remap_instance,
)
from models.mask2former.detectron2.utils.memory import retry_if_cuda_oom
from models.losses import get_loss
from .modeling import MaskFormerHead


# from ..DeepLabV3Base.deeplabv3.panopticdl import get_backbone


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result


class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    # Take Dictionaries instead of actual moduls and then call functions to get those dictionaries
    # @configurable
    def __init__(
        self,
        *,
        backbone: nn.Module,
        sem_seg_head: nn.Module,
        # criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        # pixel_mean: Tuple[float],
        # pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        losses: dict,
        dm,
        split_rows: int,
        vis_embeding: bool,
        mask_embed_type: str = "binary",
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.losses = {}
        self.split_rows = split_rows
        self.num_classes = dm.num_classes
        self.dm = dm
        for loss_name in losses:
            self.losses[loss_name] = {
                "weight": losses[loss_name]["weight"],
                "loss": get_loss(
                    **losses[loss_name], num_classes=self.num_classes, dm=dm
                ),
            }
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        # self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        # self.register_buffer(
        #    "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        # )
        # self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = True
        self.instance_on = instance_on
        self.panoptic_on = True
        self.vis_embeding = vis_embeding
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        self.mask_embed_type = mask_embed_type
        self.overwrite_stuff_classes_in_panoptic = True

    @classmethod
    def from_config(cls, cfg, dm):

        # FIXME, quick hack to not break exisitng runs. Decide on one version after experiments
        if "RESNETS" in cfg:
            from models.mask2former.detectron2.modeling.backbone.resnet import (
                build_resnet_backbone,
            )

            backbone = build_resnet_backbone(
                cfg,
                input_shape=ShapeSpec(channels=3, height=None, width=None, stride=None),
            )
        elif "SWIN" in cfg:
            from models.mask2former.detectron2.modeling.backbone.swin import (
                D2SwinTransformer,
            )

            backbone = D2SwinTransformer(cfg, input_shape=None)
            if (
                "backbone_file" in cfg["backbone"]
                and cfg["backbone"]["backbone_file"] is not None
                and cfg["backbone"]["backbone_file"] != ""
            ):
                print(
                    "Loaded backbone from: ",
                    os.path.expanduser(cfg["backbone"]["backbone_file"]),
                )
                print(
                    backbone.load_state_dict(
                        torch.load(
                            os.path.expanduser(cfg["backbone"]["backbone_file"])
                        ),
                        strict=False,
                    )
                )

        else:
            from Prior2Former import (
                get_backbone,
            )

            backbone = get_backbone(**cfg["backbone"])
        cfg["args"]["num_classes"] = dm.num_classes
        cfg["SEM_SEG_HEAD"]["NUM_CLASSES"] = dm.num_classes
        sem_seg_head = MaskFormerHead(
            **MaskFormerHead.from_config(cfg, backbone.output_shape())
        )
        ret = {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            # "criterion": criterion,
            "num_queries": cfg["args"]["NUM_OBJECT_QUERIES"],
            "object_mask_threshold": cfg["args"]["TEST"]["OBJECT_MASK_THRESHOLD"],
            "overlap_threshold": cfg["args"]["TEST"]["OVERLAP_THRESHOLD"],
            # "metadata": MetadataCatalog.get(cfg["args"]["DATASETS"]["TRAIN"][0]),
            "size_divisibility": cfg["args"]["SIZE_DIVISIBILITY"],
            "sem_seg_postprocess_before_inference": (
                cfg["args"]["TEST"]["SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE"]
                or cfg["args"]["TEST"]["PANOPTIC_ON"]
                or cfg["args"]["TEST"]["INSTANCE_ON"]
            ),
            # "pixel_mean": cfg["args"]["PIXEL_MEAN"],
            # "pixel_std": cfg["args"]["PIXEL_STD"],
            # inference
            "semantic_on": cfg["args"]["TEST"]["SEMANTIC_ON"],
            "instance_on": cfg["args"]["TEST"]["INSTANCE_ON"],
            "panoptic_on": cfg["args"]["TEST"]["PANOPTIC_ON"],
            "test_topk_per_image": cfg["args"]["DETECTIONS_PER_IMAGE"],
            "losses": cfg["args"]["losses"],
            "split_rows": cfg["args"]["split_rows"],
        }
        if "mask_embed_type" in cfg["args"]:
            ret["mask_embed_type"] = cfg["args"]["mask_embed_type"]
        ret["vis_embeding"] = cfg["args"].get("vis_embeding", True)
        return ret

    def get_model_parameter_dict(self, optimizer: dict):
        weight_decay_norm = optimizer["WEIGHT_DECAY_NORM"]
        weight_decay_embed = optimizer["WEIGHT_DECAY_EMBED"]

        defaults = {}
        defaults["lr"] = optimizer["args"]["lr"]
        defaults["weight_decay"] = optimizer["args"]["weight_decay"]

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in self.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * optimizer["backbone_lr_factor"]
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
        return params

    @property
    def device(self):
        return "cuda"

    def forward(self, images):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        features = self.backbone(images)
        outputs = self.sem_seg_head(features)
        return outputs

    def loss(self, results, targets, step, epoch, training, dm):
        losses = {"total": 0}
        if "mask" in self.losses:
            loss = self.losses["mask"]["loss"](results, targets)
            for k, v in loss.items():
                losses["total"] += v
                losses[k] = v.clone().detach().cpu()
        if "discrimitativ" in self.losses:
            loss = self.losses["discrimitativ"]["loss"](
                results["embedding"], targets, num_classes=dm.num_classes
            )["total"]
            losses["total"] += loss * self.losses["discrimitativ"]["weight"]
            losses["discrimitativ"] = loss.clone().detach().cpu()
        if "contrastive" in self.losses:
            loss = self.losses["contrastive"]["loss"](
                results["embedding"], targets, num_classes=dm.num_classes
            )
            losses["total"] += loss * self.losses["contrastive"]["weight"]
            losses["contrastive"] = loss.clone().detach().cpu()
        return losses

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
        ignore_instance=None,
        oodis=False,
        ood_color=None,
        ood_dif=None,
        min_ood=0.5,
    ):
        extra_vis = []
        if "rgb" in out:
            out_rgb = out["rgb"]["result"]
            image = out_rgb.cpu().detach()
            image = dm.unnormalize(image)
            extra_vis.append(image)
        if "uncertainty" in postprocess:
            uncertainty = (
                postprocess["uncertainty"].squeeze().clone().detach().cpu()
            )  # Shape becomes (num_cls,512, 1024)
            uncertainty = (uncertainty - uncertainty.min()) / (
                uncertainty.max() - uncertainty.min()
            )
            if len(uncertainty.shape) > 2:
                img = torch.zeros((uncertainty.shape[0], 3, *uncertainty.shape[-2:]))
                img[:, 0] = uncertainty.clone().detach()
            else:
                img = torch.zeros((3, *uncertainty.shape[-2:]))
                img[0] = uncertainty.clone().detach()
            extra_vis.append(img)
        if self.vis_embeding and "embedding" in out:
            try:
                tsne = self.visualize_embed(out, postprocess)
                extra_vis.append(tsne.cpu())
            except Exception as e:
                gl_error(f"Got error: {str(e)} with trace {traceback.format_exc()}")
        if "ood_mask" in postprocess:
            ood_mask = postprocess["ood_mask"]
            ood_mask = ood_mask.int()
            outlier_flag = (ood_mask == -1).any()
            ood_mask[ood_mask == -1] = ood_mask.max() + 1
            colors = get_color_range(ood_mask.max() + 1 - outlier_flag.int())
            if len(colors) < 2 and outlier_flag:
                colors = [torch.zeros(3), torch.ones(3)]
            elif outlier_flag:
                colors[0] = torch.zeros(3)
                colors.append(torch.ones(3))
            else:
                colors[0] = torch.zeros(3)
            ood_mask_img = segmentation_to_img(
                ood_mask.unsqueeze(0),
                ood_mask.max() + 1,
                colors=colors,
            ).squeeze()
            extra_vis.append(ood_mask_img)
        if "panoptic_vis" in postprocess:
            pan_vis = plot_panoptic_img(
                postprocess["panoptic_vis"].squeeze(),
                dm,
                ood_color=ood_color,
                ood_dif=ood_dif,
                min_ood=min_ood,
            )
            open_world = plot_panoptic_img(
                (postprocess["panoptic_vis"].squeeze() // 1000) * 1000,
                dm,
                ood_color=ood_color,
                ood_dif=ood_dif,
                min_ood=min_ood,
            )
            extra_vis.append(pan_vis.unsqueeze(0))
            extra_vis.append(open_world.unsqueeze(0))

        grid, panoptic_gt_img, _ = plot_panoptic_prediction(
            x,
            dm.get_semantic_from_batch(batch),
            torch.zeros_like(batch["semantic"]),
            postprocess["semantic"],
            (
                postprocess["panoptic"]
                if "panoptic" in postprocess
                else torch.zeros_like(postprocess["semantic"])
            ),
            datamodule=dm,
            extra_imgs=extra_vis,
            split_rows=self.split_rows,
            ignore_instance=ignore_instance,
            return_list=oodis,
        )

        if experiment is None:
            return grid, panoptic_gt_img
        if isinstance(experiment, MlflowClient):
            img = Image.fromarray(
                (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            img.save(os.path.join(log_folder, f"current/predictions-{batch_idx}.png"))
            # experiment.log_image(
            #     run_id=self.run_id,
            #     image=img,
            #     artifact_file=f"current/predictions-{batch_idx}.png",
            # )
        else:
            experiment.add_image(f"predictions-{batch_idx}", grid, global_step)

        # maybe plot mask embeddings

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def visualize_embed(self, out, postprocess, **kwargs):
        if "perplexity" not in kwargs:
            kwargs["perplexity"] = 10
        if "metric" not in kwargs:
            kwargs["metric"] = "cosine"  # "euclidean"
        if "n_neighbors" not in kwargs:
            kwargs["n_neighbors"] = 90
        tsne_embed = tsne_embedding(out["embedding"], **kwargs)

        tsne_embed = F.interpolate(
            tsne_embed,
            size=postprocess["semantic"].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        return tsne_embed

    def semantic_inference_keep(self, mask_cls, mask_pred):
        scores, labels = self.sem_seg_head.predictor.class_embed.probabilities(
            mask_cls
        ).max(-1)
        # mask_pred = mask_pred

        keep = labels.ne(self.sem_seg_head.num_classes)

        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        semseg = torch.einsum("qc,qhw->chw", cur_mask_cls, cur_masks)
        return semseg

    def semantic_inference(self, mask_cls, mask_pred,num_classes=-1):
        # mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        #remove last class only if present
        num_classes=self.sem_seg_head.num_classes
        mask_cls = self.sem_seg_head.predictor.class_embed.probabilities(mask_cls)[
            ..., :num_classes
        ]
        # mask_pred = mask_pred
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def certainty_inference(
        self, mask_cls, mask_pred, class_logits, semantic, alpha, beta
    ):
        if self.mask_embed_type == "binary":
            return -1 * (
                torch.gather(class_logits.squeeze(0).softmax(0), 0, semantic)
                .squeeze()
                .detach()
            )
        elif self.mask_embed_type == "beta":
            return (1 / (alpha + beta)).mean(0)
        else:
            return torch.zeros_like(semantic)

    def panoptic_inference(self, mask_cls, mask_pred):

        if self.sem_seg_head.predictor.class_embed.__class__.__name__ == "DPNHead_BinaryNNO":
            # DPNHead_BinaryNNO hase a binary prediction for the no-object class, hence it is decoupled from the class prediciton
            scores, labels = self.sem_seg_head.predictor.class_embed.probabilities(
                mask_cls[...,:-1]
            ).max(-1)
            mask_scores =self.sem_seg_head.predictor.class_embed.mask_probabilities(
                mask_cls
            )
            keep = (mask_scores > 0.5) & (
                    scores > self.object_mask_threshold  
            )
        else:
            scores, labels = self.sem_seg_head.predictor.class_embed.probabilities(
                mask_cls
            ).max(-1)
            
            keep = labels.ne(self.sem_seg_head.num_classes) & (
                scores > self.object_mask_threshold  # softmax threshholding
            )

            
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        if self.sem_seg_head.predictor.class_embed.__class__.__name__ != "DPNHead_BinaryNNO":
            cur_mask_cls = cur_mask_cls[..., :-1]

        # cur_scores.shape [17], cur_masks.shape [17,h,w]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(
                0
            )  # cur_prob_masks = sigmoid * softmax for every mask -> cur_mask_ids is class of mask with highest score
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.dm.mapped_thing_list
                mask_area = (cur_mask_ids == k).sum().item()  # area
                original_area = (
                    (cur_masks[k] >= 0.5).sum().item()
                )  # masks thresholding, area of mask k of the current masks
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if (
                    mask_area > 0 and original_area > 0 and mask.sum().item() > 0
                ):  # in case there is an actual mask
                    if (
                        mask_area / original_area < self.overlap_threshold
                    ):  # argmax area ~ logit area
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def postprocess(self, out, dm, post_process_conf, semantic_gt=None):
        unknown_clustering = (
            post_process_conf.get("unknown_clustering", {})
            if post_process_conf is not None
            else {}
        )

        width, height = dm.base_size_val
        mask_cls_results = out["pred_logits"]
        mask_pred_results = out["pred_masks"]
        embeddings = out["embedding"]
        device = mask_cls_results.device
        alphas = betas = None
        if self.mask_embed_type == "beta":
            # mean of beta distribution
            alphas = F.interpolate(
                mask_pred_results[:, 0],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            betas = F.interpolate(
                mask_pred_results[:, 1],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            mask_pred_results = alphas / (alphas + betas)
        else:
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()

        # loops through the batch
        # assert mask_cls_results.shape[0] == 1, "implemented for validation batch size 1"
        processed_results = {
            "semantic_logits": [],
            "semantic": [],
            "mask_pred": [],
            "mask_cls": [],
            "uncertainty": [],
            "ood_mask": [],
            "segment_info": [],
            "instances": [],
            "panoptic": [],
        }
        image_size = (height, width)
        for i, (mask_cls_result, mask_pred_result, embedding) in enumerate(
            zip(mask_cls_results, mask_pred_results, embeddings)
        ):
            # height = input_per_image.get("height", image_size[0])
            # width = input_per_image.get("width", image_size[1])
            # processed_results.append({})
            if alphas != None:
                alpha, beta = alphas[i], betas[i]
            else:
                alpha = beta = None

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results["semantic_logits"].append(r.unsqueeze(0))
                semantic = torch.argmax(r, dim=0).unsqueeze(0)
                processed_results["semantic"].append(semantic)
                processed_results["mask_pred"].append(
                    retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    .cpu()
                    .unsqueeze(0)
                )
                processed_results["mask_cls"].append(mask_cls_result.cpu().unsqueeze(0))

                torch.cuda.empty_cache()

                if "uncertanty_estimator" in post_process_conf:
                    if isinstance(post_process_conf["uncertanty_estimator"], dict):
                        estimator = get_estimator(
                            post_process_conf["uncertanty_estimator"].get("name"),
                            args={
                                "out_size": (height, width),
                                "normalize": False,
                                "threshold": post_process_conf[
                                    "uncertanty_estimator"
                                ].get("threshold", 0),
                            },
                        )
                    else:
                        estimator = get_estimator(
                            name=post_process_conf["uncertanty_estimator"],
                            args={"out_size": (height, width), "normalize": False},
                        )
                    estimator.set_mean_std()
                    uncertainty = estimator(self, out=out)

                else:
                    if self.mask_embed_type == "binary":
                        uncertainty = -1 * (
                            torch.gather(r.squeeze(0).softmax(0), 0, semantic)
                            .squeeze()
                            .detach()
                            .cpu()
                        )
                    elif self.mask_embed_type == "beta":
                        uncertainty = (1 / (alpha.cpu() + beta.cpu())).mean(0)
                    else:
                        uncertainty = torch.zeros_like(semantic).cpu()

                # softmax over the predicitons as uncertainty
                # uncertainty = -1 * (
                #     torch.gather(r.squeeze(0).softmax(0), 0, semantic)
                #     .squeeze()
                #     .detach()
                # )
                processed_results["uncertainty"].append(uncertainty.unsqueeze(0))
            
            


            # instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(
                    mask_cls_result, mask_pred_result
                )
                processed_results["instances"].append(instance_r)

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r, segment_info = retry_if_cuda_oom(self.panoptic_inference)(
                    mask_cls_result, mask_pred_result
                )
                mask = panoptic_r.unsqueeze(0) == 0
                panoptic_r = panoptic_r.unsqueeze(0) + self.dm.label_divisor * semantic
                panoptic_r[mask] = 255 * self.dm.label_divisor
                processed_results["segment_info"].append(segment_info)
            semantic = semantic.cpu()
            uncertainty_small = uncertainty.clone()

            if unknown_clustering:
                # if semantic_gt is not None:
                #     # set the outlier class of the dataset to the anomaly prediction of th edb scan, such that they are ignored for the ood visualisation
                #     uncertainty_small[semantic_gt] = uncertainty_small.min() - 0.5

                uncertainty_small = F.interpolate(
                    uncertainty_small.unsqueeze(0).unsqueeze(0),
                    size=embedding.shape[-2:],
                    mode="bilinear",
                ).squeeze()
                # ood_mask = uncertainty_small > torch.quantile(
                #     uncertainty_small,
                #     q=post_process_conf["unknown_clustering"].get(
                #         "uncertainty_threshold", 0.9
                #     ),
                # )
                ood_mask = uncertainty_small > post_process_conf[
                    "unknown_clustering"
                ].get("uncertainty_threshold", 0.9)

                if ood_mask.sum() == 0 and unknown_clustering.get("only_ood", False):
                    panoptic_r = torch.zeros_like(panoptic_r, device="cpu")
                elif ood_mask.sum() > 0:
                    import cudf
                    import cuml

                    ood_embedding = embedding[
                        :,
                        ood_mask,
                    ]

                    # Here, cosine similarity, replace by generic distance matrix
                    if unknown_clustering.get("distance_type", "") == "cosine":
                        ood_embedding = F.normalize(ood_embedding, p=2, dim=0)
                        dist_matrix = 1 - torch.matmul(ood_embedding.T, ood_embedding)
                    elif unknown_clustering.get("distance_type", "l2") == "l2":
                        dist_matrix = torch.cdist(ood_embedding.T, ood_embedding.T, p=2)
                    else:
                        raise NotImplementedError(unknown_clustering["distance_type"])
                    dist_matrix = dist_matrix.detach().cpu()
                    del mask_pred_result
                    torch.cuda.empty_cache()
                    gdf_distance_matrix = cudf.DataFrame.from_records(
                        dist_matrix.numpy()
                    )
                    torch.cuda.empty_cache()
                    dbscan = cuml.cluster.DBSCAN(
                        eps=unknown_clustering["eps"],
                        min_samples=unknown_clustering["min_samples"],
                        metric="precomputed",
                    )
                    labels = dbscan.fit_predict(gdf_distance_matrix)
                    labels = torch.tensor(labels.to_numpy()).long()
                    unique_labels, counts = labels.unique(return_counts=True)
                    print(
                        f"Min count: {unknown_clustering.get('min_counts', 0)} filers out {(counts < unknown_clustering.get('min_counts', 0)).sum()} labels"
                    )
                    unique_labels[counts < unknown_clustering.get("min_counts", 0)] = -1
                    labels = remap_instance(labels, -1)

                    labels[labels == 0] = labels.max() + 1
                    ood_mask = ood_mask.long().cpu()
                    ood_mask[ood_mask.clone().bool()] = labels

                    ood_mask = (
                        F.interpolate(
                            ood_mask.unsqueeze(0).unsqueeze(0).to(torch.float32),
                            size=panoptic_r.shape[-2:],
                            mode="nearest",
                        )
                        .squeeze()
                        .long()
                    )

                    labels = ood_mask[
                        ood_mask.bool()
                    ]  # reasign interpolated labels, this works since labels has no 0

                    ood_copy = ood_mask.clone()
                    ood_copy[ood_copy == -1] = 0
                    processed_results["ood_mask"].append(ood_copy)


                    if unknown_clustering.get("only_ood", False):
                        print("only_ood")
                        labels[labels == -1] = 0
                        panoptic_r = torch.zeros_like(panoptic_r, device="cpu")
                        panoptic_r[0][ood_mask.bool()] = labels
                        processed_results["panoptic"].append(panoptic_r.to(device))
                        continue
                    elif unknown_clustering.get("ood_and_only_ood", False):
                        # Reassign Outliers to the old label

                        old_label = panoptic_r[0][ood_mask.bool()]
                        labels2 = labels.clone()

                        labels2 = labels2.to(old_label.device)
                        labels2[labels2 > 0] = 254 * 1000 + labels2[labels2 > 0]
                        labels2[labels2 == -1] = old_label[labels2 == -1]

                        panoptic_r = panoptic_r.cpu()
                        panoptic_r[0, ood_mask.bool()] = labels2.cpu()
                        processed_results["panoptic_vis"] = [panoptic_r]

                        print("ood_and_only_ood")
                        panoptic_r = torch.zeros_like(panoptic_r, device="cpu")
                        labels[labels == -1] = 0
                        panoptic_r[0][ood_mask.bool()] = labels
                        processed_results["panoptic"].append(panoptic_r.to(device))
                        panoptic_r = panoptic_r.clone()

                        continue

                    # Reassign Outliers to the old label
                    old_label = panoptic_r[0][ood_mask.bool()]
                    labels = labels.to(old_label.device)
                    labels[labels > -1] += 254 * 1000
                    labels[labels == -1] = old_label[labels == -1]

                    panoptic_r[0, ood_mask.bool()] = labels

            #uncertainty thresholded in the results for open cityscapes according to u3hs
            elif "open" in post_process_conf:
                panoptic_r[uncertainty.unsqueeze(0) > post_process_conf["open"].get("uncertainty_threshold")] = 254 * 1000

            processed_results["panoptic"].append(panoptic_r.to(device))

        processed_results = {k: v for k, v in processed_results.items() if len(v) != 0}

        # Synchronize stuff classes between semantic and panoptic prediction, avoiding OOD regions
        if self.overwrite_stuff_classes_in_panoptic and not unknown_clustering.get("only_ood", False) and self.semantic_on and self.panoptic_on and "panoptic" in processed_results:
            panoptic_mod = torch.cat(processed_results["panoptic"], dim=0).clone()
            semantic = torch.cat(processed_results["semantic"], dim=0).clone()

            # Define stuff class mask
            mapped_stuff = torch.tensor(self.dm.mapped_stuff_list, device=panoptic_mod.device)
            stuff_mask = torch.isin(semantic, mapped_stuff)

            # Avoid overriding OOD regions (254000 and above are considered OOD labels)
            non_ood_mask = torch.logical_or((panoptic_mod < 254 * 1000), (panoptic_mod >= 255 * 1000))

            # Combine the masks
            safe_mask = stuff_mask & non_ood_mask

            panoptic_mod[safe_mask] = semantic[safe_mask] * self.dm.label_divisor
            processed_results["panoptic"] = panoptic_mod

        for k, v in processed_results.items():
            if k != "segment_info" and not isinstance(v, torch.Tensor):
                    processed_results[k] = torch.cat(v, dim=0)

        return processed_results

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        num_classes = self.sem_seg_head.num_classes
        scores = self.sem_seg_head.predictor.class_embed.probabilities(mask_cls)[
            ..., :num_classes
        ]
        labels = (
            torch.arange(self.sem_seg_head.num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if False and self.panoptic_on:  # i want complete instance prediction
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = (
                    # lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    lab
                    in self.dm.train_id_thing_list
                )

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)
        ).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

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

    def training_step(batch, out, batch_idx):
        pass

    def collect_set_data(self, batch, out, postprocess, dm):
        # used in Prototypicall Deeplab for returning mean embeddings per class, i.e providing additional validition and interpretation
        return None
