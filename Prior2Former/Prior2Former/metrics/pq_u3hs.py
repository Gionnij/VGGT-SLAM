from distutils.version import LooseVersion
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import AverageMethod, DataType


def _auroc_update(
    preds: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, DataType]:
    """Updates and returns variables required to compute Area Under the Receiver Operating Characteristic Curve.
    Validates the inputs and returns the mode of the inputs.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    # use _input_format_classification for validating the input and get the mode of data
    _, _, mode = _input_format_classification(preds, target)

    if mode == "multi class multi dim":
        n_classes = preds.shape[1]
        preds = preds.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)
        target = target.flatten()
    if mode == "multi-label" and preds.ndim > 2:
        n_classes = preds.shape[1]
        preds = preds.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)
        target = target.transpose(0, 1).reshape(n_classes, -1).transpose(0, 1)

    return preds, target, mode


class PanopticQuality(Metric):
    def __init__(
        self,
        num_classes: Optional[int] = None,
        label_divisor: int = 1000,
        min_match_iou: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        per_class: bool = False,
        dist_sync_fn: Callable = None,
        instance_threshold=None,
    ):
        super().__init__(
            # compute_on_step=compute_on_step, #take out because of dependency issues with torchmetrics
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.num_classes = num_classes
        self.label_divisor = label_divisor
        self.min_match_iou = min_match_iou

        self.per_class = per_class and num_classes is not None
        self.state_dim = (
            num_classes if self.per_class and num_classes is not None else 1
        )

        self.instance_threshold = instance_threshold

        self.add_state("iou", default=torch.zeros(self.state_dim), dist_reduce_fx="sum")
        self.add_state("tp", default=torch.zeros(self.state_dim), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(self.state_dim), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(self.state_dim), dist_reduce_fx="sum")

    def reset(self):
        super().reset()

    def update(self, panoptic_preds: torch.Tensor, panoptic_gt: torch.Tensor):
        gt_clss_ids = torch.div(panoptic_gt, self.label_divisor, rounding_mode="trunc")
        gt_unique = panoptic_gt.unique()

        if self.instance_threshold is not None:
            equal = gt_unique.unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ) == panoptic_gt.unsqueeze(0)
            count = equal.sum(dim=(1, 2, 3))

            lt_ids = gt_unique[count < self.instance_threshold]
            mask = (panoptic_gt.unsqueeze(0) == lt_ids.unsqueeze(1).unsqueeze(1)).any(
                dim=0
            )
            panoptic_gt[mask] = 255 * self.label_divisor

            gt_clss_ids = panoptic_gt // self.label_divisor
            gt_unique = panoptic_gt.unique()

        panoptic_preds = panoptic_preds.clone()
        panoptic_preds[gt_clss_ids == 255] = 255 * self.label_divisor

        pred_unique = panoptic_preds.unique()

        pred_set = set([p.item() for p in pred_unique])
        if (255 * self.label_divisor) in pred_set:
            pred_set.remove(255 * self.label_divisor)
        gt_set = set([p.item() for p in gt_unique])

        ious = [0] * self.state_dim
        tp = [0] * self.state_dim

        for gt_id in gt_unique:
            gt_id = gt_id.item()
            clss_id = gt_id // self.label_divisor

            if clss_id != 255:
                for pred_id in pred_set:
                    pred_clss_id = pred_id // self.label_divisor

                    if clss_id == pred_clss_id:
                        intersection = (
                            (panoptic_preds == pred_id) & (panoptic_gt == gt_id)
                        ).sum()
                        union = (
                            (panoptic_preds == pred_id) | (panoptic_gt == gt_id)
                        ).sum()

                        iou = intersection / union
                        if iou >= self.min_match_iou:
                            # Match!
                            index = 0 if self.state_dim == 1 else clss_id

                            ious[index] += iou
                            tp[index] += 1
                            pred_set.remove(pred_id)
                            gt_set.remove(gt_id)
                            break
            else:
                gt_set.remove(gt_id)

        self.iou += torch.tensor(ious).to(self.iou.device)
        self.tp += torch.tensor(tp).to(self.tp.device)

        for item in pred_set:
            clss_id = item // self.label_divisor

            index = 0 if self.state_dim == 1 else clss_id
            self.fp[index] += 1

        for item in gt_set:
            clss_id = item // self.label_divisor

            index = 0 if self.state_dim == 1 else clss_id
            self.fn[index] += 1

    def compute(self):
        if self.state_dim > 1 and self.num_classes is not None:
            pq = self.iou.sum() / (self.tp.sum() + (self.fp.sum() + self.fn.sum()) / 2)
            sq = self.iou.sum() / self.tp.sum()
            rq = self.tp.sum() / (self.tp.sum() + (self.fp.sum() + self.fn.sum()) / 2)

            result = [pq, sq, rq, self.tp.sum(), self.fp.sum(), self.fn.sum()]

            for i in range(self.num_classes):
                pq = self.iou[i] / (self.tp[i] + (self.fp[i] + self.fn[i]) / 2)
                sq = self.iou[i] / self.tp[i]
                rq = self.tp[i] / (self.tp[i] + (self.fp[i] + self.fn[i]) / 2)

                result.append(pq)
                result.append(sq)
                result.append(rq)
                result.append(self.tp[i])
                result.append(self.fp[i])
                result.append(self.fn[i])

            return tuple(result)
        else:
            pq = self.iou / (self.tp + (self.fp + self.fn) / 2)
            sq = self.iou / self.tp
            rq = self.tp / (self.tp + (self.fp + self.fn) / 2)

            return pq, sq, rq, self.tp, self.fp, self.fn

    def get_return_names(self):
        if self.state_dim == 1 or self.num_classes is None:
            return ("pq", "sq", "rq", "tp", "fp", "fn")
        else:
            result = ["pq", "sq", "rq", "tp", "fp", "fn"]
            for i in range(self.num_classes):
                result.extend(
                    [f"pq_{i}", f"sq_{i}", f"rq_{i}", f"tp_{i}", f"fp_{i}", f"fn_{i}"]
                )
            return tuple(result)


class PanopticQualityTest(PanopticQuality):
    def update(self, batch, out, postprocessed):
        super().update(postprocessed["panoptic"], batch["panoptic_id"])


class PanopticQualityUnknown(PanopticQuality):
    def __init__(
        self,
        only_unknown=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.only_unknown = only_unknown

    def update(self, batch, out, postprocessed):
        if not self.only_unknown:
            unknown_segmentation = torch.zeros_like(postprocessed["panoptic"])

            unknown_mask = postprocessed["panoptic"] < 0
            unknown_segmentation[~unknown_mask] = 1
            # Unknown Panoptic ids are -1000 + instance_id and
            # have to be converted to 1000 + instance_id
            unknown_segmentation[unknown_mask] = (
                postprocessed["panoptic"][unknown_mask] + 2001
            )

            super().update(unknown_segmentation, batch["panoptic_id"])
        else:
            super().update(postprocessed["panoptic"], batch["panoptic_id"])
