from typing import Optional, Tuple

import torch

try:
    # old API (<=0.11)
    from torchmetrics.classification.iou import IoU
except Exception:
    # new API (>=1.0)
    from torchmetrics.classification import MulticlassJaccardIndex as IoU


from torchmetrics.functional.classification.confusion_matrix import (
    _confusion_matrix_compute,
)

# from torchmetrics.functional.classification.iou import _iou_from_confmat
from torchmetrics.utilities.distributed import reduce


def _iou_from_confmat(
    confmat: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
    reduction: str = "elementwise_mean",
) -> torch.Tensor:
    """Computes the intersection over union from confusion matrix.

    Args:
        confmat: Confusion matrix without normalization
        num_classes: Number of classes for a given prediction and target tensor
        ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method.
        absent_score: score to use for an individual class, if no instances of the class index were present in `pred`
            AND no instances of the class index were present in `target`.
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied
    """

    # Remove the ignored class index from the scores.
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        confmat[ignore_index] = 0.0

    intersection = torch.diag(confmat)
    union = confmat.sum(0) + confmat.sum(1) - intersection

    # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
    scores = intersection.float() / union.float()
    scores[union == 0] = absent_score

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        scores = torch.cat(
            [
                scores[:ignore_index],
                scores[ignore_index + 1 :],
            ]
        )

    return reduce(scores, reduction=reduction)


class IoUConfusion(IoU):
    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes intersection over union (IoU)
        """
        confmat = self.confmat

        if self.ignore_index is not None:
            # Ignore index here means that every area in the target
            # where the value == self.ignore_index will be completely excluded
            # from evaluation. This means especially in IoU those areas should not
            # be included in the union -> we have to remove it from the confidence matrix
            confmat = torch.cat(
                [
                    confmat[: self.ignore_index],
                    confmat[self.ignore_index + 1 :],
                ]
            )
            confmat = torch.cat(
                [
                    confmat[:, : self.ignore_index],
                    confmat[:, self.ignore_index + 1 :],
                ],
                dim=1,
            )

        confusion_matrix = _confusion_matrix_compute(confmat, "true")
        iou = _iou_from_confmat(
            confmat,
            self.num_classes,
            None,
            self.absent_score,
            self.reduction,
        )
        return iou, confusion_matrix