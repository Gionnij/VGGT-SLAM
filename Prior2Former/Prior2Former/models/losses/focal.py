from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: Optional[float] = 2.0,
        reduction: Optional[str] = "mean",
        ignore_index=255,
        num_classes=21,
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, input: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(input))
            )
        if not len(input.shape) == 4:
            raise ValueError(
                "Invalid input shape, we expect BxNxHxW. Got: {}".format(input.shape)
            )
        if not input.shape[-2:] == y.shape[-2:]:
            raise ValueError(
                "input and target shapes must be the same. Got: {}".format(
                    input.shape, input.shape
                )
            )
        if not input.device == y.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, y.device
                )
            )
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        yc = y.clone()
        yc[yc == 255] = self.num_classes
        yc = F.one_hot(yc, self.num_classes + 1)
        yc = yc[:, :, :, : self.num_classes]
        yc = yc.permute(0, 3, 1, 2)

        # compute the actual focal loss
        weight = torch.pow(1.0 - input_soft, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(yc * focal, dim=1)

        loss = -1
        if self.reduction == "none":
            loss = loss_tmp
        elif self.reduction == "mean":
            loss = torch.mean(loss_tmp)
        elif self.reduction == "sum":
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(
                "Invalid reduction mode: {}".format(self.reduction)
            )
        return loss


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: Optional[float] = 2.0,
    reduction: Optional[str] = "none",
) -> torch.Tensor:
    r"""Function that computes Focal loss.

    See :class:`~torchgeometry.losses.FocalLoss` for details.
    """
    return FocalLoss(alpha, gamma, reduction)(input, target)
