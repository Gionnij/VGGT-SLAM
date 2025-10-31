from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, ignore_index=255, num_classes=21) -> None:
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.eps: float = 1e-6

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
        input_soft = F.softmax(input, dim=1)

        yc = y.clone()
        yc[yc == 255] = self.num_classes
        yc = F.one_hot(yc, self.num_classes + 1)
        yc = yc[:, :, :, : self.num_classes]
        yc = yc.permute(0, 3, 1, 2)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * yc, dims)
        cardinality = torch.sum(input_soft + yc, dims)

        dice_score = 2.0 * intersection / (cardinality + self.eps)
        return torch.mean(1.0 - dice_score)


def dice_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss()(input, target)
