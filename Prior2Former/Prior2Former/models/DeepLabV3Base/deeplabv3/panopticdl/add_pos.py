import torch
import torch.nn as nn


class AddPosition(nn.Module):
    def __init__(self):
        super(AddPosition, self).__init__()

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]

        xs = (
            torch.arange(0, 1.000000001, 1 / (width - 1), device=x.device)
            .reshape((1, 1, 1, width))
            .expand((x.shape[0], 1, height, width))
        )
        ys = (
            torch.arange(0, 1.000000001, 1 / (height - 1), device=x.device)
            .reshape((1, 1, height, 1))
            .expand((x.shape[0], 1, height, width))
        )

        return torch.cat((x, xs, ys), dim=1)
