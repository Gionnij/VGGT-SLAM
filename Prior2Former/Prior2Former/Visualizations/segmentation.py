import colorsys

import torch


def get_color_range(n: int):
    HSV_tuples = [(x * 1.0 / n, 1.0, 1.0) for x in range(n)]
    RGB_tuples = [torch.tensor(colorsys.hsv_to_rgb(*x)) for x in HSV_tuples]
    return RGB_tuples


def segmentation_to_img(segmentation: torch.Tensor, num_classes: int, colors=None):
    if colors is None:
        colors = get_color_range(num_classes)

    batches = segmentation.shape[0]

    img = torch.zeros((batches, 3, segmentation.shape[1], segmentation.shape[2]))
    for b in range(batches):
        for i in range(num_classes):
            img[b, :, segmentation[b] == i] = colors[i].reshape((3, 1))
    return img
