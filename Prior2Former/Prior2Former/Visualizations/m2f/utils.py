import torch
import numpy as np


import matplotlib.pyplot as plt


def mask_vis(
    mask,
    keep=None,
    path=None,
    normalize=False,
    vmin=None,
    vmax=None,
    red_channel=False,
    ncols=-1,
):
    """_summary_

    Args:
        mask (_type_): _description_
        keep (_type_, optional): _description_. Defaults to None.
    """
    if keep == None:
        keep = 0
    elif keep.shape[0] != 1:
        assert len(keep.shape) == 1
        keep = keep.unsqueeze(0)

    if mask.shape[0] != 1:
        assert len(mask.shape) == 3
        mask = mask.unsqueeze(0)

    if vmin and vmax:
        pass
    elif not normalize:
        vmin = mask.min() if vmin is None else vmin
        vmax = mask.max() if vmax is None else vmax

    # Number of masks to plot
    num_masks = len(mask[keep])
    h, w = mask.shape[-2:]
    ratio = w // h
    # Determine grid size (e.g., 10x10 for 100 masks)
    if ncols < 0:
        nrows = ncols = int(np.ceil(np.sqrt(num_masks)))
    else:
        nrows = int(np.ceil(num_masks / ncols))

    # Create subplots
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(w // 4, h // 4),
        gridspec_kw={"wspace": 0.02, "hspace": 0.02},
    )

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Plot each mask
    for i, a in enumerate(mask[keep]):
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        if red_channel:
            im = np.zeros((3, h, w))
            im[0] = (a - vmin) / (vmax - vmin) if vmin else a
            axes[i].imshow(im.transpose(1, 2, 0))
        else:
            # axes[i].imshow(a, vmin=vmin, vmax=vmax)
            axes[i].imshow(a)
        axes[i].axis("off")  # Hide axes
    for ax in axes[num_masks:]:
        ax.axis("off")
    # Adjust layout and show the plot
    plt.subplots_adjust(wspace=0, hspace=0)
    if path is None:
        plt.show()
    else:
        if not path.endswith(".png"):
            path = path + ".png"
        plt.savefig(path)
