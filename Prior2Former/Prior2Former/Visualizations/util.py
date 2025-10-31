import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torchvision
from Data_Loaders.lightning_data_modules.transforms.panoptic import (
    remap_instance,
)
from torch.utils.dlpack import from_dlpack, to_dlpack
from torchvision.transforms import ToTensor


from Visualizations.segmentation import segmentation_to_img


def plot_confusion_matrix(confusion_matrix, class_names):
    import seaborn as sn

    num_classes = confusion_matrix.shape[0]
    df_cm = pd.DataFrame(confusion_matrix, range(num_classes), range(num_classes))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(
        df_cm,
        annot=False,
        annot_kws={"size": 16},
        ax=ax,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
    )  # font size
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.cla()
    plt.close()

    return image


def plot_segmentation_prediction(x, y, preds, datamodule):
    imgs = x.cpu() * torch.tensor(datamodule.std).reshape((3, 1, 1)) + torch.tensor(
        datamodule.mean
    ).reshape((3, 1, 1))
    gt_img = segmentation_to_img(
        y.cpu(), datamodule.num_classes, datamodule.get_class_colors()
    )
    pred_img = segmentation_to_img(
        preds, datamodule.num_classes, datamodule.get_class_colors()
    )

    b = imgs.shape[0]
    h = imgs.shape[2]
    w = imgs.shape[3]
    img = torch.zeros((b, 3, 3, h, w))
    img[:, 0] = imgs
    img[:, 1] = gt_img
    img[:, 2] = pred_img

    grid = torchvision.utils.make_grid(img.reshape((b * 3, 3, h, w)), nrow=3)
    return grid


def tsne_embedding(embedding, n_components=2, **kwargs):
    batch_size, height, width = (
        embedding.shape[0],
        embedding.shape[2],
        embedding.shape[3],
    )
    tsne_embedding = torch.zeros(
        (
            batch_size,
            3,
            height,
            width,
        )
    )

    # cudf = get_cudf()
    # cuml = get_cuml()
    try:
        import cuml
        import cudf
    except:
        return torch.zeros(3, height, width)

    for i in range(batch_size):
        data = embedding[i].detach()
        data = data.permute((1, 2, 0)).reshape((-1, data.shape[0]))
        df = cudf.from_dlpack(to_dlpack(data))

        tsne = cuml.TSNE(n_components=n_components, n_iter=2000, **kwargs)
        embedded = tsne.fit_transform(df)

        embedded = from_dlpack(embedded.to_dlpack()).contiguous()

        embedded = embedded.reshape(height, width, n_components).permute((2, 0, 1))
        max_e = embedded.max(dim=1).values.max(dim=1).values.unsqueeze(1).unsqueeze(1)
        min_e = embedded.min(dim=1).values.min(dim=1).values.unsqueeze(1).unsqueeze(1)

        tsne_embedding[i, 0:n_components] = (embedded - min_e) / (max_e - min_e)
    return tsne_embedding


def plot_panoptic_prediction(
    x,
    y_sem,
    y_instance,
    segmentation,
    panoptic,
    centers=None,
    offset=None,
    datamodule=None,
    extra_imgs=[],
    split_rows=None,
    ignore_instance=None,  #
    return_list=False,
):
    imgs = x.cpu() * torch.tensor(datamodule.std).reshape((3, 1, 1)) + torch.tensor(
        datamodule.mean
    ).reshape((3, 1, 1))
    pred_img = segmentation_to_img(
        segmentation, datamodule.num_classes, datamodule.get_class_colors()
    )

    b = imgs.shape[0]
    h = imgs.shape[2]
    w = imgs.shape[3]

    panoptic_img = torch.zeros((b, 3, h, w))

    for i in range(b):
        panoptic_img[i] = plot_panoptic_img(panoptic[i], datamodule)

    panoptic_gt_img = torch.zeros((b, 3, h, w))
    for i in range(b):
        pano = y_sem[i] * datamodule.label_divisor + remap_instance(
            y_instance[i], ignore_instance
        )
        panoptic_gt_img[i] = plot_panoptic_img(pano, datamodule)

    ncol = 4
    if centers is not None and offset is not None:
        ncol = 6

    ncol += len(extra_imgs)

    img = torch.zeros((b, ncol, 3, h, w))
    img[:, 0] = imgs
    img[:, 1] = panoptic_gt_img
    img[:, 2] = pred_img
    img[:, 3] = panoptic_img
    ix = 4
    if centers is not None and offset is not None:
        center_max = centers.max()
        center_min = centers.min()
        vs = ((centers[:, 0] - center_min) / (center_max - center_min)).unsqueeze(1)
        center_img = (1 - vs) * imgs + vs * torch.tensor([1.0, 0, 0]).reshape(
            (1, 3, 1, 1)
        )

        offset = offset / offset.norm(dim=1, keepdim=True)
        offset_img = torch.zeros((b, 3, h, w))
        offset_img[:, 0:2, :, :] = (offset + 1) / 2

        img[:, 4] = center_img
        img[:, 5] = offset_img
        ix += 2

    for i, extra_img in enumerate(extra_imgs):
        img[:, ix + i] = extra_img
    if return_list:
        return img, None, None
    grid_cols = ncol
    if split_rows is not None:
        grid_cols = grid_cols // split_rows
    grid = torchvision.utils.make_grid(img.reshape((b * ncol, 3, h, w)), nrow=grid_cols)
    return grid, panoptic_gt_img, panoptic_img


def plot_panoptic_img(
    panoptic, datamodule, instance_diff=0.1, ood_color=None, ood_dif=None, min_ood=0.5
):
    h = panoptic.shape[0]
    w = panoptic.shape[1]

    colors = datamodule.get_class_colors()

    panoptic_img = torch.zeros((3, h, w))

    clss = panoptic // datamodule.label_divisor
    clss[panoptic < 0] = -1
    instances = panoptic % datamodule.label_divisor

    clss_u = clss.unique().to(int)
    for cl in clss_u:
        if cl.item() != 255:
            unique_instances = instances[clss == cl].unique()
            n_instances = unique_instances.shape[0]
            if cl.item() == 254:
                if ood_color is not None:
                    color = ood_color
                else:
                    color = torch.tensor(
                        [210 / 255, 105 / 255, 30 / 255]
                    )  # orange-brown
                if ood_dif is None:
                    ood_dif = instance_diff
                color_range = min(ood_dif * (n_instances - 1), min_ood)
                print(color_range)
                min_color = color * (1 - color_range / 2)
                max_color = color * (1 - color_range / 2) + torch.tensor([1, 1, 1]) * (
                    color_range / 2
                )
            else:
                if cl.item() >= 0 and cl.item() < len(colors):
                    color = colors[cl.item()]
                elif cl.item() >= len(colors):
                    color = torch.tensor([0.0, 0.0, 0.0])
                else:
                    color = torch.tensor([0.8, 0.8, 0.8])

                color_range = min(instance_diff * (n_instances - 1), 0.5)
                min_color = color * (1 - color_range / 2)
                max_color = color * (1 - color_range / 2) + torch.tensor([1, 1, 1]) * (
                    color_range / 2
                )

            for j, instance in enumerate(unique_instances):
                mask = (clss == cl) & (instances == instance)
                col_for_instance = (
                    min_color + (max_color - min_color) * (j + 1) / n_instances
                )
                panoptic_img[:, mask] = col_for_instance.unsqueeze(1)
    return panoptic_img


def plot_segmentation_errors_uncertainty(x, y, preds, certainties, datamodule):
    imgs = x.cpu() * torch.tensor(datamodule.std).reshape((3, 1, 1)) + torch.tensor(
        datamodule.mean
    ).reshape((3, 1, 1))
    b = imgs.shape[0]
    h = imgs.shape[2]
    w = imgs.shape[3]

    y_cpu = y.cpu()

    wrong = (y_cpu != preds.cpu()) & (y_cpu != 255)
    wrong_img = torch.zeros((3, b, h, w))
    wrong_img[:, wrong] = torch.tensor([1.0, 0, 0]).reshape((3, 1))
    wrong_img = wrong_img.transpose(1, 0)

    certainty_img = torch.zeros((3, b, h, w))
    certainty_img[0] = 1 - certainties
    certainty_img = certainty_img.transpose(1, 0)

    wrong_high_certainty = wrong * certainties
    wc_img = torch.zeros((3, b, h, w))
    wc_img[0] = wrong_high_certainty
    wc_img = wc_img.transpose(1, 0)

    nrow = 4

    img = torch.zeros((b, nrow, 3, h, w))
    img[:, 0] = imgs
    img[:, 1] = wrong_img
    img[:, 2] = certainty_img
    img[:, 3] = wc_img

    grid = torchvision.utils.make_grid(img.reshape((b * nrow, 3, h, w)), nrow=nrow)
    return grid


def plot_reliability_diagram(bin_accuracy, bin_confidence, bin_counts):
    fig, ax_left = plt.subplots(figsize=(10, 10))

    ax_right = ax_left.twinx()

    x = np.linspace(0, 1, 5)

    ax_left.plot(bin_confidence, bin_accuracy, color="red")
    ax_left.plot(x, x, color="green")
    ax_right.plot(bin_confidence, bin_counts, color="black")
    ax_left.set_xlabel("Confidence")
    ax_left.set_ylabel("Accuracy")
    ax_right.set_ylabel("Counts")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.cla()
    plt.close()
    return image


def plot_embedding_scatter_plot(tsne_embed, panoptic_gt_img, sizes=1, zorder=1):
    fig, ax = plt.subplots(figsize=(10, 10))

    tsne_embed = tsne_embed[:2, :, :].reshape((2, -1)).numpy()
    panoptic_gt_img = panoptic_gt_img.reshape((3, -1)).numpy()

    plt.scatter(
        tsne_embed[0], tsne_embed[1], s=sizes, c=panoptic_gt_img.T, zorder=zorder
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.cla()
    plt.close()
    return image


def scatter_plot(points, colors):
    fig, ax = plt.subplots(figsize=(10, 10))

    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.cla()
    plt.close()
    return image


def plot_roc(fpr, tpr):
    fig, ax_left = plt.subplots(figsize=(10, 10))

    x = np.linspace(0, 1, 5)

    ax_left.plot(fpr, tpr, color="red")
    ax_left.plot(x, x, color="green")
    ax_left.set_xlabel("FPR")
    ax_left.set_ylabel("TPR")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.cla()
    plt.close()
    return image


def plot_uiou(uiou):
    fig, ax_left = plt.subplots(figsize=(10, 10))

    thetas = np.linspace(0, 1, uiou.shape[0])

    ax_left.plot(thetas, uiou, color="red")
    ax_left.set_xlabel("theta")
    ax_left.set_ylabel("UIoU")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.cla()
    plt.close()
    return image


def plot_ti_rate(ti_rate):
    fig, ax_left = plt.subplots(figsize=(10, 10))

    thetas = np.linspace(0, 1, ti_rate.shape[0])

    ax_left.plot(thetas, ti_rate, color="red")
    ax_left.set_xlabel("theta")
    ax_left.set_ylabel("UIoU")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.cla()
    plt.close()
    return image
