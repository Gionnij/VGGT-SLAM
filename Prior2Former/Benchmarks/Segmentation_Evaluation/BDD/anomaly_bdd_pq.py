import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

torch.set_grad_enabled(False)
from Benchmarks.Open_World_Benchmark.OoDIS.utils import certainty_stats_U3HS, uncertainty_stats_m2p
from metrics.pq_u3hs import PanopticQuality
from lightning_utils.panoptic import (
    PanopticSegmentationModule,
)
import os


os.environ["DIR_FISHY_LAF"] = "/Datasets/dataset_FishyLAF"
os.environ["DIR_LAF"] = "/Datasets/dataset_LostAndFound"
os.environ["DIR_DATASETS"] = "/Datasets"
torch.set_grad_enabled(False)
import argparse


def compute_iou(array1, array2):
    """
    Compute the Intersection over Union (IoU) of two binary arrays.

    Returns:
    float: IoU value.
    """
    # Ensure the arrays are binary
    if array1.sum() == 0:
        return 0.0
    array1 = array1.to(bool)
    array2 = array2.to(bool)

    # Compute intersection and union
    intersection = torch.logical_and(array1, array2).sum()
    union = torch.logical_or(array1, array2).sum()

    # Compute IoU
    iou = intersection / union if union != 0 else 0.0

    return iou.cpu()


#both are the final checkpoints for the paper
checkpoint_dict = {
    "p2f": os.path.expanduser(
        "~/Experiments/bdd/p2f/final_checkpoint.ckpt"
    ),
    "m2f": os.path.expanduser(
        "~/Experiments/bdd/m2f/final_checkpoint.ckpt"
    ),
    "u3hs": os.path.expanduser(
        "~/Experiments/bdd/u3hs/final_checkpoint.ckpt"
    ),
}


def pq(th=-0.5, e=0.04, args={},key="p2f"):
    modul = PanopticSegmentationModule.load_from_checkpoint(checkpoint_dict[key])
    modul = modul.to("cuda")
    modul.eval()
    modul.hparams["post_process"][
        "uncertanty_estimator"
    ] = "max_alpha_beta_cls_uncertainty_keep"
    modul.dm.base_size_val = (1280, 720)
    if args.computeCertaintyStats != 0:
        mean, var = uncertainty_stats_m2p(modul, iters=args.computeCertaintyStats)
    else:
        mean = 0
        var = 1
    print(mean, var)
    modul.hparams["post_process"]["unknown_clustering"] = {
        "only_ood": False,
        "eps": args.e,
        "min_samples": 17,
        "distance_type": "cosine",
        "uncertainty_threshold": mean + args.th * torch.tensor(var).sqrt().item(),
    }

    loader = modul.dm.anomaly_dataloader(debug=True, split=args.split)
    print("Dataloader length: ", len(loader))
    pq = PanopticQuality(
        num_classes=modul.dm.num_classes + 1, per_class=True, label_divisor=1000
    )
    pq.reset()
    count = 0
    for l, batch in tqdm(enumerate(loader)):
        ood = batch["semantic"] > 38
        batch["semantic"][batch["semantic"] > 38] = modul.dm.num_classes
        gt_pan = batch["semantic"] * modul.dm.label_divisor + batch["instance"]
        unique_ood, counts = gt_pan[ood].unique(return_counts=True)
        keep = False
        for id, c in zip(unique_ood, counts):
            if c < args.min_area:
                gt_pan[gt_pan == id] = 255000
            else:
                keep = True  # at leas one image fullfills requirement
        if not keep:
            print("number skipped: ", l - count)
            continue
        count += 1
        image = batch["image"].cuda()
        out = modul(image)
        postprocess = modul.model.postprocess(
            out, modul.dm, modul.hparams["post_process"]
        )
        uncertainty = postprocess["uncertainty"]
        panoptic = postprocess["panoptic"]  # -1 dbscan outlier

        pred_class = panoptic // modul.dm.label_divisor
        pred_instance = panoptic % modul.dm.label_divisor

        pred_class[pred_class == 254] = modul.dm.num_classes
        panoptic = pred_class * modul.dm.label_divisor + pred_instance

        pq.update(panoptic.cpu(), gt_pan)
        if args.verbose:
            os.makedirs("../../plots", exist_ok=True)
            grid, pgt = modul.model.plot_prediction(
                batch["image"],
                batch,
                postprocess,
                out,
                modul.dm,
                None,
                0,
                0,
                log_folder=None,
            )
            Image.fromarray(
                (grid * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            ).save(
                os.path.join(
                    f"plots/{batch['file_name'][0]}_grid.png",
                )
            )
            plot(
                [
                    batch["original"].squeeze(),
                    uncertainty.cpu().squeeze(),
                    ood.squeeze(),
                ],
                f"plots/{batch['file_name'][0]}_uncertainty.png",
            )
        del image, out, postprocess, panoptic, pred_class, pred_instance, gt_pan

    r = pq.compute()
    print(r)
    r = torch.tensor(r).reshape(modul.dm.num_classes + 2, 6)
    pq.reset()
    print("overall: ", r[0])
    print("OOD: ", r[-1])
    print("Evaluated on number of images: ", count)
    return r[0], r[-1]


def pq_m2f(th=-0.5, e=0.04, args={},key="m2f"):
    modul = PanopticSegmentationModule.load_from_checkpoint(checkpoint_dict[key])
    modul = modul.to("cuda")
    modul.eval()
    modul.hparams["post_process"][
        "uncertanty_estimator"
    ] = "m2f_temperature_uncertainty"
    modul.dm.base_size_val = (1280, 720)
    if args.computeCertaintyStats != 0:
        mean, var = uncertainty_stats_m2p(modul, iters=args.computeCertaintyStats)
    else:
        mean = 0
        var = 1
    print(mean, var)
    modul.hparams["post_process"]["unknown_clustering"] = {
        "only_ood": False,
        "eps": args.e,
        "min_samples": 17,
        "distance_type": "cosine",
        "uncertainty_threshold": mean + args.th * torch.tensor(var).sqrt().item(),
    }

    loader = modul.dm.anomaly_dataloader(debug=True, split=args.split)
    print("Dataloader length: ", len(loader))
    batch = next(iter(loader))
    pq = PanopticQuality(
        num_classes=modul.dm.num_classes + 1, per_class=True, label_divisor=1000
    )
    pq.reset()
    count = 0
    for l, batch in tqdm(enumerate(loader)):
        ood = batch["semantic"] > 38
        batch["semantic"][batch["semantic"] > 38] = modul.dm.num_classes
        gt_pan = batch["semantic"] * modul.dm.label_divisor + batch["instance"]
        unique_ood, counts = gt_pan[ood].unique(return_counts=True)
        keep = False
        for id, c in zip(unique_ood, counts):
            if c < args.min_area:
                gt_pan[gt_pan == id] = 255000
            else:
                keep = True  # at leas one image fullfills requirement
        if not keep:
            print("number skipped: ", l - count)
            continue
        count += 1
        image = batch["image"].cuda()
        out = modul(image)
        postprocess = modul.model.postprocess(
            out, modul.dm, modul.hparams["post_process"]
        )
        uncertainty = postprocess["uncertainty"]
        panoptic = postprocess["panoptic"]  # -1 dbscan outlier

        pred_class = panoptic // modul.dm.label_divisor
        pred_instance = panoptic % modul.dm.label_divisor

        pred_class[pred_class == 254] = modul.dm.num_classes
        panoptic = pred_class * modul.dm.label_divisor + pred_instance

        pq.update(panoptic.cpu(), gt_pan)
        if args.verbose:
            os.makedirs("../../plots", exist_ok=True)
            grid, pgt = modul.model.plot_prediction(
                batch["image"],
                batch,
                postprocess,
                out,
                modul.dm,
                None,
                0,
                0,
                log_folder=None,
            )
            Image.fromarray(
                (grid * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            ).save(
                os.path.join(
                    f"plots/{batch['file_name'][0]}_grid.png",
                )
            )
            plot(
                [
                    batch["original"].squeeze(),
                    uncertainty.cpu().squeeze(),
                    ood.squeeze(),
                ],
                f"plots/{batch['file_name'][0]}_uncertainty.png",
            )
        del image, out, postprocess, panoptic, pred_class, pred_instance, gt_pan

    r = pq.compute()
    print(r)
    r = torch.tensor(r).reshape(modul.dm.num_classes + 2, 6)
    pq.reset()
    print("overall: ", r[0])
    print("OOD: ", r[-1])
    print("Evaluated on number of images: ", count)
    return r[0], r[-1]

def pq_U3HS(th=3, e=0.5, args={}):
    modul = PanopticSegmentationModule.load_from_checkpoint(checkpoint_dict["u3hs"])
    modul = modul.to("cuda")
    modul.eval()
    modul.dm.data_dir = "/Datasets/BDD100k_Anomaly"
    if args.computeCertaintyStats != 0:
        mean, var = certainty_stats_U3HS(modul, iters=args.computeCertaintyStats)
    else:
        mean = 0.9661463499069214
        var = 0.009469615295529366
    print(mean, var)
    modul.model.set_certainty_stats(torch.tensor(mean), torch.tensor(var))
    modul.hparams["post_process"]["unknown_clustering"] = {
        "only_ood": False,
        "epsilon": e,
        "min_samples": 15,
    }
    modul.hparams["post_process"]["certainty_threshold"] = -args.th

    # Mean: 0.9489491581916809, Variance: 0.001957752974703908
    # mean, var = certainty_stats_U3HS(modul, iters=100)

    loader = modul.dm.anomaly_dataloader(debug=True, split=args.split)

    pq = PanopticQuality(num_classes=modul.dm.num_classes + 1, per_class=True)
    pq.reset()
    count = 0
    for l, batch in tqdm(enumerate(loader)):
        ood = batch["semantic"] > 38
        batch["semantic"][batch["semantic"] > 38] = modul.dm.num_classes
        gt_pan = batch["semantic"] * modul.dm.label_divisor + batch["instance"]
        unique_ood, counts = gt_pan[ood].unique(return_counts=True)
        keep = False
        for id, c in zip(unique_ood, counts):
            if c < args.min_area:
                gt_pan[gt_pan == id] = 255000
            else:
                keep = True  # at leas one image fullfills requirement
        if not keep:
            print("number skipped: ", l - count)
            continue
        count += 1
        image = batch["image"].cuda()
        out = modul(image)
        postprocess = modul.model.postprocess(
            out, modul.dm, modul.hparams["post_process"]
        )
        uncertainty = postprocess["uncertainty"]
        panoptic = postprocess["panoptic"]  # -1 dbscan outlier

        pred_class = panoptic // modul.dm.label_divisor
        pred_instance = panoptic % modul.dm.label_divisor
        print((pred_class == 254).sum())
        pred_class[pred_class == 254] = modul.dm.num_classes
        panoptic = pred_class * modul.dm.label_divisor + pred_instance

        instance = batch[
            "instance"
        ]  # everything is class zero, matched are the instances only
        batch["semantic"][batch["semantic"] > 38] = modul.dm.num_classes
        gt_pan = batch["semantic"] * modul.dm.label_divisor + instance
        # instance[known_mask] = 255
        # gt_pan[batch["semantic"] == 254] = 254000
        pq.update(panoptic.cpu(), gt_pan)
        if args.verbose:
            os.makedirs("plots_U3HS", exist_ok=True)
            grid, pgt = modul.model.plot_prediction(
                batch["image"],
                batch,
                postprocess,
                out,
                modul.dm,
                None,
                0,
                0,
                log_folder=None,
            )
            Image.fromarray(
                (grid * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            ).save(
                os.path.join(
                    f"plots_U3HS/{batch['file_name'][0]}_grid.png",
                )
            )
            plot(
                [
                    batch["original"].squeeze(),
                    uncertainty.cpu().squeeze(),
                    ood.squeeze(),
                ],
                f"plots_U3HS/{batch['file_name'][0]}_uncertainty.png",
            )
        del image, out, postprocess, panoptic, pred_class, pred_instance, gt_pan

    r = pq.compute()
    r = torch.tensor(r).reshape(modul.dm.num_classes + 2, 6)
    pq.reset()
    print("overall: ", r[0])
    print("OOD: ", r[-1])
    print("Evaluated on number of images: ", count)
    return r[0], r[-1]


def plot(img_list: list, store_path: str):
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    fig, axs = plt.subplots(
        1,
        len(img_list),
        figsize=(9, 3),
        gridspec_kw={"wspace": 0.02, "hspace": 0.02},
    )
    for i, img in enumerate(img_list):
        axs[i].imshow(img)
        axs[i].axis("off")
    plt.savefig(store_path)
    plt.close()


def iou(ts=[0.1, 0.3, 0.4, 0.6, 0.8, 0.9], args={}):
    modul = PanopticSegmentationModule.load_from_checkpoint(checkpoint_dict["u3hs"])
    modul = modul.to("cuda")
    modul.eval()
    modul.dm.base_size_val = (1280, 720)
    modul.dm.data_dir = "/Datasets/BDD100k_Anomaly"
    # modul.hparams["post_process"][
    #     "uncertanty_estimator"
    # ] = "sm_max_alpha_beta_cls_uncertainty_keep"
    modul.hparams["post_process"]["certainty_threshold"] = args.th
    ious = {t: [] for t in ts}
    loader = modul.dm.anomaly_dataloader(debug=True, split=args.split)
    batch = next(iter(loader))
    print("ma: ", args.min_area)
    count = 0
    for l, batch in tqdm(enumerate(loader)):
        ood = batch["semantic"] > 38
        if ood.sum() < args.min_area:
            print(ood.sum())
            continue
        count += 1
        image = batch["image"].cuda()
        out = modul(image)
        postprocess = modul.model.postprocess(
            out, modul.dm, modul.hparams["post_process"]
        )

        uncertainty = postprocess["uncertainty"]
        # make a grid plot with
        if args.verbose:
            plot(
                [
                    batch["original"].squeeze(),
                    uncertainty.cpu().squeeze(),
                    ood.squeeze(),
                ],
                f"plots/{l}.png",
            )
        for t in ts:
            ious[t].append(compute_iou(ood, uncertainty.cpu() > t))
    print(ious)
    print("evaluated on number of images: ", count)
    for t in ts:
        ious[t] = torch.stack(ious[t]).mean()
    return ious

def main():
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose mode"
    )
    parser.add_argument(
        "--function",
        "-f",
        type=str,
        default="pq",
        help="Specify the function to execute",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="anom_val",
        help="Specify the split to use for the function",
    )
    parser.add_argument(
        "--min_area",
        "-ma",
        type=int,
        default=2500,
        help="Specify the minimum area for the oods",
    )
    parser.add_argument(
        "--th", "-t", type=float, default=-0.8, help="Specify the threshold"
    )
    parser.add_argument(
        "--e", "-e", type=float, default=0.04, help="Specify the epsilon"
    )
    parser.add_argument("--key", "-k", type=str, default="p2f")
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--computeCertaintyStats", "-c", type=int, default=0)
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode enabled")

    if args.function == "iou":
        print(iou(args=args))
    elif args.function == "pq":
        pq(th=args.th, e=args.e, args=args,key=args.key)
    elif args.function == "pq_m2f":
        pq_m2f(th=args.th, e=args.e, args=args,key=args.key)
    elif args.function == "pq_U3HS":
        pq_U3HS(args=args)
    else:
        print("Invalid function specified")


if __name__ == "__main__":
    main()
