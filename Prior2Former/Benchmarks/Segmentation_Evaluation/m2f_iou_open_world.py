import argparse
import json
import os
import torch
from tqdm import tqdm

from lightning_utils.panoptic import (
    PanopticSegmentationModule,
)
from metrics.iou_confusion import IoUConfusion

torch.set_grad_enabled(False)
RESULTS_FOLDER=""

def main():
    
    #final paper checkpoints
    ckpt_paths_coco = {
        "p2f": os.path.expanduser(
            "~/Experiments/coco/final_checkpoint.ckpt"
        ),
        "u3hs": os.path.expanduser(
            "~/Experiments/coco/u3hs/final_checkpoint.ckpt"
        ),
        "p2f_bdd": os.path.expanduser(
            "~/Experiments/bdd/final_checkpoint.ckpt"
        ),
    
    }
    dataset_name = "coco_val_anomaly"
    for key, ckpt in tqdm(ckpt_paths_coco.items()):

        modul = PanopticSegmentationModule.load_from_checkpoint(ckpt)
        modul = modul.to("cuda")
        modul.eval()
        # modul.dm.data_dir = "/Datasets/BDD100k_Anomaly"
        if key == "u3hs":

            mean = 0  # 0.988146960735321
            var = 1  # 0.0017467098077759147
            modul.model.certainty_stats = True
            modul.model.set_certainty_stats(torch.tensor(mean), torch.tensor(var))

            modul.hparams["post_process"]["unknown_clustering"] = {
                "eps": args.eps,
                "min_samples": 10,
            }
            modul.hparams["post_process"]["only_unknown"] = False
            modul.hparams["post_process"]["certainty_threshold"] = args.t

        if key == "p2f" or key == "p2f_bdd":

            modul.hparams["post_process"][
                "uncertanty_estimator"
            ] = "sm_max_alpha_beta_cls_uncertainty_keep"
            modul.hparams["post_process"]["unknown_clustering"] = {
                "ood_and_only_ood": True,
                # "only_ood": True,
                "eps": args.eps,
                "min_samples": 10,
                "distance_type": "cosine",
                "uncertainty_threshold": args.t,
                "min_counts": 20,
            }
        iou = IoUConfusion(
            num_classes=modul.dm.num_classes + 2,
            ignore_index=modul.dm.num_classes + 1,
            compute_on_step=False,
        )
        modul.dm.no_resize = True
        if dataset_name == "coco_val_anomaly":
            loader = modul.dm.anomaly_dataloader(split="full_val")
        elif dataset_name == "coco_val_anomaly_100":
            loader = modul.dm.anomaly_dataloader()
        elif "bdd" in dataset_name:
            loader = modul.dm.anomaly_dataloader(
                split="anom_val" if "val" in dataset_name else "anom_all"
            )
        print("Dataloader length: ", len(loader))
        for i, batch in tqdm(enumerate(loader)):
            x = modul.dm.get_image_from_batch(batch).cuda()
            out = modul(x)
            modul.dm.base_size_val = (x.shape[-1], x.shape[-2])
            postprocessed = modul.model.postprocess(
                out, modul.dm, modul.hparams["post_process"]
            )
            open_world_pred = postprocessed["semantic"].cpu()
            print(postprocessed["panoptic"].cpu().unique())
            if "ood_mask" in postprocessed:
                open_world_pred[postprocessed["ood_mask"].cpu().unsqueeze(0) != 0] = (
                    modul.dm.num_classes
                )
            else:
                # only for only_ood case
                open_world_pred[postprocessed["panoptic"].cpu() != 0] = (
                    modul.dm.num_classes
                )

            gt_sem = batch["semantic"]
            if "bdd" in dataset_name:
                gt_sem[gt_sem > 38] = 254
            print(
                (open_world_pred == modul.dm.num_classes).sum(), (gt_sem == 254).sum()
            )
            open_world_pred[open_world_pred == 254] = modul.dm.num_classes
            gt_sem[gt_sem == 254] = modul.dm.num_classes
            gt_sem[gt_sem == 255] = modul.dm.num_classes + 1

            iou.update(
                open_world_pred.squeeze(),
                gt_sem.squeeze(),
            )
            del out, postprocessed, x

        iou_res = iou.compute()
        print("this is the iou: ", iou_res)
        break
        with open(f"{RESULTS_FOLDER}/{key}.json", "w") as f:
            json.dump(iou.item(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--t", type=float, default=0.93)
    args = parser.parse_args()
    main()
