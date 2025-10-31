# %%
from argparse import ArgumentParser
import json
import torch
import numpy as np
from pathlib import Path

torch.set_grad_enabled(False)

from lightning_utils.panoptic import (
    PanopticSegmentationModule,
)
from detectron2.evaluation import (
    inference_on_dataset,
)
from detectron2.utils.logger import setup_logger


import os
from metrics.coco_panoptic.panoptic_evaluation import (
    COCOPanopticEvaluator,
)

os.makedirs("__pycache__", exist_ok=True)


def generate_latex_table(data, output_file="output.tex"):
    """
    Generates a LaTeX table from a list of dictionaries.

    Args:
        data (list): A list of dictionaries, where each dictionary represents a column in the table.
        output_file (str, optional): The name of the output file. Defaults to 'output.tex'.
    """
    # Get the keys from the first dictionary in the list
    keys = list(data[0]["panoptic_seg"].keys())

    # Create the LaTeX table
    table = r"\begin{table}[h!]" + "\n"
    table += r"\centering" + "\n"
    table += r"\begin{tabular}{lc}" + "\n"
    table += "& Placeholder 0 \\\\ \n"
    table += r"\hline" + "\n"

    for key in keys:
        row = f"{key} & "
        for d in data:
            d = d["panoptic_seg"]
            try:
                row += f"{d[key]:.3f} & "
            except (ValueError, TypeError):
                row += f"{d[key]} & "
        table += row[:-3] + "\\\\ \n"

    table += r"\hline" + "\n"
    table += r"\end{tabular}" + "\n"
    table += r"\caption{PQ Metrics}" + "\n"
    table += r"\end{table}"

    # Write the table to a file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(table)

    print(f"LaTeX table written to {output_file}")


class ModulWrapper(torch.nn.Module):

    def __init__(self, modul, _thing_contiguous_id_to_dataset_id):
        super().__init__()
        self.modul = modul
        self._thing_contiguous_id_to_dataset_id = _thing_contiguous_id_to_dataset_id
        if hasattr(self.modul.model, "certainty_stats"):
            self.modul.model.certainty_stats = True

    def forward(self, inputs):
        # returns the correct output needed for the evaluation and changes the input inplace
        x = self.modul.dm.get_image_from_batch(inputs).to(self.modul.device)
        out = self.modul(x)
        modul.dm.base_size_val = x.shape[-1], x.shape[-2]
        postprocess = self.modul.model.postprocess(
            out, self.modul.dm, self.modul.hparams["post_process"]
        )
        res = {"panoptic_seg": []}
        inputs["file_name"] = []
        inputs["image_id"] = []
        for file_name in inputs["info"]:
            inputs["file_name"].append(file_name)
            inputs["image_id"].append(int(file_name.replace(".jpg", "")))

        segments_info = []
        panoptic = postprocess["panoptic"].squeeze().cpu() + 1
        for id in np.unique(panoptic):

            pred_class = id // self.modul.dm.label_divisor
            if pred_class == 255:
                continue
            if pred_class == -1:
                continue
            if pred_class == 253:
                continue
            isthing = pred_class in self._thing_contiguous_id_to_dataset_id
            # merge stuff regions

            segments_info.append(
                {
                    "id": int(id),
                    "isthing": bool(isthing),
                    "category_id": int(pred_class),
                }
            )
        mask = panoptic // self.modul.dm.label_divisor == 255
        panoptic[mask] = 0
        res["panoptic_seg"].append(
            (
                panoptic,
                segments_info,
            )
        )
        return res

if __name__ == "__main__":

    ckpt_paths_coco = {
        #final paper coco open-world
        "u3hs": os.path.expanduser(
            "~/Experiments/Coco/u3hs/final_checkpoint.ckpt"
        ),

        #final paper Coco open-world
        "p2f": os.path.expanduser(
            "~/Experiments/Coco/p2f/final_checkpoint.ckpt"
        ),
        "m2f": os.path.expanduser(
            "~/Experiments/Coco/m2f/final_checkpoint.ckpt"
        ),
        "m2a":
         os.path.expanduser(
            "~/Experiments/Coco/m2a/final_checkpoint.ckpt"
        )
    }


    parser = ArgumentParser()
    parser.add_argument("--key", "-k", type=str, default="p2f")
    parser.add_argument("--dataset_name", "-d", type=str, default="coco_val_anomaly_100")
    parser.add_argument("--eps","-e", type=float, default=0.005)
    parser.add_argument("--threshold","-t", type=float, default=-0.3)
    parser.add_argument("--resize","-r", type=int, default=0)
    args = parser.parse_args()
    key = args.key
    resize = args.resize

    dataset_name = args.dataset_name

    # modul = OldPanopticSegmentationModule.load_from_checkpoint(ckpt_paths_coco[key])
    modul = PanopticSegmentationModule.load_from_checkpoint(ckpt_paths_coco[key], datamodule={"args": {"root": "/root/Datasets/"}})
    
    modul.to("cuda")
    modul.eval()
    if "anomaly" in args.dataset_name and "InD" not in args.dataset_name:
        if key in ["p2f"]:
            modul.hparams["post_process"]["unknown_clustering"] = {
                "only_ood": False,
                "eps": 0.12,# 0.04,
                "min_samples": 500, #17,
                "distance_type": "cosine", #"cosine",
                "uncertainty_threshold": -0.5,# -1 # -0.5 # -0.3,
                "min_counts": 0,

            }
            modul.hparams["post_process"][
                "uncertanty_estimator"
            ] = "max_alpha_beta_cls_uncertainty_keep"
        elif key in ["m2a"]:
            modul.hparams["post_process"]["unknown_clustering"] = {
                "only_ood": False,
                "eps": 0.12,# 0.04,
                "min_samples": 500, #17,
                "distance_type": "cosine", #"cosine",
                "uncertainty_threshold": 0.9,# -1 # -0.5 # -0.3,
                "min_counts": 0,

            }
            modul.hparams["post_process"][
                "uncertanty_estimator"
            ] = "mask2anomaly"

            modul.hparams["post_process"][
                "uncertanty_estimator"
            ] = "max_alpha_beta_cls_uncertainty_keep"
        elif key == ["m2f"]:
            modul.hparams["post_process"]["unknown_clustering"] = {
                "only_ood": False,
                "eps": 0.005,
                "min_samples": 17,
                "distance_type": "cosine",
                "uncertainty_threshold": -0.5,
            }

            modul.hparams["post_process"][
                "uncertanty_estimator"
            ] = "m2f_temperature_uncertainty"
        elif key in ["u3hs"]:

            modul.model.certainty_stats = True
            modul.hparams["post_process"]["unknown_clustering"] = {
                "only_ood": False,
                "epsilon": 0.05,
                "min_samples": 5,
            }
            modul.hparams["post_process"]["certainty_threshold"] = -3

            modul.dm.batch_size = 5
            # Mean: 0.9489491581916809, Variance: 0.001957752974703908
            # mean, var = certainty_stats_U3HS(modul, iters=100)
            mean = 0.9489491581916809
            var = 0.001957752974703908
            print(mean, var)
            modul.model.set_certainty_stats(torch.tensor(mean), torch.tensor(var))
    evaluator = COCOPanopticEvaluator(
        dataset_name, modul.dm, os.path.expanduser(f"~/Experiments/PQ/coco_open_{args.key}"), resize=resize
    )

    modul.dm.val_batch_size = 1
    if resize == 0:
        modul.dm.no_resize = True
    elif resize == 640:
        modul.dm.base_size_val = 640,480
    elif resize == 512:
        modul.dm.base_size_val = 512, 512
    elif resize == 1024:
        modul.dm.base_size_val = 1024, 1024



    logger = setup_logger(output=None)
    logger.info("enabled logging")


    if dataset_name == "coco_val_anomaly":
        loader = modul.dm.anomaly_dataloader(split="full_val")
    elif dataset_name == "coco_val_anomaly_100":
        loader = modul.dm.anomaly_dataloader()
    elif dataset_name == "coco_val_100":
        loader = modul.dm.val_dataloader()
    elif dataset_name == "coco_val":
        loader = modul.dm.val_dataloader(split="full_val")
    elif dataset_name == "coco_val_anomaly_InD_100":
        loader = modul.dm.anomaly_dataloader(known_set="InD")
    elif dataset_name == "coco_val_anomaly_InD":
        loader = modul.dm.anomaly_dataloader(known_set="InD",split="full_val")
    elif dataset_name == "coco_val_anomaly_OOD_100":
        loader = modul.dm.anomaly_dataloader(known_set="OOD")
    elif dataset_name == "coco_val_anomaly_OOD":
        loader = modul.dm.anomaly_dataloader(known_set="OOD",split="full_val")
    results = inference_on_dataset(
        ModulWrapper(modul, evaluator._thing_contiguous_id_to_dataset_id),
        loader,
        evaluator,
    )
    print(results["panoptic_seg"])

    print("PQ")
    if "anomaly" in args.dataset_name:
        print(results["panoptic_seg"]["PQ"], results["panoptic_seg"]["per_class"][254])
        print(modul.hparams["post_process"]["unknown_clustering"])

    with open(f"__pycache__/{key}_{resize}_panoptic_{dataset_name}_open.txt", "a") as myfile:
        myfile.write(f"\nResults on :\n")
        myfile.write(f'{results["panoptic_seg"]["PQ"]}, {results["panoptic_seg"]["SQ"]}, {results["panoptic_seg"]["RQ"]}\n')
        if "anomaly" in args.dataset_name:
            myfile.write(f'{results["panoptic_seg"]["per_class"][254]}\n')
            myfile.write(f'{modul.hparams["post_process"]["unknown_clustering"]}\n')

    with open(f"__pycache__/{key}_{resize}_panoptic_{dataset_name}_open.json", "w") as f:
        json.dump(results, f)

