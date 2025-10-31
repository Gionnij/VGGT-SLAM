# %%
import json
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path

from argparse import ArgumentParser
torch.set_grad_enabled(False)

from lightning_utils.panoptic import (
    PanopticSegmentationModule,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.utils.logger import setup_logger


import os
from metrics.coco_panoptic.panoptic_evaluation import (
    COCOPanopticEvaluator,
)

RESULTS_PAHT=""


class ModulWrapper(torch.nn.Module):
    def __init__(self, modul):
        super().__init__()
        self.modul = modul
        if hasattr(self.modul.model, "certainty_stats"):
            self.modul.model.certainty_stats = None

    def forward(self, inputs):
        # returns the correct output needed for the evaluation and changes the input inplace
        x = self.modul.dm.get_image_from_batch(inputs).to(self.modul.device)
        out = self.modul(x)
        postprocess = self.modul.model.postprocess(
            out, self.modul.dm, self.modul.hparams["post_process"]
        )
        res = {"panoptic_seg": []}
        inputs["file_name"] = []
        inputs["image_id"] = []
        for city, file_name in zip(inputs["info"][0], inputs["info"][1]):
            inputs["file_name"].append(file_name)
            inputs["image_id"].append(file_name.replace("_leftImg8bit.png", ""))
        if "segment_info" in postprocess:
            res["panoptic_seg"].append(
                (
                    postprocess["panoptic"].squeeze() % self.modul.dm.label_divisor,
                    postprocess["segment_info"][0],
                )
            )
            return res

        segments_info = []
        for id in np.unique(postprocess["panoptic"].cpu()):
            pred_class = id // self.modul.dm.label_divisor
            isthing = pred_class in self.modul.dm.mapped_thing_list
            # merge stuff regions

            segments_info.append(
                {
                    "id": int(id),
                    "isthing": bool(isthing),
                    "category_id": int(pred_class),
                }
            )
        res["panoptic_seg"].append(
            (
                postprocess["panoptic"].squeeze() // self.modul.dm.label_divisor,
                segments_info,
            )
        )
        return res


def eval_closed_world_old():

    ckpt_paths = {
        "P2F": "/Experiments/Cityscapes/p2f/final_checkpoint.ckpt",
        "u3hs": "/Experiments/Cityscapes/u3hs/final_checkpoint.ckpt",
        "M2F": "/Experiments/Cityscapes/m2f/final_checkpoint.ckpt",
    }
    for key, ckpt in tqdm(ckpt_paths.items()):
        if key in [
            # "beta-softmax",
            # "baseline_our_training",
            # "beta-dpn",
            # "beta-softmax-ps",
            # "beta-softmax-gma",
            # "beta-softmax-gma-ps",
        ]:
            continue
        modul = PanopticSegmentationModule.load_from_checkpoint(ckpt)
        modul.to("cuda")
        modul.eval()

        evaluator = COCOPanopticEvaluator("cityscapes", modul.dm, "./output/inference")

        modul.dm.val_batch_size = 1
        modul.dm.base_size_val = (2048, 1024)
        modul.dm.base_size_val = (1024, 512)
        # modul.dm.base_size_val = (512, 256)

        logger = setup_logger(output=None)
        logger.info("enabled logging")

        results = inference_on_dataset(
            ModulWrapper(modul),
            modul.dm.val_dataloader(),
            evaluator,
        )
        with open(f"__pycache__/{key}_panoptic.json", "w") as f:
            json.dump(results, f)

    result = []
    for key in ckpt_paths.keys():
        if key == "u3hs":
            continue
        with open(f"{RESULTS_PAHT}/{key}_panoptic.json", "r") as f:
            result.append(json.load(f))
    generate_latex_table(
        result,
        os.path.expanduser(
            "~/Experiments/Open_World_Benchmark/benchmark_results_coco/metrics_panpotic.tex"
        ),
    )


def eval_closed_world(key,ckpt_path,storage_folder):

    os.makedirs(storage_folder, exist_ok=True)

    # beta-softmax is final for paper


    modul = PanopticSegmentationModule.load_from_checkpoint(ckpt_path)
    modul.to("cuda")
    modul.eval()

    evaluator = COCOPanopticEvaluator("cityscapes", modul.dm, "./output/inference")

    modul.dm.val_batch_size = 1

    modul.dm.base_size_val = (2048, 1024)
    # modul.dm.base_size_val = (1024, 512)
    # modul.dm.base_size_val = (512, 256)


    logger = setup_logger(output=None)
    logger.info("enabled logging")

    results = inference_on_dataset(
        ModulWrapper(modul),
        modul.dm.val_dataloader(),
        evaluator,
    )

    with open(f"{storage_folder}/{key}_closed_panoptic.json", "w") as f:
        json.dump(results, f)

    result = []


def eval_open_world(key,ckpt_path,storage_folder):

    os.makedirs(storage_folder, exist_ok=True)

    # beta-softmax is final for paper


    modul = PanopticSegmentationModule.load_from_checkpoint(ckpt_path)
    modul.to("cuda")
    modul.eval()
    if key in ["P2F"]:
        modul.hparams["post_process"]["unknown_clustering"] = {
            "only_ood": False,
            "eps": 0.04,
            "min_samples": 17,
            "distance_type": "cosine",
            "uncertainty_threshold": -0.3,
        }

        modul.hparams["post_process"][
            "uncertanty_estimator"
        ] = "m2f_logit_uncertainty_beta_keep"
    elif key == "u3hs":

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

    evaluator = COCOPanopticEvaluator("cityscapes", modul.dm, "./output/inference")

    modul.dm.val_batch_size = 1
    modul.dm.base_size_val = (2048, 1024)
    # modul.dm.base_size_val = (1024, 512)
    # modul.dm.base_size_val = (512, 256)

    logger = setup_logger(output=None)
    logger.info("enabled logging")

    results = inference_on_dataset(
        ModulWrapper(modul),
        modul.dm.val_dataloader(),
        evaluator,
    )
    with open(f"{storage_folder}/{key}_open_panoptic.json", "w") as f:
        json.dump(results, f)

    # read in the files from folder and extract the tables
    result = []


def eval_laf_world(key,ckpt_path,storage_folder):

    os.makedirs(storage_folder, exist_ok=True)

    # beta-softmax is final for paper


    modul = PanopticSegmentationModule.load_from_checkpoint(ckpt_path)
    modul.to("cuda")
    modul.eval()
    if key in ["P2F"]:
        modul.hparams["post_process"]["unknown_clustering"] = {
            "only_ood": False,
            "eps": 0.04,
            "min_samples": 17,
            "distance_type": "cosine",
            "uncertainty_threshold": -0.3,
        }

        modul.hparams["post_process"][
            "uncertanty_estimator"
        ] = "m2f_logit_uncertainty_beta_keep"
    elif key == "u3hs":

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


if __name__ == "__main__":

    # This experiment is not in the paper
    ckpts = {
        "P2F": "/Experiments/Cityscapes/p2f/final_checkpoint.ckpt",
        "u3hs": "/Experiments/Cityscapes/u3hs/final_checkpoint.ckpt",
        "M2F": "/Experiments/Cityscapes/m2f/final_checkpoint.ckpt",
    }

    parser = ArgumentParser()
    parser.add_argument("--key", "-k", type=str, default="u3hs")
    parser.add_argument("--pattern", "-p", type=float, default=-1)
    parser.add_argument("--store_dir", "-d", type=str, default="~/Experiments/foggy/pq")
    parser.add_argument("--name", "-n", type=str, default="closed")
    parser.add_argument("--mode", "-m", type=str, default="open")
    parser.add_argument("--subset", "-s", choices = [0,5,10,50] , type = int)
    parser.add_argument("--dataset_root", "-ds", type = str, default = "/Datasets/Cityscapes")
    parser.add_argument("--threshold", "-t", type = float, default=-0.3)
    args = parser.parse_args()

    if args.name == "closed":
        eval_closed_world(args.key,ckpts[args.key],args.store_dir)
    elif args.name == "open":
        eval_open_world(args.key,ckpts[args.key],args.store_dir)
    elif args.name == "laf":
        eval_open_world(args.key,ckpts[args.key],args.store_dir)


