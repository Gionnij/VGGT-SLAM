from argparse import ArgumentParser
import json
import os
import torch
from tqdm import tqdm

from Data_Loaders.lightning_data_modules import get_datamodule
from lightning_utils.panoptic import (
    PanopticSegmentationModule,
)
from metrics.cityscapes_eval_scripts.evalPixelLevelSemanticLabeling import (
    get_cityscapes_evaluator,
)


torch.set_grad_enabled(False)

def get_loader(name, modul, args):
    if name == "cityscapes":
        if args.key == "u3hs":
            modul.dm.base_size_val = (1024, 512)
        return modul.dm.val_dataloader()
    elif name == "foggy":
        if args.key == "u3hs":
            datamodule = {
                "name": "cityscapes_foggy",
                "args": {
                    "batch_size": 2,
                    "val_batch_size": 1,
                    "crop_size": [1024, 512],
                    "base_size": [1024, 512],
                    "base_size_val": [1024, 512],
                    "target_type": ["semantic", "instance"],
                    # "panoptic_preprocessing": "deeplab",
                    "exclude_classes": [],
                    "data_dir": "/Datasets/Cityscapes_foggy",
                    "pattern": args.pattern,
                },
            }
        else:
            datamodule = {
                "name": "cityscapes_foggy",
                "args": {
                    "batch_size": 2,
                    "val_batch_size": 1,
                    "crop_size": [1024, 512],
                    "base_size": [1024, 512],
                    "base_size_val": [2048, 1024],
                    "target_type": ["semantic", "instance"],
                    # "panoptic_preprocessing": "deeplab",
                    "exclude_classes": [],
                    "data_dir": "/Datasets/Cityscapes_foggy",
                    "pattern": args.pattern,
                },
            }

        dm = get_datamodule(
            datamodule["name"], datamodule["args"], datamodule["args"]["data_dir"]
        )
        loader = dm.val_dataloader()
        return loader
    elif name == "rainy":
        return None


ckpts = {
    "P2F": "/Experiments/Cityscapes/p2f/final_checkpoint.ckpt",
    "u3hs": "/Experiments/Cityscapes/u3hs/final_checkpoint.ckpt",
}


parser = ArgumentParser()
parser.add_argument("--key", "-k", type=str, default="u3hs")
parser.add_argument("--pattern", "-p", type=float, default=-1)
parser.add_argument("--store_dir", "-d", type=str, default="~/Experiments/cityscapes_script/mIoU")
parser.add_argument("--name", "-n", type=str, default="cityscapes")
parser.add_argument("--mode", "-m", type=str, default="closed")

args = parser.parse_args()


modul = PanopticSegmentationModule.load_from_checkpoint(ckpts[args.key])
modul = modul.to("cuda")
modul.eval()
#modul.dm.base_size_val = (512, 256)


key=args.key

if args.key == "u3hs":
    modul.model.certainty_stats=None
if args.mode == "open":
    os.makedirs("__cache__", exist_ok=True)
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




cityscapes_evaluator = get_cityscapes_evaluator()



loader = get_loader(args.name, modul, args)


for i, batch in tqdm(enumerate(loader)):
    x = modul.dm.get_image_from_batch(batch).cuda()
    out = modul(x)
    postprocessed = modul.model.postprocess(
        out, modul.dm, modul.hparams["post_process"]
    )
    cityscapes_evaluator.update(
        postprocessed["semantic"].cpu().numpy(), batch["semantic"].numpy()
    )
    del out, postprocessed, x

result = cityscapes_evaluator.compute()
print(result)

os.makedirs(args.store_dir, exist_ok=True)

filename = os.path.join(args.store_dir, f"{args.name}_{args.mode}_{args.key}_{args.pattern}.json")

with open(filename, 'w') as file:
    json.dump(result, file, indent=4)
