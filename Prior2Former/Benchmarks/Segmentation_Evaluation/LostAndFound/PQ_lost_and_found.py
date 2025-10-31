import os
import torch
import itertools
import argparse
import csv
import numpy as np
from tqdm import tqdm
from Data_Loaders.lightning_data_modules import LostAndFoundDataModule
from lightning_utils.panoptic import PanopticSegmentationModule
from metrics.pq_u3hs import PanopticQuality

def parse_arguments():
    parser = argparse.ArgumentParser(description="Panoptic Segmentation Evaluation Script")
    parser.add_argument("--key", type=str, default="P2F", help="Model key to load checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--estimator", "-e", type=str, default = "", help="set uncertainty estimator, only for M2F")
    parser.add_argument("--data_dir", type=str, default="/Datasets/dataset_LostAndFound", help="Path to the dataset directory")
    parser.add_argument("--exp_folder", type=str, default="/Experiments/", help="Folder to save results")
    parser.add_argument("--number_of_elements", "-nof", type=int, default=-1, help="Number of elements to evaluate, -1 for all")
    parser.add_argument("--evaluation_size", type=str, default="low",choices=["high", "low"], help="Size of the evaluation images,choices: 'high', 'low'")
    return parser.parse_args()

def load_model(ckpt_file, device, evaluation_size):
    module = PanopticSegmentationModule.load_from_checkpoint(ckpt_file)
    module.dm.base_size_val = evaluation_size
    module.to(device)
    module.eval()
    return module

def load_data(evaluation_size, number_of_elements, data_dir):
    dm_config = {
        "data_dir": data_dir,
        "base_size": tuple(evaluation_size),
        "crop_size": tuple(evaluation_size),
        "batch_size": 1,
        "val_batch_size": 1,
        "variant": "original",
        "dataset_args": {"load_instances": True},
    }
    datamodule = LostAndFoundDataModule(**dm_config)
    return datamodule.test_dataloader(number_of_elements=number_of_elements, shuffle=number_of_elements > 0)

def evaluate_model(module, dataloader, device):
    pq = PanopticQuality(compute_on_step=False, per_class=True, num_classes=2)
    pq.to(device)
    
    for x in tqdm(dataloader, desc="Evaluating"):
        image = x["image"].to(device)
        with torch.no_grad():
            result = module(image)
            postprocessed = module.model.postprocess(result, module.dm, module.hparams["post_process"])

        # x["instance"]==0 is the background
        # x["instance"]==1000 is the road which is the area of interest
        # x["instance"] in [2000,3000,4000,...] are the anomalies
        panoptic_gt = x["instance"].clone()
        panoptic_gt[x["instance"] == 0] = 255000
        anomaly = (x["instance"] > 1000) & (x["instance"] != 255000)

        # after the following we have the ignore label 255000 and 0 for street, 1001, 1002,1003,... for anomalies
        panoptic_gt[anomaly] = x["instance"][anomaly] // 1000 + 1000
        panoptic_gt[x["instance"] == 1000] = 0

        # 0 for all known pixels, 1001,1002,... for all unknowns
        panoptic_pred = postprocessed["panoptic"][0]
        panoptic_pred[panoptic_pred > 0] += 1000
        pq.update(panoptic_pred, panoptic_gt[0].to(device))
    
    pq_values = torch.stack(pq.compute()).cpu()
    return pq_values[0:3], pq_values[12:15]

def main():
    args = parse_arguments()
    evaluation_size = (2048,1024) if args.evaluation_size == "high" else (1024,512)
    number_of_elements = args.number_of_elements

    s = str(number_of_elements) + "_samples" if number_of_elements != -1 else "all"
    results_file = f"{args.exp_folder}/PQ_LandF_{args.key}_{evaluation_size[0]}x{evaluation_size[1]}_{s}.csv"
    os.makedirs(args.exp_folder,exist_ok=True)
    if args.estimator != "":
        results_file = results_file.replace(".csv", f"_{args.estimator}.csv")
    ckpts = {
        "P2F":  "/Experiments/Cityscapes/p2f/final_checkpoint.ckpt",
        "M2A":  "/Experiments/Cityscapes/m2a/final_checkpoint.ckpt",
        "u3hs": "/Experiments/Cityscapes/u3hs/final_checkpoint.ckpt",
        "M2F":  "/Experiments/Cityscapes/m2f/final_checkpoint.ckpt",
    }
   
    ckpt_file = ckpts[args.key]  # Default to P2F if key not found
    
    module = load_model(ckpt_file, args.device, evaluation_size)
    dataloader = load_data(evaluation_size, number_of_elements, args.data_dir) 
    
    
    if "P2F" == args.key:
        uncertainty_thresholds = [-0.2, -0.3]
        min_samples_values = [15,20]
        eps_values = [0.05, 0.8,0.12]
        module.hparams["post_process"]["uncertanty_estimator"] = "max_alpha_beta_cls_uncertainty_keep"
    elif "M2F" == args.key:
        print(args.estimator)
        if args.estimator == "rba":
            uncertainty_thresholds = [0.95]
            module.hparams["post_process"]["normalize"] = True
        elif args.estimator == "eam":
            uncertainty_thresholds = [0.95]
            module.hparams["post_process"]["normalize"] = True
        else:
            uncertainty_thresholds = [0.9]
        min_samples_values = [20]
        eps_values = [0.1]

        #default is temperature=1, hence, this is the classical uncertainty
        module.hparams["post_process"]["uncertanty_estimator"] = args.estimator
    elif "M2A" == args.key:
        print("M2A")
        uncertainty_thresholds = [0.9]
        min_samples_values = [20]
        eps_values = [0.1]
        module.hparams["post_process"]["uncertanty_estimator"] = "mask2anomaly"
    
    iter_fn = itertools.product 
    
    for uncertainty_threshold, min_samples, eps in iter_fn(uncertainty_thresholds, min_samples_values, eps_values):
        print(f"Testing with uncertainty_threshold={uncertainty_threshold}, min_samples={min_samples}, eps={eps}, estimator={args.estimator}")
        
        if "u3hs" == args.key:
            raise NotImplementedError()
        else:
            module.hparams["post_process"]["unknown_clustering"] = {
                "eps": eps,
                "min_samples": min_samples,
                "uncertainty_threshold": uncertainty_threshold,
                "only_ood": True,
                "distance_type": "cosine",
            }
            
        pq_values = evaluate_model(module, dataloader, args.device)
        overall_pq = pq_values[0]
        anomaly_pq = pq_values[-1]
        
        result = {
            "uncertainty_threshold": uncertainty_threshold,
            "min_samples": min_samples,
            "eps": eps,
            "overall_pq": overall_pq[0].item(),
            "overall_sq": overall_pq[1].item(),
            "overall_rq": overall_pq[2].item(),
            "anomaly_pq": anomaly_pq[0].item(),
            "anomaly_sq": anomaly_pq[1].item(),
            "anomaly_rq": anomaly_pq[2].item(),
        }
        
        write_header = not os.path.exists(results_file)
        with open(results_file, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "uncertainty_threshold", "min_samples", "eps",
                "overall_pq", "overall_sq", "overall_rq",
                "anomaly_pq", "anomaly_sq", "anomaly_rq"
            ])
            if write_header:
                writer.writeheader()
            writer.writerow(result)
        print(f"Intermediate results saved to {results_file}")
    
    print("Grid search completed. Results saved to results file.")

if __name__ == "__main__":
    main()
