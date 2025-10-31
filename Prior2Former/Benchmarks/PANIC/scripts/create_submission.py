import csv
import os
import argparse
import json
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from Prior2Former.Data_Loaders.SemanticSegmentation.panic import PANICDataset
from Prior2Former.lightning_utils.panoptic import PanopticSegmentationModule

from Prior2Former.metrics.pq_u3hs import PanopticQuality


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export PANIC Benchmark Submission")
    parser.add_argument("--key", type=str, default="P2F", help="Model key to load checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--data_dir", type=str,  help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str,  help="Path to store semantic/instance masks and zip")
    parser.add_argument("--split", default = "val", type=str, choices=["val", "test"])
    parser.add_argument(
        "--eval_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Evaluation image size as two integers: --eval_size 1024 512"
    )
    parser.add_argument("--number_of_elements", type=int, default=-1, help="Number of elements to export, -1 for all")
    parser.add_argument("--create_submission", action="store_true", help="If set, creates the ZIP submission file")
    parser.add_argument("--compute_pq", action="store_true", help="If set, evaluates the model using anomaly PQ metric")
    args = parser.parse_args()
    print("Parsed args:", vars(args))
    assert args.compute_pq or args.create_submission, "at least one of these two should be used, else the script has no purpose"
    assert not (args.split == "test" and args.compute_pq), "no labels available for the test split, hence pq computation not possible"
    return args

def get_checkpoint_path(key):
    ckpts = {
        "P2F":  "<path_to_your_P2F_checkpoint>",
    }
    return ckpts.get(key, "")

def load_model(ckpt_path, device, eval_size):
    module = PanopticSegmentationModule.load_from_checkpoint(ckpt_path)
    module.dm.base_size_val = eval_size
    module.to(device)
    module.eval()
    return module


def load_dataloader(eval_size, number_of_elements, data_dir, split="val", num_workers=4, shuffle=False):
    """
    Loads a standard PyTorch DataLoader for the PANIC dataset with transforms.

    Args:
        eval_size (tuple): Desired image size (W, H) for resizing (e.g., (1024, 512))
        number_of_elements (int): If > 0, only load a subset of that many samples
        data_dir (str): Root directory of the PANIC dataset
        split (str): 'val' or 'test'
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
        shuffle (bool): Shuffle dataset (only if using a subset)

    Returns:
        torch.utils.data.DataLoader
    """

    # Define basic transform 
    def transform(sample):
        image = sample["image"]
        print("no resize", eval_size)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=(0.49041906, 0.49897644, 0.48534194),
                                     std=(0.26374286, 0.2672853, 0.27683198))(image)
        sample["image"] = image

        if "semantic" in sample:
            sample["semantic"] = transforms.ToTensor()(sample["semantic"]).squeeze(0).long()
        if "instance" in sample:
            sample["instance"] = transforms.ToTensor()(sample["instance"]).squeeze(0).long()

        return sample

    dataset = PANICDataset(root=data_dir, split=split, transforms=transform)

    # Optional: take a subset of the dataset
    if number_of_elements > 0:
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)
        dataset = Subset(dataset, indices[:number_of_elements])

    return DataLoader(dataset,
                      batch_size=1,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)

def export_panic_predictions(module, dataloader, device, out_dir, key, compute_pq, create_submission, number_of_elements):
    """
    Runs inference on the provided dataloader using a given segmentation model and 
    exports the results in the required PANIC Open-Set Panoptic Segmentation benchmark format.

    This function:
    - Configures the model's open-set postprocessing settings based on the `key`
    - Performs forward inference on each sample
    - Generates two outputs for each image:
        1. A binary semantic segmentation mask (1 = unknown/anomaly, 0 = known)
        2. An instance segmentation mask (unique ID > 0 for each unknown object, 0 for known/background)
    - Saves the outputs into `semantic/` and `instance/` subfolders within `out_dir`

    Parameters:
    ----------
    module : torch.nn.Module
        The panoptic segmentation model (should implement `.model.postprocess`).
    
    dataloader : torch.utils.data.DataLoader
        A PyTorch DataLoader yielding dicts containing at least:
        - 'image': input image tensor

    
    device : str
        Computation device ('cuda' or 'cpu') used for inference.

    out_dir : str
        Directory where the PANIC-formatted outputs will be stored. If it does not end on submission the subfolder submission is added.
        Subfolders:
        - `semantic/` will contain binary PNG masks
        - `instance/` will contain instance segmentation PNG masks

    key : str
        Model key used to select model-specific uncertainty thresholds and clustering configs.

    Raises:
    -------
    NotImplementedError
        If no post-processing configuration is defined for the specified `key`.

    Output Structure:
    -----------------
    out_dir/submission/      # if create_submission
    ├── semantic/
    │   ├── img_001.png      # Binary semantic mask (int): 1=unknown, 0=known
    │   └── ...
    └── instance/
        ├── img_001.png      # Instance mask (int): 0=known, unique ID per unknown instance
        └── ...
    out_dir/pq/              # if compute_pq
    ├── results.csv
    """

    if "P2F" == key:
        uncertainty_thresholds = [-0.1]
        min_samples_values = [200]
        eps_values = [0.08] #0.05 for resize
        module.hparams["post_process"]["uncertanty_estimator"] = "max_alpha_beta_cls_uncertainty_keep"
    else:
        raise NotImplementedError(f"No open-world prediction config for key: {key}")
    
    use_original_image_size = True if module.dm.base_size_val is None else False
    # this is done so i can extend it later better to a grid search
    for uncertainty_threshold in uncertainty_thresholds:
        for eps in eps_values:
            for min_samples in min_samples_values:
    

                module.hparams["post_process"]["unknown_clustering"] = {
                    "uncertainty_threshold": uncertainty_threshold,
                    "min_samples": min_samples,
                    "eps": eps,
                    "only_ood": True,
                    "distance_type": "cosine",
                }   
                if create_submission:
                    out_dir_sub = os.path.join(out_dir, "submission")
                    semantic_dir = os.path.join(out_dir_sub, "semantic")
                    instance_dir = os.path.join(out_dir_sub, "instance")
                    os.makedirs(semantic_dir, exist_ok=True)
                    os.makedirs(instance_dir, exist_ok=True)
                if compute_pq:
                    out_dir_pq = os.path.join(out_dir, "pq")
                    os.makedirs(out_dir_pq, exist_ok=True)
                    out_file_pq = os.path.join(out_dir_pq, f"results_{number_of_elements}.csv")


                if compute_pq:
                    pq = PanopticQuality(compute_on_step=False, per_class=True, num_classes=2)
                    pq.to(device)
                    

                count = 0
                for batch in tqdm(dataloader, desc="Exporting PANIC masks"):
                    count +=1
                    image = batch["image"].to(device)
                    img_name = batch["meta"]["name"][0] #.replace(".png", "") + ".png" # has multiply .png in file name at the end :(?? haha

                    if use_original_image_size:
                        module.dm.base_size_val = image.shape[-1], image.shape[-2]
                        print(module.dm.base_size_val)
                    else:
                        prev_size = image.shape[-2:]
                        image = F.interpolate(image,size=(module.dm.base_size_val[1],module.dm.base_size_val[0]), mode="bilinear")

                    with torch.no_grad():
                        result = module(image)
                        post = module.model.postprocess(result, module.dm, module.hparams["post_process"])

                    if use_original_image_size:
                        panoptic_pred = post["panoptic"][0].cpu().numpy()
                    else:
                        panoptic_pred = F.interpolate(post["panoptic"].unsqueeze(0).to(float), size=prev_size, mode="nearest").to(int).squeeze().cpu().numpy()

                    # Binary semantic: unknown = 1
                    anomaly_mask = panoptic_pred > 0
                    semantic = anomaly_mask.astype(int)

                    # Instance mask: unique ID per unknown, 0 for known
                    instance = panoptic_pred.astype(int)
                    if create_submission:
                        cv2.imwrite(os.path.join(semantic_dir, img_name), semantic)
                        cv2.imwrite(os.path.join(instance_dir, img_name), instance)
                    
                    if compute_pq:
                        panoptic = instance
                        # 0 for known and 1001, 1002,...  for unknowns
                        panoptic[anomaly_mask] = panoptic[anomaly_mask]+1000
                        panoptic = torch.tensor(panoptic).to(device)

                        # generate labels, since the ground truth for known classes is not gicen i assume
                        # that the most common label belongs to the ground truth class
                        panoptic_gt = batch["instance"]
                        inst_values, inst_counts = torch.unique(panoptic_gt, return_counts=True)
                        most_frequent_inst = inst_values[torch.argmax(inst_counts)].item()

                        
                        known_mask = panoptic_gt == most_frequent_inst
                        panoptic_gt+=1000
                        panoptic_gt[known_mask] = 0
                        panoptic_gt = panoptic_gt.to(device)

                        pq.update(panoptic, panoptic_gt.squeeze())
                        
                if compute_pq:
                    pq_values = torch.stack(pq.compute()).cpu()
                    overall_pq, anomaly_pq = pq_values[0:3], pq_values[12:15]
                    
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
                    
                    write_header = not os.path.exists(out_file_pq)
                    with open(out_file_pq, "a", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            "uncertainty_threshold", "min_samples", "eps",
                            "overall_pq", "overall_sq", "overall_rq",
                            "anomaly_pq", "anomaly_sq", "anomaly_rq"
                        ])
                        if write_header:
                            writer.writeheader()
                        writer.writerow(result)
                    print(f"Intermediate results saved to {out_file_pq}")
                


                if create_submission:
                    print(f"Wrote binary semantic prediction to {semantic_dir}")
                    print(f"Wrote anomaly instance prediction to {instance_dir}")
   
    


def write_information_json(out_dir):
    info = {
        "model_name": "Prior2Former",
        "link_to_paper": "https://arxiv.org/abs/2504.04841",
        "link_to_code": "",
        "network_params": "",
        "gflops": "",
        "use_ood_data": "no"
    }
    os.makedirs(os.path.join(out_dir, "submission"), exist_ok=True)
    with open(os.path.join(out_dir, "submission", "information.json"), "w") as f:
        json.dump(info, f, indent=2)


def main():
    args = parse_arguments()
    eval_size = args.eval_size

    ckpt_path = get_checkpoint_path(args.key)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint for {args.key} not found at {ckpt_path}")

    print(f"Loading model: {args.key} from {ckpt_path}")
    module = load_model(ckpt_path, args.device, eval_size)
    dataloader = load_dataloader(eval_size, args.number_of_elements, args.data_dir, args.split , shuffle=args.number_of_elements>0)

    print("Generating semantic and instance masks...")
    export_panic_predictions(module, dataloader, args.device, args.output_dir, args.key, args.compute_pq, args.create_submission, args.number_of_elements)

    if args.create_submission:
        write_information_json(args.output_dir)

if __name__ == "__main__":
    main()
