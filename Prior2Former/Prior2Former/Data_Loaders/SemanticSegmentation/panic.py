import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PANICDataset(Dataset):
    def __init__(self, root, split="val", transforms=None):
        """
        Args:
            root (str): Path to the PANIC root directory (e.g., '/Datasets/PANIC')
            split (str): One of {'val', 'test'}
            transforms (callable, optional): Optional transforms applied to the image and labels
        """
        assert split in {"val", "test"}, f"Invalid split: {split}"
        self.root = root
        self.split = split
        self.transforms = transforms

        self.image_dir = os.path.join(root, split, "image")
        self.semantic_dir = os.path.join(root, split, "semantic") if split == "val" else None
        self.instance_dir = os.path.join(root, split, "instance") if split == "val" else None

        self.filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(".png") and "_label_" not in f
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        sample = {
            "image": image,
            "meta": {
                "name": img_name
            }
        }

        if self.split == "val":
            base = img_name.replace(".png.png", ".png") + "_label_ground-truth"
            semantic_path = os.path.join(self.semantic_dir, f"{base}_semantic.png")
            instance_path = os.path.join(self.instance_dir, f"{base}_instance.png")

            semantic = Image.open(semantic_path)
            instance = Image.open(instance_path)

            sample["semantic"] = semantic
            sample["instance"] = instance

        if self.transforms:
            sample = self.transforms(sample)

        return sample
    
    def compute_normalization_constants(self, max_samples=None):
        """
        Computes the mean and standard deviation (per channel) over the dataset.

        Args:
            max_samples (int or None): Number of images to use for computing statistics.
                                    If None, uses the entire dataset.

        Returns:
            mean (tuple of floats): (mean_R, mean_G, mean_B)
            std (tuple of floats): (std_R, std_G, std_B)
        """
        from torchvision import transforms
        from torch.utils.data import Subset
        from tqdm import tqdm

        to_tensor = transforms.ToTensor()

        if max_samples is not None:
            dataset = Subset(self, list(range(min(max_samples, len(self)))))
        else:
            dataset = self

        mean = 0.0
        std = 0.0
        total = 0

        for sample in tqdm(dataset, desc="Computing normalization constants"):
            img = to_tensor(sample["image"])  # shape: [C, H, W]
            total += 1
            mean += img.mean(dim=[1, 2])
            std += img.std(dim=[1, 2])

        mean /= total
        std /= total

        return tuple(mean.numpy()), tuple(std.numpy())

def main():
    root_dir = "/Datasets/PANIC"

    print("🧪 Testing VAL split:")
    val_dataset = PANICDataset(root=root_dir, split="val")
    print(f"Total val samples: {len(val_dataset)}")
    val_sample = val_dataset[0]

    print(f"Image name: {val_sample['meta']['name']}")
    print(f"Image size: {val_sample['image'].size}")
    print(f"Semantic size: {val_sample['semantic'].size}")
    print(f"Instance size: {val_sample['instance'].size}")
    print(val_dataset.compute_normalization_constants())

if __name__ == "__main__":
    main()
