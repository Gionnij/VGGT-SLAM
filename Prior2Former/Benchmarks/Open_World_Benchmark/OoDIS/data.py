import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class LAFOdisDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        root_dir = os.path.expanduser(root_dir)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.gt_files = []
        self.img_files = []

        # Traverse the ground truth directory and collect file paths
        gt_dir = os.path.join(root_dir, "gtCoarse_Odis", split)
        for subdir, _, files in os.walk(gt_dir):
            for file in files:
                if file.endswith("_gtCoarse_instanceIds.png"):
                    gt_file = os.path.join(subdir, file)
                    img_file = os.path.join(
                        root_dir,
                        "leftImg8bit",
                        split,
                        os.path.basename(subdir),
                        file.replace("_gtCoarse_instanceIds.png", "_leftImg8bit.png"),
                    )
                    self.gt_files.append(gt_file)
                    self.img_files.append(img_file)

        self.img_files = sorted(self.img_files)

    def __len__(self):
        return len(self.gt_files)

    def get_item_path(self, path):
        # get index of self.img_files where path is contained
        for i, img_file in enumerate(self.img_files):
            if path in img_file:
                return self.__getitem__(i)
        raise ValueError(f"Path {path} not found in dataset")

    def __getitem__(self, idx):
        gt_file = self.gt_files[idx]
        img_file = self.img_files[idx]

        # Load the image and ground truth
        image = Image.open(img_file)
        gt = Image.open(gt_file)

        # Apply any transformations
        result = {"image": image, "semantic": gt}
        result["original_image"] = image
        if self.transform:
            result = self.transform(result)
        result["filename"] = os.path.basename(img_file)

        return result


class Fishyscapes(Dataset):

    def __init__(self, root_dir, split, transform=None):
        root_dir = os.path.expanduser(root_dir)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.img_files = []

        img_dir = os.path.join(root_dir, "leftImg8bit")
        for subdir, _, files in os.walk(img_dir):
            if "/" + split in subdir:
                for file in files:
                    if file.endswith(".png"):

                        img_file = os.path.join(
                            subdir,
                            file,
                        )
                        self.img_files.append(img_file)
        self.img_files = sorted(self.img_files)
        print(
            f"__________________________________________________________________________\nLoaded {len(self.img_files)} images for {split} split\n__________________________________________________________________________"
        )

    def __len__(self):
        return len(self.img_files)

    def get_item_path(self, path):
        # get index of self.img_files where path is contained
        for i, img_file in enumerate(self.img_files):
            if path in img_file:
                return self.__getitem__(i)
        raise ValueError(f"Path {path} not found in dataset")

    def __getitem__(self, idx):
        img_file = self.img_files[idx]

        # Load the image and ground truth
        image = Image.open(img_file)

        gt = Image.fromarray(np.zeros(np.array(image).shape[:-1]))

        # Apply any transformations
        result = {"image": image, "semantic": gt}
        result["original_image"] = image
        if self.transform:
            result = self.transform(result)
        result["filename"] = os.path.basename(img_file)

        return result


class RoadObstacle(Dataset):

    def __init__(self, root_dir, split="train", transform=None):
        root_dir = os.path.expanduser(root_dir)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.gt_files = []
        self.img_files = []

        images_dir = os.path.join(root_dir, "images")
        for file in os.listdir(images_dir):
            if split == "submission" and not file.startswith("validation"):
                img_file = os.path.join(images_dir, file)
                self.img_files.append(img_file)
            elif split == "val" and file.startswith("validation"):
                # val split
                img_file = os.path.join(images_dir, file)
                self.img_files.append(img_file)

        if split == "val":
            gt_dir = os.path.join(root_dir, "labels_masks")
            self.gt_files = [
                os.path.join(
                    gt_dir,
                    os.path.basename(img_file)
                    .replace(".webp", "_labels_semantic.png")
                    .replace(".jpg", "_labels_semantic.png"),
                )
                for img_file in self.img_files
            ]
        self.img_files = sorted(self.img_files)
        self.gt_files = sorted(self.gt_files)

    def __len__(self):
        return len(self.img_files)

    def get_item_path(self, path):
        # get index of self.img_files where path is contained
        for i, img_file in enumerate(self.img_files):
            if path in img_file:
                return self.__getitem__(i)
        raise ValueError(f"Path {path} not found in dataset")

    def __getitem__(self, idx):
        img_file = self.img_files[idx]

        # Load the image and ground truth
        image = Image.open(img_file)
        if self.split == "val":
            gt_file = self.gt_files[idx]
            gt = Image.open(gt_file)
        else:
            gt = Image.fromarray(np.zeros(np.array(image).shape[:-1]))

        # Apply any transformations
        result = {"image": image, "semantic": gt}
        result["original_image"] = image
        if self.transform:
            result = self.transform(result)
        result["filename"] = os.path.basename(img_file)

        return result
