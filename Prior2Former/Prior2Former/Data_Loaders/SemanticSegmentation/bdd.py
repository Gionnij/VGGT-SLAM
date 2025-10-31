import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torchvision.datasets.utils import extract_archive, iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class BDD100K(VisionDataset):
    """`BDD100K <https://www.bdd100k.com//>`_ Dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "semantic",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(BDD100K, self).__init__(root, transforms, transform, target_transform)

        self.target_type = target_type
        self.split = split

        self.images_dir = os.path.join(
            self.root,
            "images",
            "10k",
            self.split if "anom" not in split else split.split("_")[1],
        )
        self.targets_dir = os.path.join(
            self.root,
            "labels",
            "sem_seg" if self.target_type == "semantic" else "pan_seg",
            "masks" if self.target_type == "semantic" else "bitmasks",
            self.split if "anom" not in split else split.split("_")[1],
        )
        self.images = []
        if self.split != "anom_all":
            for file_name in os.listdir(self.images_dir):
                img_name = file_name.split(".")[0]
                self.images.append(img_name)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(
            os.path.join(self.images_dir, f"{self.images[index]}.jpg")
        ).convert("RGB")

        result = {
            "image": image,
        }

        target = Image.open(os.path.join(self.targets_dir, f"{self.images[index]}.png"))
        if self.target_type == "semantic":
            result["semantic"] = target
        else:
            p = np.array(target)
            semantic = p[:, :, 0]
            instance = (p[:, :, 2] << 8) + p[:, :, 3]

            result["semantic"] = Image.fromarray(semantic)
            result["instance"] = Image.fromarray(instance)

        if self.transforms is not None:
            result = self.transforms(result)
            return result

        return result

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return "\n".join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as file:
            data = json.load(file)
        return data

    def get_image_by_path(self, rgb_path: str):
        for i, b in enumerate(self.images):
            if rgb_path in b:
                return self.__getitem__(i)
        raise FileNotFoundError(f"Image with path {rgb_path} not found in dataset.")


class BDD100KAnomaly(BDD100K):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "semantic",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):

        super(BDD100KAnomaly, self).__init__(
            root, split, target_type, transform, target_transform, transforms
        )
        self.images = []
        self.targets = []

        self.targets_dir = os.path.join(
            self.root,
            "labels",
            "anm_seg" if self.target_type == "semantic" else "pan_seg",
            (
                "labels" if self.target_type == "semantic" else "anom_bitmasks"
            ),
            self.split if "anom" not in split else split.split("_")[1],
        )
        if split == "train":
            file = "train_list.json"
        elif split == "val":
            file = "validation_list.json"
        elif split == "anom_val":
            file = "anom_files_val_list.json"
        elif split == "anom_all":
            file = "anom_files.json"
        else:
            file = "test_list.json"
        with open(os.path.join(self.root, file), "r") as json_file:
            self.train_image_list = json.load(json_file)

        if split == "anom_all":
            self.images_dir = os.path.join(*self.images_dir.split("/")[:-3])
            self.targets_dir = os.path.join(*self.targets_dir.split("/")[:-1])
            for file_name in self.train_image_list:
                img_name = file_name.split(".")[0].split("/")[1:]
                self.images.append(
                    os.path.join("/", self.images_dir, "images", "10k", *img_name)
                )
                self.targets.append(os.path.join("/", self.targets_dir, *img_name))
        else:
            for file_name in self.train_image_list:
                img_name = file_name.split("/")[-1].split(".")[0]
                self.images.append(os.path.join(self.images_dir, img_name))
                self.targets.append(os.path.join(self.targets_dir, img_name))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(
            os.path.join(self.images_dir, f"{self.images[index]}.jpg")
        ).convert("RGB")

        result = {
            "image": image,
        }

        target = Image.open(
            os.path.join(self.targets_dir, f"{self.targets[index]}.png")
        )
        if self.target_type == "semantic":
            result["semantic"] = target
        else:
            p = np.array(target)
            semantic = p[:, :, 0]
            instance = (p[:, :, 2] << 8) + p[:, :, 3]

            result["semantic"] = Image.fromarray(semantic)
            result["instance"] = Image.fromarray(instance)

        if self.transforms is not None:
            result = self.transforms(result)
        result["original"] = torch.tensor(np.array(image))
        result["file_name"] = self.images[index].split("/")[-1].split(".")[0]
        return result
