import copy
import os
from types import SimpleNamespace

from typing import Any, Callable, Dict, Optional

from PIL import Image, ImageFile
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset

from Data_Loaders.lightning_data_modules.transforms.panoptic import (
    remap_instance,
)
from .coco_utils import (
    COCO_CATEGORIES,
    load_coco_panoptic_json,
    get_metadata,
    COCO_CATEGORIES_WITHOUT_UNKNOWN,
)
from panopticapi.utils import rgb2id

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CocoDataset(VisionDataset):

    # CocoClass = namedtuple(
    #     "CocoClass",
    #     [
    #         "color",
    #         "isthing",
    #         "id",
    #         "name",
    #     ],
    # )
    classes = [SimpleNamespace(**cat) for cat in COCO_CATEGORIES]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: str = ["panoptic"],
        anomaly: bool = False,
        original_split: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        root = os.path.expanduser(root)
        super(CocoDataset, self).__init__(root, transforms, transform, target_transform)
        split = split if split != "main_train" else "train"
        if anomaly:
            if original_split:
                prefix = "org_anomaly_InD_"
            else:
                prefix = "anomaly_InD_"
            remove_unknowns=True
            self.classes = [SimpleNamespace(**cat) for cat in COCO_CATEGORIES_WITHOUT_UNKNOWN]
        else:
            remove_unknowns = False
            prefix = ""
        if split == "val":
            img_dir = os.path.join(root, f"panoptic_{split}2017_100")
            ann_file = os.path.join(
                root, "annotations", f"{prefix}panoptic_{split}2017_100.json"
            )
        elif split == "full_val":
            split = "val"
            img_dir = os.path.join(root, f"panoptic_{split}2017")
            ann_file = os.path.join(
                root, "annotations", f"{prefix}panoptic_{split}2017.json"
            )
        elif split == "train":
            img_dir = os.path.join(root, f"panoptic_{split}2017")
            ann_file = os.path.join(
                root, "annotations", f"{prefix}panoptic_{split}2017.json"
            )
        else:
            raise NotImplementedError(f"split {split} is unknown")

        gt_panoptic_segmentation = os.path.join(
            root, "annotations", f"panoptic_{split}2017"
        )
        gt_semantic_segmentation = os.path.join(
            root, "annotations", f"segmentation_{split}2017"
        )
        # list of dictionaries with image_file, panoptic_segmentation file and semantic segmentation file
        self.dataset = load_coco_panoptic_json(
            ann_file,
            img_dir,
            gt_panoptic_segmentation,
            gt_semantic_segmentation,
            meta=get_metadata(remove_unknowns),
        )

        self.target_type = target_type
        self.split = split

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index: int

        Returns:
            dict: returns fromat fitting for data augmentations
        """

        dataset_dict = copy.deepcopy(
            self.dataset[index]
        )  # it will be modified by code below
        image = Image.open(dataset_dict["file_name"]).convert("RGB")

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = Image.open(dataset_dict.pop("sem_seg_file_name"))
        else:
            sem_seg_gt = None

        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = Image.open(dataset_dict.pop("pan_seg_file_name"))
            instance = remap_instance(torch.tensor(rgb2id(np.array(pan_seg_gt))))
            pan_seg_gt = Image.fromarray(instance.numpy())
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )
        result = {
            "image": image,
            "original_image": image,
            "semantic": sem_seg_gt,
            "instance": pan_seg_gt,
        }

        if self.transforms is not None:
            result = self.transforms(result)

        result["info"] = os.path.basename(dataset_dict["file_name"])
        return result

    def __getitem2__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")

        targets: Dict = {}
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets[t] = target

        result = {
            "image": image,
            "original_image": image,
            "info": self.infos[index],
            **targets,
        }

        if self.transforms is not None:
            result = self.transforms(result)
            return result

        return result

    def get_image_by_path(self, rgb_path: str):
        for i, b in enumerate(self.dataset):
            if rgb_path in b["file_name"]:
                break
        return self.__getitem__(i)
        image = Image.open(rgb_path).convert("RGB")

        targets: Dict = {}
        for k in target_paths.keys():
            if k == "polygon":
                target = self._load_json(target_paths[k])
            else:
                target = Image.open(target_paths[k])

            targets[k] = target

        result = {
            "image": image,
            "original_image": image,
            **targets,
        }

        if self.transforms is not None:
            result = self.transforms(result)
            return result

        return result

    def __len__(self) -> int:
        return len(self.dataset)


class CocoDatasetAnomaly(CocoDataset):
    """Dataclass for the CocoImages containing the left out classes

    Args:
        CocoDataset (_type_): _description_
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        known_set: str = "full",
        target_type: str = ["panoptic"],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        full_set: bool = True,
    ) -> None:
        root = os.path.expanduser(root)
        super(CocoDataset, self).__init__(root, transforms, transform, target_transform)
        split = split if split != "main_train" else "train"
        if known_set =="full":
            prefix = "anomaly_full_"
        elif known_set == "OOD":
            prefix = "anomaly_OOD_"
        elif known_set == "InD":
            prefix = "anomaly_InD_"
        if split == "val":
            img_dir = os.path.join(root, f"panoptic_{split}2017_100")
            ann_file = os.path.join(
                root,
                "annotations",
                f"{prefix}panoptic_{split}2017_100.json",
            )
        elif split == "full_val":
            split = "val"
            img_dir = os.path.join(root, f"panoptic_{split}2017")
            ann_file = os.path.join(
                root, "annotations", f"{prefix}panoptic_{split}2017.json"
            )
        elif split == "train":
            img_dir = os.path.join(root, f"panoptic_{split}2017")
            ann_file = os.path.join(
                root, "annotations", f"{prefix}panoptic_{split}2017.json"
            )
        else:
            raise NotImplementedError(f"split {split} is unknown")
        prefix = ""
        gt_panoptic_segmentation = os.path.join(
            root, "annotations", f"panoptic_{split}2017"
        )
        gt_semantic_segmentation = os.path.join(
            root, "annotations", f"segmentation_{prefix}{split}2017"
        )
        # list of dictionaries with image_file, panoptic_segmentation file and semantic segmentation file
        self.dataset = load_coco_panoptic_json(
            ann_file,
            img_dir,
            gt_panoptic_segmentation,
            gt_semantic_segmentation,
            meta=get_metadata(remove_unknowns=True),
        )

        self.target_type = target_type
        self.split = split