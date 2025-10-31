import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, random_split
from torchvision import transforms as transforms_lib

from Data_Loaders.SemanticSegmentation.bdd import (
    BDD100K,
    BDD100KAnomaly,
)
from Data_Loaders.lightning_data_modules.transforms.panoptic import (
    MaxDeeplabTargetGenerator,
    PanopticTargetGenerator,
    PrototypicalDeeplabTargetGenerator,
    Mask2FormerSemanticTargetGenerator,
    Mask2FormerPanopticTargetGenerator,
)

from .transforms import transforms as tr
from .utils import get_collate

BDD100K_PATH = ""
semseg_classes = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

semantic_colors = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

panoptic_classes = {
    0: "unlabeled",
    1: "dynamic",
    2: "ego vehicle",
    3: "ground",
    4: "static",
    5: "parking",
    6: "rail track",
    7: "road",
    8: "sidewalk",
    9: "bridge",
    10: "building",
    11: "fence",
    12: "garage",
    13: "guard rail",
    14: "tunnel",
    15: "wall",
    16: "banner",
    17: "billboard",
    18: "lane divider",
    19: "parking sign",
    20: "pole",
    21: "polegroup",
    22: "street light",
    23: "traffic cone",
    24: "traffic device",
    25: "traffic light",
    26: "traffic sign",
    27: "traffic sign frame",
    28: "terrain",
    29: "vegetation",
    30: "sky",
    31: "person",
    32: "rider",
    33: "bicycle",
    34: "bus",
    35: "car",
    36: "caravan",
    37: "motorcycle",
    38: "trailer",
    39: "train",
    40: "truck",
}

anom_panoptic_classes = {
    0: "unlabeled",
    1: "dynamic",
    2: "ego vehicle",
    3: "ground",
    4: "static",
    5: "parking",
    6: "rail track",
    7: "road",
    8: "sidewalk",
    9: "bridge",
    10: "building",
    11: "fence",
    12: "garage",
    13: "guard rail",
    14: "tunnel",
    15: "wall",
    16: "banner",
    17: "billboard",
    18: "lane divider",
    19: "parking sign",
    20: "pole",
    21: "polegroup",
    22: "street light",
    23: "traffic cone",
    24: "traffic device",
    25: "traffic light",
    26: "traffic sign",
    27: "traffic sign frame",
    28: "terrain",
    29: "vegetation",
    30: "sky",
    31: "person",
    32: "rider",
    33: "train",  # 39 -> 33
    34: "bus",
    35: "car",
    36: "caravan",
    37: "truck",  # 40->  37
    38: "trailer",
    # 39: "bicycle",  # 33 -> 39
    # 40: "motorcycle", # 37 -> 40
}


class BDD100KDataModule(LightningDataModule):
    name = "BDD100K"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_dir: str = BDD100K_PATH,
        num_workers: int = 10,
        batch_size: int = 8,
        val_batch_size: int = 2,
        crop_size: Tuple[int, int] = (640, 360),
        base_size: Tuple[int, int] = (640, 360),
        base_size_val: Tuple[int, int] = (640, 360),
        scale_range: Tuple[float, float] = (1.0, 1.0),
        train_size: float = 1.0,
        val_size: float = 1.0,
        test_size: float = 1.0,
        exclude_classes=[],
        target_type="semantic",
        panoptic_preprocessing=None,
        anomaly=False,
        panoptic_args={},
        val_as_test=False,
        persistent_workers=True,
        collate_name="",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if anomaly:
            self.dataset = BDD100KAnomaly
        else:
            self.dataset = BDD100K
        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.crop_size = crop_size
        self.base_size = base_size
        self.base_size_val = base_size_val
        self.train_size = train_size
        self.scale_range = scale_range
        self.val_size = val_size
        self.test_size = test_size

        self.val_as_test = val_as_test

        self.has_valid_mask = False

        self.target_type = target_type

        self.panoptic_preprocessing = panoptic_preprocessing
        self.panoptic_args = panoptic_args

        self.label_divisor = 1000

        self.persistent_workers = persistent_workers
        self.collate_name = collate_name
        self.exclude_classes = exclude_classes
        if anomaly:
            class_map = (
                semseg_classes
                if self.target_type == "semantic"
                else anom_panoptic_classes
            )
            if self.target_type == "semantic":
                del class_map[17]
                del class_map[18]
        else:
            class_map = (
                semseg_classes if self.target_type == "semantic" else panoptic_classes
            )

        self.void_cls = [
            {"id": k, "name": class_map[k]}
            for k in class_map
            if k in exclude_classes or (k == 0 and self.target_type != "semantic")
        ]
        self.ood_cls = [
            {"id": k, "name": class_map[k]} for k in class_map if k in exclude_classes
        ]
        self.valid_cls = [
            {"id": k, "name": class_map[k]}
            for k in class_map
            if k not in exclude_classes and (k != 0 or self.target_type == "semantic")
        ]

        self.from_bdd_id = {ccls["id"]: i for i, ccls in enumerate(self.valid_cls)}

        self.thing_list = [clss["id"] for clss in self.valid_cls if clss["id"] > 30]
        self.mapped_thing_list = [self.from_bdd_id[i] for i in self.thing_list]
        self.stuff_list = [clss["id"] for clss in self.valid_cls if clss["id"] <= 30]
        self.mapped_stuff_list = [self.from_bdd_id[i] for i in self.stuff_list]

        if len(exclude_classes) > 0:
            self.exclude_classes = exclude_classes

            indice_path = "data/indices_classes_bdd100k.pth"
            if self.target_type == "panoptic":
                indice_path = "data/indices_classes_bdd100k_panoptic.pth"

            if os.path.exists(indice_path):
                train_indices = torch.load(indice_path)
                self.train_indices = (
                    (~(train_indices[:, exclude_classes].any(dim=1)))
                    .nonzero()
                    .squeeze()
                )
                excluded_classes = [class_map[ix] for ix in exclude_classes]
                print(
                    f"Excluding classes: {', '.join(excluded_classes)}, resulting in {len(self.train_indices)}/{train_indices.shape[0]} training samples"
                )
            else:
                raise Exception(
                    "Please run 'precompute_class_indices.py' before specifying train classes to exclude"
                )
        else:
            self.train_indices = None

    @property
    def num_classes(self):
        return len(self.valid_cls)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def train_dataloader(
        self,
        shuffle=True,
    ):
        dataset = self.dataset(
            self.data_dir,
            split="train",
            target_type=self.target_type,
            transforms=self.train_transforms,
        )

        if self.train_indices is not None:
            dataset = Subset(dataset, self.train_indices)
        if self.train_size < 1.0:
            train_len = int(len(dataset) * self.train_size)
            dataset = Subset(dataset, range(train_len))

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0) and self.persistent_workers,
            collate_fn=get_collate(self.collate_name),
        )
        return loader

    def val_dataloader(self):
        dataset = self.dataset(
            self.data_dir,
            split="val",
            target_type=self.target_type,
            transforms=self.test_transforms,
        )

        if self.val_size < 1.0:
            val_len = int(len(dataset) * self.val_size)

            sets = random_split(
                dataset,
                [val_len, len(dataset) - val_len],
                torch.Generator().manual_seed(42),
            )
            dataset = sets[0]

        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0) & self.persistent_workers,
            collate_fn=get_collate(self.collate_name),
        )
        return loader

    def anomaly_dataloader(self, debug=False, split="anom_train"):
        lst = [
            tr.PanopticEncodeSegmap(
                [ccls["id"] for ccls in self.void_cls],
                255,
                [ccls["id"] for ccls in self.valid_cls],
                self.from_bdd_id,
                semantic_key="semantic",
            ),
            # tr.PanopticFixedResize(size=self.base_size_val),
            tr.PanopticNormalize(mean=self.mean, std=self.std),
            tr.PanopticToTensor(),
        ]

        composed_transforms = transforms_lib.Compose(lst)
        dataset = self.dataset(
            self.data_dir,
            split=split,
            target_type=self.target_type,
            transforms=composed_transforms,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers if not debug else 0,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0) & self.persistent_workers,
        )
        return loader

    def test_dataloader(self):
        dataset = self.dataset(
            self.data_dir,
            split="val" if self.val_as_test else "test",
            target_type=self.target_type,
            transforms=self.test_transforms,
        )

        if self.test_size < 1.0:
            test_len = int(len(dataset) * self.test_size)

            sets = random_split(
                dataset,
                [test_len, len(dataset) - test_len],
                torch.Generator().manual_seed(42),
            )
            dataset = sets[0]

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0) & self.persistent_workers,
            collate_fn=get_collate(self.collate_name),
        )
        return loader

    @property
    def train_transforms(self):
        lst = [
            tr.PanopticEncodeSegmap(
                [ccls["id"] for ccls in self.void_cls],
                255,
                [ccls["id"] for ccls in self.valid_cls],
                self.from_bdd_id,
                semantic_key="semantic",
            ),
            # tr.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            tr.PanopticRandomHorizontalFlip(),
            tr.PanopticRandomZoomCrop_CategoryAreaConstraint(
                crop_size=self.crop_size,
                base_size=self.base_size,
                scale_range=self.scale_range,
                ignored_category=255,
            ),
            # tr.RandomGaussianBlur(),
            tr.PanopticNormalize(mean=self.mean, std=self.std),
            tr.PanopticToTensor(),
        ]

        if self.panoptic_preprocessing is not None:
            lst.append(self.get_panoptic_preprocessor())

        composed_transforms = transforms_lib.Compose(lst)

        return composed_transforms

    @property
    def test_transforms(self):
        if hasattr(self, "no_resize"):
            lst = [
                tr.PanopticEncodeSegmap(
                    [ccls["id"] for ccls in self.void_cls],
                    255,
                    [ccls["id"] for ccls in self.valid_cls],
                    self.from_bdd_id,
                    ood_classes=[ood_cls["id"] for ood_cls in self.ood_cls],
                    semantic_key="semantic",
                ),
                tr.PanopticNormalize(mean=self.mean, std=self.std),
                tr.PanopticToTensor(),
            ]
            composed_transforms = transforms_lib.Compose(lst)
            return composed_transforms
        lst = [
            tr.PanopticEncodeSegmap(
                [ccls["id"] for ccls in self.void_cls],
                255,
                [ccls["id"] for ccls in self.valid_cls],
                self.from_bdd_id,
                ood_classes=[ood_cls["id"] for ood_cls in self.ood_cls],
                semantic_key="semantic",
            ),
            tr.PanopticFixedResize(size=self.base_size_val),
            tr.PanopticNormalize(mean=self.mean, std=self.std),
            tr.PanopticToTensor(),
        ]

        if self.panoptic_preprocessing is not None:
            lst.append(self.get_panoptic_preprocessor())

        composed_transforms = transforms_lib.Compose(lst)

        return composed_transforms

    def get_panoptic_preprocessor(self):
        if self.panoptic_preprocessing == "deeplab":
            return PanopticTargetGenerator(
                thing_list=self.thing_list,
                label_divisor=self.label_divisor,
                **self.panoptic_args,
            )
        elif self.panoptic_preprocessing == "maxdeeplab":
            return MaxDeeplabTargetGenerator(
                thing_list=self.thing_list,
                to_train_id=self.from_bdd_id,
                label_divisor=self.label_divisor,
                **self.panoptic_args,
            )
        elif self.panoptic_preprocessing == "u3hs":
            return PrototypicalDeeplabTargetGenerator(
                thing_list=self.thing_list,
                stuff_list=self.stuff_list,
                label_divisor=self.label_divisor,
                **self.panoptic_args,
            )
        elif self.panoptic_preprocessing == "mask2former":
            return Mask2FormerPanopticTargetGenerator(
                thing_list=self.thing_list,
                stuff_list=self.stuff_list,
                label_divisor=self.label_divisor,
                from_dataset_id=self.from_bdd_id,
                **self.panoptic_args,
            )
        elif self.panoptic_preprocessing == "mask2former_semantic":
            return Mask2FormerSemanticTargetGenerator(
                thing_list=self.thing_list,
                stuff_list=self.stuff_list,
                label_divisor=self.label_divisor,
                from_dataset_id=self.from_bdd_id,
                **self.panoptic_args,
            )

    def get_image_label_from_batch(self, batch):
        return batch["image"], batch["semantic"]

    def get_image_from_batch(self, batch):
        return batch["image"]

    def get_semantic_from_batch(self, batch):
        return batch["semantic"]

    def get_ood_label_from_batch(self, batch):
        return batch["ood"]

    def get_classnames(self):
        return [ccls["name"] for ccls in self.valid_cls]

    def get_class_colors(self):
        if self.target_type == "semantic":
            return [
                torch.tensor(semantic_colors[ccls["id"]]) / 255
                for ccls in self.valid_cls
            ]


        colormap = plt.get_cmap("gist_rainbow")
        colors = colormap(np.linspace(0, 1, self.num_classes))[:, :3]
        return [torch.tensor(color, dtype=torch.float) for color in colors]

    def get_masks_weight_dict(self, median=0.5, power=2, method=None, **kwargs):
        if method is None:
            return torch.ones(self.num_classes)
        [
            1715.0,
            4913.0,
            1124.0,
            5103.0,
            594.0,
            53.0,
            6057.0,
            4069.0,
            1041.0,
            5502.0,
            1937.0,
            59.0,
            1752.0,
            91.0,
            988.0,
            888.0,
            1176.0,
            427.0,
            251.0,
            5943.0,
            285.0,
            4053.0,
            306.0,
            322.0,
            2836.0,
            4690.0,
            1323.0,
            2400.0,
            5751.0,
            5945.0,
            7559.0,
            44.0,
            0.0,
            1385.0,
            63419.0,
            319.0,
            3342.0,
            86.0,
        ]
        counts = torch.tensor(counts)
        counts_rel = counts / counts.sum()
        weight = 1 / np.power(counts_rel, 1 / power)
        weight = weight / np.median(weight[weight != np.inf])
        weight[weight < median] = median

        return weight
