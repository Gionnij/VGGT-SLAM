import os
from typing import Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, random_split
from torchvision import transforms as transforms_lib

from Data_Loaders.SemanticSegmentation.cityscapes import (
    Cityscapes,
)
from Data_Loaders.lightning_data_modules.transforms.panoptic import (
    MaxDeeplabTargetGenerator,
    PanopticTargetGenerator,
    PrototypicalDeeplabTargetGenerator,
    Mask2FormerPanopticTargetGenerator,
    Mask2FormerSemanticTargetGenerator,
)
from Pytorch_Extentions import transforms as tr
from .transforms import transforms as tr
from .utils import get_collate
from utils.logging_utils.log_writers import gl_info


CITYSCAPES_PATH = ""
valid_classes = [ccls for ccls in Cityscapes.classes if not ccls.ignore_in_eval]
void_classes = [ccls for ccls in Cityscapes.classes if ccls.ignore_in_eval]
to_cityscapes_id = [ccls.id for ccls in valid_classes]
from_cityscapes_id = {ccls.id: i for i, ccls in enumerate(valid_classes)}


class CityscapesDataModule(LightningDataModule):
    name = "cityscapes"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_dir: str = CITYSCAPES_PATH,
        num_workers: int = 10,
        batch_size: int = 8,
        val_batch_size: int = 2,
        test_batch_size: int = 1,
        crop_size: Tuple[int, int] = (512, 256),
        base_size: Tuple[int, int] = (512, 256),
        base_size_val: Tuple[int, int] = (512, 256),
        scale_range: Tuple[int, int] = (1.0, 1.0),
        train_size: float = 1.0,
        val_size: float = 1.0,
        test_size: float = 1.0,
        single_category_max_area: float = 1.0,
        exclude_classes=[],
        target_type="semantic",
        panoptic_preprocessing=None,
        panoptic_args={},
        val_as_test=False,
        persistent_workers=True,
        collate_name="",
        flip=True,
        random_blur=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.flip = flip
        self.random_blur=random_blur
        self.crop_size = crop_size
        self.base_size = base_size
        self.base_size_val = base_size_val
        self.scale_range = scale_range
        self.single_category_max_area = single_category_max_area
        self.train_size = train_size
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
        if len(exclude_classes) > 0:
            self.exclude_classes = exclude_classes
            if os.path.exists("data/indices_classes_cityscapes.pth"):
                train_indices = torch.load("data/indices_classes_cityscapes.pth")
                self.train_indices = (
                    (~(train_indices[:, exclude_classes].any(dim=1)))
                    .nonzero()
                    .squeeze()
                )
                excluded_classes = [valid_classes[ix].name for ix in exclude_classes]
                gl_info(
                    f"Excluding classes: {', '.join(excluded_classes)}, resulting in {len(self.train_indices)}/{train_indices.shape[0]} training samples"
                )
            else:
                raise Exception(
                    "Please run 'precompute_class_indices.py' before specifying train classes to exclude"
                )
        else:
            self.train_indices = None

        self.val_cls = valid_classes
        self.void_cls = void_classes
        if len(self.exclude_classes) > 0:
            self.val_cls = [
                clss
                for i, clss in enumerate(self.val_cls)
                if i not in self.exclude_classes
            ]
            print(
                "Valid classes: ",
                [(clss.name, i) for i, clss in enumerate(self.val_cls)],
            )
            self.void_cls = [
                *self.void_cls,
                *[valid_classes[i] for i in self.exclude_classes],
            ]

        self.from_cityscapes_id = {ccls.id: i for i, ccls in enumerate(self.val_cls)}

        self.thing_list = [
            clss.id
            for i, clss in enumerate(valid_classes)
            if clss.has_instances and i not in exclude_classes
        ]
        self.train_id_thing_list = [
            clss.train_id
            for i, clss in enumerate(valid_classes)
            if clss.has_instances and i not in exclude_classes
        ]
        self.mapped_thing_list = [self.from_cityscapes_id[i] for i in self.thing_list]
        self.stuff_list = [
            clss.id
            for i, clss in enumerate(valid_classes)
            if not clss.has_instances and i not in exclude_classes
        ]
        self.mapped_stuff_list = [self.from_cityscapes_id[i] for i in self.stuff_list]

        gl_info(
            f"Data directory: {self.data_dir}"
        )
       

    @property
    def num_classes(self):
        return len(valid_classes) - len(self.exclude_classes)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_dataset(self, **kwargs):
        return Cityscapes(**kwargs)

    def train_dataloader(
        self,
        shuffle=True,
    ):
        dataset = Cityscapes(
            self.data_dir,
            split="train",
            mode="fine",
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

    def val_dataloader(self, subset=None, number_of_elements = -1):
        dataset = Cityscapes(
            self.data_dir,
            split="val",
            mode="fine",
            target_type=self.target_type,
            transforms=self.test_transforms,
            subset = number_of_elements
        )

        if self.val_size < 1.0:
            val_len = int(len(dataset) * self.val_size)

            sets = random_split(
                dataset,
                [val_len, len(dataset) - val_len],
                torch.Generator().manual_seed(42),
            )
            dataset = sets[0]

        if subset is not None:
            dataset = Subset(dataset, subset)

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

    def test_dataloader(self):
        dataset = Cityscapes(
            self.data_dir,
            split="val" if self.val_as_test else "test",
            mode="fine",
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
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0) & self.persistent_workers,
            collate_fn=get_collate(self.collate_name),
        )
        return loader

    def unnormalize(self, image):
        return image * torch.tensor([self.std]).reshape((1, 3, 1, 1)) + torch.tensor(
            [self.mean]
        ).reshape((1, 3, 1, 1))

    @property
    def train_transforms(self):
        lst = [
            tr.PanopticEncodeSegmap(
                [ccls.id for ccls in self.void_cls],
                255,
                [ccls.id for ccls in self.val_cls],
                self.from_cityscapes_id,
                semantic_key="semantic",
            ),
            tr.PanopticColorAugSSDTransform(img_format="RGB"),
            # tr.PanopticColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            tr.PanopticRandomZoomCrop_CategoryAreaConstraint(
                crop_size=self.crop_size,
                base_size=self.base_size,
                scale_range=self.scale_range,
                single_category_max_area=1.0,
                ignored_category=255,
            ),
            # tr.PanopticRandomGaussianBlur(),
        ]
        if self.flip:
            lst.append(
                tr.PanopticRandomHorizontalFlip(),
            )
        if self.random_blur:
            lst.append([
                tr.PanopticRandomGaussianBlur(),
                tr.RandomAdjustSharpness(sharpness_factor=2)
                ]
            )
        lst.append(
            tr.PanopticNormalize(mean=self.mean, std=self.std),
        )
        lst.append(
            tr.PanopticToTensor(),
        )

        if self.panoptic_preprocessing is not None:
            lst.append(self.get_panoptic_preprocessor())

        composed_transforms = transforms_lib.Compose(lst)

        return composed_transforms

    @train_transforms.setter
    def train_transforms(self, transforms):
        self._custom_train_transforms = transforms

    @property
    def test_transforms(self):
        lst = [
            tr.PanopticEncodeSegmap(
                [ccls.id for ccls in self.void_cls],
                255,
                [ccls.id for ccls in self.val_cls],
                self.from_cityscapes_id,
                ood_classes=[valid_classes[i].id for i in self.exclude_classes],
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
                to_train_id=self.from_cityscapes_id,
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
                from_dataset_id=self.from_cityscapes_id,
                label_divisor=self.label_divisor,
                **self.panoptic_args,
            )
        elif self.panoptic_preprocessing == "mask2former_semantic":
            return Mask2FormerSemanticTargetGenerator(
                thing_list=self.thing_list,
                stuff_list=self.stuff_list,
                from_dataset_id=self.from_cityscapes_id,
                label_divisor=self.label_divisor,
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

    def get_ood_mask(self, batch):
        if "ood" in batch:
            return batch["ood"]
        return None

    def get_classnames(self):
        return [ccls.name for ccls in valid_classes]

    def get_class_colors(self):
        return [torch.tensor(ccls.color) / 255 for ccls in self.val_cls]

    def get_instance_counts(self):
        """{'bicycle': 3729, 'bicyclegroup': 816, 'bridge': 392, 'building': 7141, 'bus': 385, 'car': 27155, 'caravan': 61,
         'cargroup': 1895, 'dynamic': 3491, 'ego vehicle': 2975, 'fence': 2467, 'ground': 1839, 'guard rail': 78,
         'license plate': 5424, 'motorcycle': 739, 'motorcyclegroup': 9, 'out of roi': 2975, 'parking': 1064,
         'person': 17994, 'persongroup': 927, 'pole': 42904, 'polegroup': 309, 'rail track': 112,
         'rectification border': 4392, 'rider': 1807, 'ridergroup': 11, 'road': 3118, 'sidewalk': 7132, 'sky': 2943,
         'static': 38430, 'terrain': 4457, 'traffic light': 10237, 'traffic sign': 20868, 'trailer': 76, 'train': 171,
         'truck': 489, 'truckgroup': 1, 'tunnel': 29, 'vegetation': 15022, 'wall': 1626}"""
        counts_in_cstrain={0: 2934, 1: 2811, 2: 2934, 3: 969, 4: 1296, 5: 2947, 6: 1658, 7: 2808, 8: 2891, 9: 1653, 10: 2685, 11: 18376,
         12: 1761, 13: 27895, 14: 482, 15: 379, 16: 168, 17: 741, 18: 4151}
        return counts_in_cstrain


    def get_masks_weight_dict(self, method=None, power=2, **kwargs):
        if method == None:
            return {id: 1 for id in range(34)}

        # train_id: mask_count, only for panoptic segmentation
        counts_d = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 2934,
            8: 2811,
            9: 0,
            10: 0,
            11: 2934,
            12: 969,
            13: 1296,
            14: 0,
            15: 0,
            16: 0,
            17: 2947,
            18: 0,
            19: 1658,
            20: 2808,
            21: 2891,
            22: 1653,
            23: 2685,
            24: 18376,
            25: 1761,
            26: 27895,
            27: 482,
            28: 379,
            29: 0,
            30: 0,
            31: 168,
            32: 741,
            33: 4151,
        }
        ids = [cls.id for cls in Cityscapes.classes if cls.id in counts_d]
        counts = np.array(
            [counts_d[cls.id] for cls in Cityscapes.classes if cls.id in counts_d]
        )

        if method == "power":
            s = counts.sum()
            counts = counts / s
            weight = 1 / np.power(counts, 1 / power)
            weight = weight / np.median(weight[weight != np.inf])
            weight[weight < 0.5] = 0.5
            return {
                from_cityscapes_id[id]: w
                for id, w in zip(ids, weight)
                if id in from_cityscapes_id
            }

        else:
            raise NotImplementedError(f"weight dict method of type {method} is unkown")
