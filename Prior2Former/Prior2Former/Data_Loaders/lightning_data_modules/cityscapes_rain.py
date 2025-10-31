import os
from typing import Tuple

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
)

from .transforms import transforms as tr
from Data_Loaders.SemanticSegmentation.cityscapes_rain import CityscapesOOD

CITYSCAPES_PATH = ""
valid_classes = [ccls for ccls in Cityscapes.classes if not ccls.ignore_in_eval]
void_classes = [ccls for ccls in Cityscapes.classes if ccls.ignore_in_eval]
to_cityscapes_id = [ccls.id for ccls in valid_classes]
from_cityscapes_id = {ccls.id: i for i, ccls in enumerate(valid_classes)}


class CityscapesOODDataModule(LightningDataModule):
    name = "cityscapes_ood"
    mean = [0.485, 0.456, 0.406]  # These are the ones from original cityscapes
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
        scale_range: Tuple[int, int] = [1, 1],
        train_size: float = 1.0,
        val_size: float = 1.0,
        test_size: float = 1.0,
        exclude_classes=[],
        target_type="semantic",
        panoptic_preprocessing=None,
        panoptic_args={},
        val_as_test=False,
        persistent_workers=True,
        normal_cityscapes_dir=None,
        pattern=-1,
        flip=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pattern = pattern
        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.normal_cityscapes_dir = normal_cityscapes_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.flip = flip
        self.crop_size = crop_size
        self.base_size = base_size
        self.base_size_val = base_size_val
        self.scale_range = scale_range
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
                print(
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
        self.mapped_thing_list = [self.from_cityscapes_id[i] for i in self.thing_list]
        self.stuff_list = [
            clss.id
            for i, clss in enumerate(valid_classes)
            if not clss.has_instances and i not in exclude_classes
        ]
        self.mapped_stuff_list = [self.from_cityscapes_id[i] for i in self.stuff_list]

        tmp_1 = CityscapesOOD(
            self.data_dir,
            split="train",
            mode="fine",
            target_type=self.target_type,
            transforms=self.train_transforms,
            root_normal_cityscapes=self.normal_cityscapes_dir,
            pattern=self.pattern,
        )

        tmp_2 = tmp_1[0]
        print(tmp_2)

    @property
    def num_classes(self):
        return len(valid_classes) - len(self.exclude_classes)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def train_dataloader(
        self,
        shuffle=True,
    ):
        dataset = CityscapesOOD(
            self.data_dir,
            split="train",
            mode="fine",
            target_type=self.target_type,
            transforms=self.train_transforms,
            root_normal_cityscapes=self.normal_cityscapes_dir,
            pattern=self.pattern,
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
        )
        return loader

    def val_dataloader(self, subset=None):
        dataset = CityscapesOOD(
            self.data_dir,
            split="val",
            mode="fine",
            target_type=self.target_type,
            transforms=self.test_transforms,
            root_normal_cityscapes=self.normal_cityscapes_dir,
            pattern=self.pattern,
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
        )
        return loader

    def test_dataloader(self):
        dataset = CityscapesOOD(
            self.data_dir,
            split="val" if self.val_as_test else "test",
            mode="fine",
            target_type=self.target_type,
            transforms=self.train_transforms,
            root_normal_cityscapes=self.normal_cityscapes_dir,
            pattern=self.pattern,
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
            )]
        if self.flip:
            lst.append(
                tr.PanopticRandomHorizontalFlip(),
            )
            # tr.PanopticColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        lst.append(tr.PanopticRandomZoomCrop(
            crop_size=self.crop_size,
            base_size=self.base_size,
            scale_range=self.scale_range,
        ))
        # tr.PanopticRandomGaussianBlur(),
        lst.append(tr.PanopticNormalize(mean=self.mean, std=self.std))
        lst.append(tr.PanopticToTensor())
        

        if self.panoptic_preprocessing is not None:
            lst.append(self.get_panoptic_preprocessor())

        composed_transforms = transforms_lib.Compose(lst)

        return composed_transforms

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
