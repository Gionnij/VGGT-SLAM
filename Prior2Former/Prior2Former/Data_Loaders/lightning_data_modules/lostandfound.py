from typing import Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms as transforms_lib

from Data_Loaders.SemanticSegmentation.lostandfound import LostAndFound
from Data_Loaders.lightning_data_modules.transforms.panoptic import (
    PanopticTargetGenerator,
    PrototypicalDeeplabTargetGenerator,
)

from .transforms import transforms as tr
LOSTANDFOUND_PATH=""

class LostAndFoundDataModule(LightningDataModule):
    name = "lostandfound"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_dir: str = LOSTANDFOUND_PATH,
        num_workers: int = 10,
        batch_size: int = 8,
        val_batch_size: int = 2,
        crop_size: Tuple[int, int] = (512, 256),
        base_size: Tuple[int, int] = (512, 256),
        train_size: float = 1.0,
        test_size: float = 1.0,
        variant="original",
        dataset_args={},
        panoptic_preprocessing=None,
        panoptic_args={},
        subset: int = -1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.crop_size = crop_size
        self.base_size = base_size
        self.train_size = train_size
        self.test_size = test_size

        self.has_valid_mask = False

        self.variant = variant
        if self.variant == "original":
            self.void_classes = [0]
            self.valid_classes = [1, 2]
            self.class_map = {1: 0, 2: 1}
            self.thing_list = [2]
            self.stuff_list = [1]
            self.label_divisor = 1000
            self.mapped_thing_list = [1]
        else:
            self.void_classes = [255]
            self.valid_classes = [0, 1]
            self.class_map = {0: 0, 1: 1}

        self.dataset_args = dataset_args
        self.panoptic_preprocessing = panoptic_preprocessing
        self.panoptic_args = panoptic_args

    @property
    def num_classes(self):
        return 2

    def train_dataloader(
        self,
        shuffle=True,
    ):
        dataset = LostAndFound(
            self.data_dir,
            split="train",
            transforms=self.train_transforms,
            variant=self.variant,
            **self.dataset_args,
        )

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
        )
        return loader

    def val_dataloader(self):
        raise Exception("The lost and found dataset does not offer a validation split")

    def unnormalize(self, image):
        return image * torch.tensor([self.std]).reshape((1, 3, 1, 1)) + torch.tensor(
            [self.mean]
        ).reshape((1, 3, 1, 1))

    def test_dataloader(self, subset=None, number_of_elements = -1, shuffle=False):
        dataset = LostAndFound(
            self.data_dir,
            split="test",
            transforms=self.test_transforms,
            variant=self.variant,
            subset=number_of_elements,
            **self.dataset_args,
        )

        if self.test_size < 1.0:
            test_len = int(len(dataset) * self.test_size)
            dataset = Subset(dataset, range(test_len))

        if subset is not None:
            dataset = Subset(dataset, subset)

        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    @property
    def train_transforms(self):
        lst = [
            tr.EncodeSegmap(
                self.void_classes,
                255,
                self.valid_classes,
                self.class_map,
            ),
            # tr.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            tr.RandomHorizontalFlip(),
            tr.RandomCrop(crop_size=self.crop_size, base_size=self.base_size),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor(),
        ]

        if self.panoptic_preprocessing is not None:
            lst.append(self.get_panoptic_preprocessor())

        return transforms_lib.Compose(lst)

    @property
    def test_transforms(self):
        lst = [
            tr.EncodeSegmap(
                self.void_classes,
                255,
                self.valid_classes,
                self.class_map,
            ),
            tr.FixedResize(size=self.base_size),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor(),
        ]

        if self.panoptic_preprocessing is not None:
            lst.append(self.get_panoptic_preprocessor())

        return transforms_lib.Compose(lst)

    def get_panoptic_preprocessor(self):
        if self.panoptic_preprocessing == "deeplab":
            return PanopticTargetGenerator(
                thing_list=self.thing_list,
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
        return batch["image"], batch["label"]

    def get_image_from_batch(self, batch):
        return batch["image"]

    def get_semantic_from_batch(self, batch):
        return batch["label"]

    def get_ood_mask(self, batch):
        return batch["label"] == 1

    def get_classnames(self):
        return ["In-distribution", "Out-of-distribution"]

    def get_class_colors(self):
        return [torch.tensor([0.1, 1.0, 0.1]), torch.tensor([1.0, 0.1, 0.1])]
