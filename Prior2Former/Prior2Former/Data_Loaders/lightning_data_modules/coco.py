import os
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, random_split
from torchvision import transforms as transforms_lib

from Data_Loaders.PanopticSegmentation.coco import (
    CocoDataset,
    CocoDatasetAnomaly,
)
from Data_Loaders.lightning_data_modules.transforms.panoptic import (
    MaxDeeplabTargetGenerator,
    PanopticTargetGenerator,
    PrototypicalDeeplabTargetGenerator,
    Mask2FormerPanopticTargetGenerator,
    Mask2FormerSemanticTargetGenerator,
)
from utils.logging_utils.log_writers import gl_info

from .transforms import transforms as tr
from .utils import get_collate
from Data_Loaders.PanopticSegmentation.coco_utils import COCO_CATEGORIES, COCO_CATEGORIES_WITHOUT_UNKNOWN

COCO_PATH = ""




class CocoDataModule(LightningDataModule):
    name = "coco"
    mean = [0.4850, 0.4560, 0.4060]
    std = [0.2290, 0.2240, 0.2250]

    def __init__(
        self,
        data_dir: str = COCO_PATH,
        num_workers: int = 10,
        batch_size: int = 8,
        val_batch_size: int = 2,
        test_batch_size: int = 1,
        crop_size: Tuple[int, int] = (512, 512),
        base_size: Tuple[int, int] = (512, 512),
        base_size_val: Tuple[int, int] = (512, 512),
        scale_range: Tuple[int, int] = (1.0, 1.0),
        target_shape: Tuple[int, int] = (1024, 1024),
        train_size: float = 1.0,
        val_size: float = 1.0,
        test_size: float = 1.0,
        single_category_max_area: float = 1.0,
        exclude_classes=[],
        target_type="semantic",
        anomaly: bool = False,
        panoptic_preprocessing=None,
        panoptic_args={},
        val_as_test=False,
        persistent_workers=True,
        collate_name="",
        flip: bool = True,
        keep_ar: bool = True,
        fixed_size: Tuple[int, int] = (758, 512),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.keep_ar = keep_ar
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.flip = flip
        self.crop_size = crop_size
        self.base_size = base_size
        self.base_size_val = base_size_val
        self.fixed_size = fixed_size
        self.scale_range = scale_range
        self.target_shape = target_shape
        self.single_category_max_area = single_category_max_area
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.anomaly = anomaly
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
            classes = [SimpleNamespace(**cat) for cat in COCO_CATEGORIES_WITHOUT_UNKNOWN]
            gl_info(f"Using reduced classes: {len(classes)}")
        else:
            classes = [SimpleNamespace(**cat) for cat in COCO_CATEGORIES]
            gl_info(f"Using all classes: {len(classes)}")
        self.valid_classes = [ccls for ccls in classes]

        self.void_classes = []


        self.to_coco_id = [ccls.id for ccls in self.valid_classes]
        self.from_coco_id = {ccls.id: i for i, ccls in enumerate(self.valid_classes)}

        if len(exclude_classes) > 0:
            self.exclude_classes = exclude_classes
            if os.path.exists("data/indices_classes_coco.pth"):
                train_indices = torch.load("data/indices_classes_coco.pth")
                self.train_indices = (
                    (~(train_indices[:, exclude_classes].any(dim=1)))
                    .nonzero()
                    .squeeze()
                )
                excluded_classes = [self.valid_classes[ix].name for ix in exclude_classes]
                gl_info(
                    f"Excluding classes: {', '.join(excluded_classes)}, resulting in {len(self.train_indices)}/{train_indices.shape[0]} training samples"
                )
            else:
                raise Exception(
                    "Please run 'precompute_class_indices.py' before specifying train classes to exclude"
                )
        else:
            self.train_indices = None

        self.val_cls = self.valid_classes
        self.void_cls = self.void_classes
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
                *[self.valid_classes[i] for i in self.exclude_classes],
            ]

        self.from_coco_id = {ccls.id: i for i, ccls in enumerate(self.val_cls)}
        self.to_coco_id = [ccls.id for ccls in self.val_cls]

        self.thing_list = [
            clss.id
            for i, clss in enumerate(self.valid_classes)
            if clss.isthing and i not in exclude_classes
        ]

        #  should be train_id -> id to contiguous id
        self.train_id_thing_list = [
            clss.id
            for i, clss in enumerate(self.valid_classes)
            if clss.isthing and i not in exclude_classes
        ]
        self.mapped_thing_list = [self.from_coco_id[i] for i in self.thing_list]
        self.stuff_list = [
            clss.id
            for i, clss in enumerate(self.valid_classes)
            if not clss.isthing and i not in exclude_classes
        ]
        self.mapped_stuff_list = [self.from_coco_id[i] for i in self.stuff_list]

        tmp_1 = CocoDataset(
            self.data_dir,
            split="train",
            anomaly=self.anomaly,
            target_type=self.target_type,
            transforms=self.train_transforms,
        )

        tmp_2 = tmp_1[0]
        gl_info(tmp_2)

    @property
    def num_classes(self):
        return len(self.valid_classes) - len(self.exclude_classes)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_dataset(self, **kwargs):
        return CocoDataset(**kwargs)

    def train_dataloader(
        self,
        shuffle=True,
    ):
        dataset = CocoDataset(
            self.data_dir,
            split="train",
            anomaly=self.anomaly,
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

    def val_dataloader(self, subset=None, split="val"):
        dataset = CocoDataset(
            self.data_dir,
            split=split,
            anomaly=self.anomaly,
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

    def anomaly_dataloader(self, subset=None, split="val", resize=False, debug=False,known_set="full"):
        # only normalize and toTensor transform
        # transform = transforms_lib.Compose(self.test_transforms.transforms[:-1])
        dataset = CocoDatasetAnomaly(
            self.data_dir,
            split=split,
            known_set=known_set,
            target_type=self.target_type,
            transforms=self.test_transforms,
        )

        if subset is not None:
            dataset = Subset(dataset, subset)

        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers if not debug else 1,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0) & self.persistent_workers,
            collate_fn=get_collate(self.collate_name),
        )
        return loader

    def val_dataloader_full(self, subset=None):
        dataset = CocoDataset(
            self.data_dir,
            split="full_val",
            anomaly=self.anomaly,
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
        dataset = CocoDataset(
            self.data_dir,
            split="val" if self.val_as_test else "test",
            anomaly=self.anomaly,
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
                self.from_coco_id,
                semantic_key="semantic",
            )]
        if self.flip:
            lst.append(
                tr.PanopticRandomHorizontalFlip(),
            )
        if self.keep_ar:
            lst.extend([
                tr.PanopticResizeScale(
                    self.scale_range, target_shape=self.target_shape
                ),
                tr.PanopticFixedSizeCrop(crop_size=self.crop_size),
                ]
            )
        elif self.keep_ar and len(self.fixed_size) == 1:
            lst.extend(
                [
                tr.PanopticRandomZoomCrop_CategoryAreaConstraint(
                    crop_size=None,
                    base_size=None,
                    scale_range=self.scale_range,
                    single_category_max_area=1.0,
                    ignored_category=255,
                )
            ]
            )
        else:
            lst.extend(
                [
                #tr.PanopticFixedResize(size=self.fixed_size),
                tr.PanopticRandomZoomCrop_CategoryAreaConstraint(
                    crop_size=self.crop_size,
                    base_size=self.base_size,
                    scale_range=self.scale_range,
                    single_category_max_area=1.0,
                    ignored_category=255,
                )
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

    @property
    def test_transforms(self):
        if hasattr(self, "no_resize"):
            lst = [
                tr.PanopticEncodeSegmap(
                    [ccls.id for ccls in self.void_cls],
                    255,
                    [ccls.id for ccls in self.val_cls],
                    self.from_coco_id,
                    ood_classes=[self.valid_classes[i].id for i in self.exclude_classes],
                    semantic_key="semantic",
                ),
                tr.PanopticNormalize(mean=self.mean, std=self.std),
                tr.PanopticToTensor(),
            ]
            composed_transforms = transforms_lib.Compose(lst)
            return composed_transforms

        lst = [
            tr.PanopticEncodeSegmap(
                [ccls.id for ccls in self.void_cls],
                255,
                [ccls.id for ccls in self.val_cls],
                self.from_coco_id,
                ood_classes=[self.valid_classes[i].id for i in self.exclude_classes],
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
                to_train_id=self.from_coco_id,
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
                from_dataset_id=self.from_coco_id,
                label_divisor=self.label_divisor,
                **self.panoptic_args,
            )
        elif self.panoptic_preprocessing == "mask2former_semantic":
            return Mask2FormerSemanticTargetGenerator(
                thing_list=self.thing_list,
                stuff_list=self.stuff_list,
                from_dataset_id=self.from_coco_id,
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
        return [ccls.name for ccls in self.valid_classes]

    def get_class_colors(self):
        return [torch.tensor(ccls.color) / 255 for ccls in self.val_cls]

    def get_masks_weight_dict(self, median=0.2, power=1.2, method=None, **kwargs):
        if method is None:
            return torch.ones(self.num_classes)
        counts = [
            216252.0,
            6379.0,
            34218.0,
            8371.0,
            5101.0,
            5620.0,
            4464.0,
            8239.0,
            10582.0,
            11263.0,
            8303.0,
            10448.0,
            4195.0,
            4504.0,
            6478.0,
            9386.0,
            8057.0,
            5479.0,
            5297.0,
            5119.0,
            7395.0,
            10964.0,
            10980.0,
            6217.0,
            5943.0,
            5469.0,
            4954.0,
            8986.0,
            1385.0,
            5428.0,
            6038.0,
            4777.0,
            15793.0,
            7058.0,
            15432.0,
            5052.0,
            5867.0,
            4854.0,
            11194.0,
            9143.0,
            5380.0,
            3769.0,
            5954.0,
            7158.0,
            7414.0,
            5498.0,
            6723.0,
            5787.0,
            30251.0,
            4817.0,
            7327.0,
            3886.0,
            13595.0,
            3873.0,
            3436.0,
            2747.0,
            5005.0,
            5377.0,
            1557.0,
            3637.0,
            18287.0,
            5789.0,
            5845.0,
            4555.0,
            3349.0,
            2375.0,
            1561.0,
            2784.0,
            2678.0,
            4207.0,
            7516.0,
            4960.0,
            2849.0,
            1966.0,
            2349.0,
            5326.0,
            9504.0,
            2991.0,
            1153.0,
            1796.0,
            1949.0,
            3384.0,
            2659.0,
            2120.0,
            12736.0,
            3891.0,
            4165.0,
            6335.0,
            3359.0,
            3361.0,
            2208.0,
            1254.0,
            1900.0,
            4344.0,
            1771.0,
            3884.0,
            5416.0,
            2098.0,
            1736.0,
            11498.0,
            34607.0,
            12139.0,
            8735.0,
            35010.0,
            4749.0,
            14180.0,
            12514.0,
            16081.0,
            6136.0,
            19594.0,
            8590.0,
            7746.0,
            6707.0,
            20295.0,
            4022.0,
            34775.0,
            6052.0,
        ]
        counts = torch.tensor(counts)
        counts_rel = counts / counts.sum()
        weight = 1 / np.power(counts_rel, 1 / power)
        weight = weight / np.median(weight[weight != np.inf])
        weight[weight < median] = median

        return weight
