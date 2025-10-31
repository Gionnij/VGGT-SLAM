import json
import os
import re
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class LostAndFound(VisionDataset):
    def __init__(
        self,
        root: str,
        variant="original",
        split: str = "train",
        label_name: str = "label",
        load_instances: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        return_ood=False,
        subset: int = -1
    ) -> None:
        """
        Initialize the LostAndFound dataset.

        This dataset loader expects a Cityscapes-like directory layout for the
        **Lost and Found** data, with images and coarse labels organized by split
        (e.g., ``train``, ``val``, ``test``). It supports several variants,
        including foggy and Fishyscapes exports.

        Directory conventions (per variant):
          - ``variant="original"``:
              - Images: ``{root}/leftImg8bit/{split}/{city}/{city}_{seq}_leftImg8bit.png``
              - Labels: ``{root}/gtCoarse/{split}/{city}/{city}_{seq}_gtCoarse_labelTrainIds.png``
          - ``variant in {"foggy_0.005","foggy_0.01","foggy_0.02"}``:
              - Images: ``{root}/leftImg8bit_foggy/{split}/{city}/{city}_{seq}_foggy_beta_{beta}.png``
              - Labels: same as ``original``
          - ``variant="fishyscapes"``:
              - Images: ``{root}/fs/images/{city}/{city}_{seq}_leftImg8bit.png``
              - Labels: ``{root}/fs/labels/{id}_{city}_{seq}_labels.png`` (matched internally)

        During initialization the dataset scans the filesystem and builds a list of
        (city, sequence) pairs (or triplets for Fishyscapes) based on the expected
        filename patterns. Optionally, the list can be truncated via ``subset``.

        Parameters
        ----------
        root : str
            Path to the dataset root directory containing the image/label subfolders
            described above.
        variant : {"original", "foggy_0.005", "foggy_0.01", "foggy_0.02", "fishyscapes"}, optional
            Which image variant to load. Determines subdirectories and filename
            suffixes used for discovery and loading. Defaults to ``"original"``.
        split : {"train", "val", "test"}, optional
            Dataset split to use when discovering files. For ``"fishyscapes"``,
            discovery uses the Fishyscapes directories regardless of this value,
            but ``split`` is still stored on the instance. Defaults to ``"train"``.
        label_name : str, optional
            Key under which the label image is returned in each sample dict. Defaults to
            ``"label"``.
        load_instances : bool, optional
            If ``True``, also loads instance ID maps from
            ``*_gtCoarse_instanceIds.png`` (when available) and includes them
            in each sample under the key ``"instance"``. Defaults to ``False``.
        transform : callable, optional
            Per-sample image transform applied to the **image** only (TorchVision
            convention; used if ``transforms`` is not supplied).
        target_transform : callable, optional
            Per-sample transform applied to the **label/target** only (TorchVision
            convention; used if ``transforms`` is not supplied).
        transforms : callable, optional
            A joint transform taking and returning a **dict** with keys
            ``{"image", "label"}`` (and possibly ``"instance"``, ``"ood"``). If
            provided, this is called with the entire sample dict and its return
            value becomes the sample. This takes precedence over ``transform`` and
            ``target_transform``. Defaults to ``None``.
        return_ood : bool, optional
            If ``True``, adds an ``"ood"`` PIL image to each sample where pixels are
            ``True`` (1) wherever the label equals 1 in the trainId map
            (``labelTrainIds``). Defaults to ``False``.
        subset : int, optional
            If positive, only the first ``subset`` discovered samples are kept.
            If zero, the dataset will be empty. **If negative (e.g., the default
            ``-1``), Python slicing semantics apply** and that many samples are
            excluded from the end (e.g., ``-1`` drops the last sample). Defaults
            to ``-1``.

        Raises
        ------
        ValueError
            If ``variant`` is not one of the supported options.

        Notes
        -----
        - Image mode: images are opened with PIL and converted to ``"RGB"``.
        - Labels: opened with PIL without conversion (use their original mode).
        - When ``transforms`` is provided, it receives a dict like
          ``{"image": PIL.Image, "label": PIL.Image, "instance": PIL.Image?, "ood": PIL.Image?}``
          and should return a dict of the same structure.
        - File discovery is performed at construction time.

        Example
        -------
        >>> ds = LostAndFound(
        ...     root="/data/lostandfound",
        ...     variant="foggy_0.01",
        ...     split="test",
        ...     load_instances=True,
        ...     return_ood=True,
        ...     subset=100,
        ... )
        >>> sample = ds[0]
        >>> sample.keys()
        dict_keys(['image', 'label', 'instance', 'ood'])
        """
        super(LostAndFound, self).__init__(
            root, transforms, transform, target_transform
        )
        self.variant = variant
        if self.variant == "original":
            self.intermediat_dir = "leftImg8bit"
            self.ending = "leftImg8bit"
        elif self.variant in ["foggy_0.005", "foggy_0.01", "foggy_0.02"]:
            self.intermediat_dir = f"leftImg8bit_foggy"
            self.ending = f"foggy_beta_{self.variant.split('_')[1]}"
        elif self.variant == "fishyscapes":
            self.intermediat_dir = "fs"
            self.ending = "leftImg8bit"

        else:
            raise ValueError(f"Variant {self.variant} not recognized.")
        
        if split == "main_train":
            split = "train"
        self.targets_dir = os.path.join(self.root, "gtCoarse", split)
        self.images_dir = os.path.join(self.root, self.intermediat_dir, split)
        if self.variant == "fishyscapes":
            self.targets_dir = os.path.join(self.root, "fs", "labels")
            self.images_dir = os.path.join(self.root, "fs", "images")
            self.split = "test"

        self.split = split
        self.label_name = label_name
        self.subset = subset
        self.load_instances = load_instances

        self.return_ood = return_ood
        self.discover()
        self.images = self.images[:self.subset]
        
    def discover(self):
        self.images = []
        for location in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, location)
            print(img_dir)
            for file_name in os.listdir(img_dir):
                if not file_name.endswith(f"{self.ending}.png"):
                    continue
                match = re.match(
                    f"([0-9]+_[0-9]+)_{self.ending}.png", file_name[len(location) + 1 :]
                )
                number = match[1]
                self.images.append((location, number))

        if self.variant == "fishyscapes":
            targets = os.listdir(self.targets_dir)

            targets_splitted = []
            target_map = {}
            for target in targets:
                targets_splitted.append((target[:4], target[5:]))
                target_map[target[5:]] = target[:4]

            filtered_images = []
            for loc, num in self.images:
                key = f"{loc}_{num}_labels.png"
                if key in target_map:
                    filtered_images.append((loc, num, target_map[key]))
            self.images = filtered_images
        
    def discover_foggy(self):
        pass
        
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        if self.variant in ["original", "foggy_0.01", "foggy_0.02", "foggy_0.005"]:
            location, number = self.images[index]
            target_number = None
        else:
            location, number, target_number = self.images[index]

        img_path = os.path.join(
            self.images_dir, location, f"{location}_{number}_{self.ending}.png"
        )
        image = Image.open(img_path).convert("RGB")

        if target_number is None:
            target_path = os.path.join(
                self.targets_dir,
                location,
                f"{location}_{number}_gtCoarse_labelTrainIds.png",
            )
        else:
            target_path = os.path.join(
                self.targets_dir,
                f"{target_number}_{location}_{number}_labels.png",
            )
        target = Image.open(target_path)

        result = {"image": image, self.label_name: target}

        if self.return_ood:
            result["ood"] = Image.fromarray(np.array(target) == 1)

        if self.load_instances:
            instance_path = os.path.join(
                self.targets_dir,
                location,
                f"{location}_{number}_gtCoarse_instanceIds.png",
            )
            instance = Image.open(instance_path)
            result["instance"] = instance

        if self.transforms is not None:
            result = self.transforms(result)

        return result

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return "\n".join(lines).format(**self.__dict__)
