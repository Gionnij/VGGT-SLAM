import numbers
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms as transforms_lib
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# In this file the test image augmentations are modified in a way that a key is modified as image, this
# makes it possible that the datamodules treat two keys the same so that the ood image ,i.e. rainy, and the unaltered image are treated the same
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample, keys=None):
        if keys is None:
            keys = ["image", "image_cityscapes"]
        if "image" not in keys:
            raise ValueError("'keys' must contain 'image'")
        rest = {}

        for key in keys:
            img = sample[key]

            img = np.array(img).astype(np.float32)
            img /= 255.0
            img -= self.mean
            img /= self.std
            rest[key] = img

        for k in sample.keys():
            if k not in keys:
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = np.array(sample[k]).astype(np.float32)
                else:
                    rest[k] = sample[k]

        return rest


class EncodeSegmap(object):
    """Encode the segmentation image"""

    def __init__(
        self,
        void_classes: List[int],
        ignore_index: int,
        valid_classes: List[int],
        class_map: Dict[int, int],
        ood_classes: Optional[List[int]] = None,
        semantic_key="label",
    ):
        self.void_classes = void_classes
        self.ignore_index = ignore_index
        self.valid_classes = valid_classes
        self.class_map = class_map
        self.ood_classes = ood_classes
        self.semantic_key = semantic_key

    def __call__(self, sample, key="image"):
        img = sample[key]
        mask = sample[self.semantic_key]

        # original_mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        try:
            original_mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        except Exception as e:
            print(
                e
            )  # Maybe a race condition happens here. This avoids the buffer exception.
            original_mask = Image.fromarray(np.array(mask, dtype=np.uint8))

        mask = np.array(mask, dtype=np.uint8)
        mask_copy = None
        if self.ood_classes is not None and len(self.ood_classes) > 0:
            mask_copy = np.zeros_like(mask)
            for ood_cls in self.ood_classes:
                mask_copy[mask == ood_cls] = 1

        mask_u = mask.copy()
        for _voidc in self.void_classes:
            mask_u[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask_u[mask == _validc] = self.class_map[_validc]
        mask = Image.fromarray(mask_u)

        rest = {}
        for k in sample.keys():
            if k != key and k != self.semantic_key:
                rest[k] = sample[k]

        if self.ood_classes is not None and len(self.ood_classes) > 0:
            return {
                key: img,
                self.semantic_key: mask,
                "ood": Image.fromarray(mask_copy),
                f"{self.semantic_key}_original": original_mask,
                **rest,
            }
        else:
            return {
                key: img,
                self.semantic_key: mask,
                f"{self.semantic_key}_original": original_mask,
                **rest,
            }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, keys=None):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        ret = {}
        if keys is None:
            keys = ["image", "image_cityscapes"]
        if "image" not in keys:
            raise ValueError("'keys' must contain 'image'")
        for k in keys:
            img = sample[k]
            img = np.array(img).astype(np.float32).transpose((2, 0, 1))
            img = torch.from_numpy(img).float()
            ret[k] = img

        for k in sample.keys():
            if k not in keys:
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    ret[k] = torch.from_numpy(np.array(sample[k])).long()
                else:
                    ret[k] = sample[k]

        return ret


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample["image"]

        flip = random.random() < 0.5

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        rest = {}
        for k in sample.keys():
            if k != "image":
                if flip and isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = sample[k].transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample["image"]
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)

        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = sample[k].rotate(rotate_degree, Image.NEAREST)
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample["image"]
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        rest = {}
        for k in sample.keys():
            if k != "image":
                rest[k] = sample[k]

        return {"image": img, **rest}


class RandomScaleCrop(object):
    def __init__(self, crop_size: Tuple[int, int], base_size: Tuple[int, int]):
        self.crop_size = crop_size
        self.base_size = base_size

    def __call__(self, sample):
        img = sample["image"]

        img = img.resize(self.base_size, Image.BILINEAR)

        # random scale (short edge)
        scale_factor = random.random() / 2 + 0.5
        w, h = img.size
        ow = int(w * scale_factor)
        oh = int(h * scale_factor)

        if ow < self.crop_size[0]:
            scale_factor = self.crop_size[0] / w
            ow = int(w * scale_factor)
            oh = int(h * scale_factor)

        if oh < self.crop_size[1]:
            scale_factor = self.crop_size[1] / h
            ow = int(w * scale_factor)
            oh = int(h * scale_factor)

        img = img.resize((ow, oh), Image.BILINEAR)

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = (
                        sample[k]
                        .resize(self.base_size, Image.NEAREST)
                        .resize((ow, oh), Image.NEAREST)
                        .crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
                    )
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class RandomCrop(object):
    def __init__(self, crop_size: Tuple[int, int], base_size: Tuple[int, int]):
        self.crop_size = crop_size
        self.base_size = base_size

    def __call__(self, sample):
        img = sample["image"]

        img = img.resize(self.base_size, Image.BILINEAR)

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = (
                        sample[k]
                        .resize(self.base_size, Image.NEAREST)
                        .crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
                    )
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class FixScaleCrop(object):
    def __init__(self, crop_size: Tuple[int, int]):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample["image"]
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size[0]) / 2.0))
        y1 = int(round((h - self.crop_size[1]) / 2.0))
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = sample[k].crop(
                        (x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1])
                    )
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class FixedResize(object):
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # size: (w, h)

    def __call__(self, sample):
        img = sample["image"]

        img = img.resize(self.size, Image.BILINEAR)

        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = sample[k].resize(self.size, Image.NEAREST)
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class ToFromDict(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask, **kwargs):
        res = self.transform({"image": img, "label": mask, **kwargs})

        rest = []
        for k in res.keys():
            if k != "image" and k != "label":
                rest.append(res[k])

        return res["image"], res["label"], *rest


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                LambdaImg(lambda img: F.adjust_brightness(img, brightness_factor))
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                LambdaImg(lambda img: F.adjust_contrast(img, contrast_factor))
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                LambdaImg(lambda img: F.adjust_saturation(img, saturation_factor))
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(LambdaImg(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = transforms_lib.Compose(transforms)

        return transform

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        return transform(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        return format_string


class LambdaImg(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, sample):
        img = sample["image"]
        rest = {}
        for k in sample.keys():
            if k != "image":
                rest[k] = sample[k]
        img = self.lambd(img)
        return {"image": img, **rest}

    def __repr__(self):
        return self.__class__.__name__ + "()"
