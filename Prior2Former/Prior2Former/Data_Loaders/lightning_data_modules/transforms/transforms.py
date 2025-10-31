import numbers
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms as transforms_lib
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



class PanopticNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        rest = {}
        if "image_cityscapes" in sample:
            img = sample["image_cityscapes"]

            img = np.array(img).astype(np.float32)
            img /= 255.0
            img -= self.mean
            img /= self.std
            rest["image_cityscapes"] = img
        img = sample["image"]
        if isinstance(img, Image.Image):
            img = np.array(img).astype(np.float32)
            img /= 255.0
        if isinstance(img, torch.Tensor):
            img = img.numpy().astype(np.float32)
        img -= self.mean
        img /= self.std

        for k in sample.keys():
            if k != "image" and k != "image_cityscapes":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = np.array(sample[k]).astype(np.float32)
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class PanopticEncodeSegmap(object):
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

    def __call__(self, sample):
        img = sample["image"]
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
            if k != "image" and k != self.semantic_key:
                rest[k] = sample[k]

        if self.ood_classes is not None and len(self.ood_classes) > 0:
            return {
                "image": img,
                self.semantic_key: mask,
                "ood": Image.fromarray(mask_copy),
                f"{self.semantic_key}_original": original_mask,
                **rest,
            }
        else:
            return {
                "image": img,
                self.semantic_key: mask,
                f"{self.semantic_key}_original": original_mask,
                **rest,
            }


class PanopticToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        rest = {}
        for k,v in sample.items():
            if "image" in k:
                img = sample[k]
                img = np.array(img).astype(np.float32).transpose((2, 0, 1))
                img = torch.from_numpy(img).float()
                rest[k] = img

        for k in sample.keys():
            if "image" not in k:
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = torch.from_numpy(np.array(sample[k])).long()
                else:
                    rest[k] = sample[k]

        return rest


class PanopticRandomHorizontalFlip(object):
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


class PanopticRandomRotate(object):
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


class PanopticRandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample["image"]
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        rest = {}
        for k in sample.keys():
            if k != "image":
                rest[k] = sample[k]

        return {"image": img, **rest}


class PanopticRandomScaleCrop(object):
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
                        .resize((ow, oh), Image.NEAREST)
                        .crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
                    )
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class PanopticRandomCropFirstResize(object):
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


class PanopticRandomCrop(object):
    def __init__(self, crop_size: Tuple[int, int], base_size: Tuple[int, int]):
        self.crop_size = crop_size
        self.base_size = base_size

    def __call__(self, sample):
        img = sample["image"]

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        img = img.resize(self.base_size, Image.BILINEAR)
        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = (
                        sample[k]
                        .crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
                        .resize(self.base_size, Image.NEAREST)
                    )
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class PanopticRandomZoomCrop(object):
    def __init__(
        self,
        crop_size: Tuple[int, int],
        base_size: Tuple[int, int],
        scale_range=(1.0, 1.0),
    ):
        self.crop_size = crop_size
        self.base_size = base_size
        self.scale_range = scale_range

    def __call__(self, sample):
        img = sample["image"]

        # Step 1: Apply random zoom
        w, h = img.size
        scale_factor = random.uniform(*self.scale_range)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Step 2: Apply random crop
        x1 = random.randint(0, new_w - self.crop_size[0])
        y1 = random.randint(0, new_h - self.crop_size[1])
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        # Step 3: Resize to the base size
        img = img.resize(self.base_size, Image.BILINEAR)

        # Process the rest of the sample similarly
        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    if isinstance(sample[k], Image.Image):
                        rest[k] = (
                            sample[k]
                            .resize((new_w, new_h), Image.NEAREST)
                            .crop(
                                (x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1])
                            )
                            .resize(self.base_size, Image.NEAREST)
                        )
                    else:
                        # Handling for numpy arrays (e.g., masks)
                        rest[k] = np.array(
                            Image.fromarray(sample[k])
                            .resize((new_w, new_h), Image.NEAREST)
                            .crop(
                                (x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1])
                            )
                            .resize(self.base_size, Image.NEAREST)
                        )
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class PanopticFixScaleCrop(object):
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


class PanopticFixedResize(object):
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # size: (w, h)

    def __call__(self, sample):
        rest = {}
        if "image_cityscapes" in sample:
            img = sample["image_cityscapes"]
            img = img.resize(self.size, Image.BILINEAR)
            rest["image_cityscapes"] = img
        img = sample["image"]

        img = img.resize(self.size, Image.BILINEAR)

        for k in sample.keys():
            if k != "image" and k != "image_cityscapes":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = sample[k].resize(self.size, Image.NEAREST)
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class PanopticToFromDict(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask, **kwargs):
        res = self.transform({"image": img, "label": mask, **kwargs})

        rest = []
        for k in res.keys():
            if k != "image" and k != "label":
                rest.append(res[k])

        return res["image"], res["label"], *rest


class PanopticColorJitter(object):
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


class PanopticLambdaImg(object):
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


class PanopticRandomZoomCrop_CategoryAreaConstraint(object):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single category
    occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
    truth, which can cause unstability in training. The function attempts to find such a valid
    cropping window for at most 10 times.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int],
        base_size: Tuple[int, int],
        scale_range=(1.0, 1.0),
        single_category_max_area: float = 1.0,
        ignored_category: int = None,
    ):
        self.crop_size = crop_size
        self.base_size = base_size
        self.scale_range = scale_range
        self.single_category_max_area = single_category_max_area
        self.ignored_category = ignored_category

    def __call__(self, sample):
        img = sample["image"]
        w, h = img.size
        if self.crop_size is None:
            self.crop_size = (w, h)
        # Step 1: Apply random zoom
        scale_factor = random.uniform(*self.scale_range)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        for _ in range(10):
            # Step 2: Apply random crop to segmentation
            x1 = random.randint(0, np.max((0, new_w - self.crop_size[0])))
            y1 = random.randint(0, np.max((new_h - self.crop_size[1], 0)))
            sem_seg_temp = sample["semantic"].crop(
                (x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1])
            )
            labels, cnt = np.unique(sem_seg_temp, return_counts=True)
            if self.ignored_category is not None:
                cnt = cnt[labels != self.ignored_category]
            if (
                len(cnt) > 1
                and np.max(cnt) < np.sum(cnt) * self.single_category_max_area
            ):
                break

        # Step 2: Apply random crop
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        # Step 3: Resize to the base size
        if self.base_size is not None:
            img = img.resize(self.base_size, Image.BILINEAR)
        else:
            img = img.resize((w, h), Image.BILINEAR)
        # Process the rest of the sample similarly
        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    if isinstance(sample[k], Image.Image):
                        rest[k] = (
                            sample[k]
                            .resize((new_w, new_h), Image.NEAREST)
                            .crop(
                                (x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1])
                            )
                            .resize(self.base_size, Image.NEAREST)
                        )
                    else:
                        # Handling for numpy arrays (e.g., masks)
                        rest[k] = np.array(
                            Image.fromarray(sample[k])
                            .resize((new_w, new_h), Image.NEAREST)
                            .crop(
                                (x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1])
                            )
                            .resize(self.base_size, Image.NEAREST)
                        )
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


class PanopticColorAugSSDTransform:
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__()
        assert img_format in ["BGR", "RGB"]
        self.is_rgb = img_format == "RGB"
        del img_format
        self.brightness_delta = brightness_delta
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high
        self.hue_delta = hue_delta

    def __call__(self, sample):
        image = np.array(sample["image"])
        image = self.apply_image(image)
        sample["image"] = Image.fromarray(image)
        return sample

    def apply_image(self, img, interp=None):
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        img = self.brightness(img)
        if random.randrange(2):
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            img = self.saturation(img)
            img = self.hue(img)
            img = self.contrast(img)
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randrange(2):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img):
        if random.randrange(2):
            return self.convert(
                img, alpha=random.uniform(self.contrast_low, self.contrast_high)
            )
        return img

    def saturation(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_low, self.saturation_high),
            )
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int)
                + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img


class PanopticResizeScale(object):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        scale_range: tuple = (1.0, 1.0),
        target_shape: tuple = (1024, 1024),
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        super().__init__()
        self.scale_range = scale_range
        self.target_shape = target_shape

    def __call__(self, sample):
        image = sample["image"]
        random_scale = np.random.uniform(*self.scale_range)
        w, h = image.size
        target_scale_shape = np.multiply(self.target_shape, random_scale)
        output_scale = np.minimum(target_scale_shape[0] / w, target_scale_shape[1] / h)
        new_w, new_h = int(w * output_scale), int(h * output_scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = sample[k].resize((new_w, new_h), Image.NEAREST)
                else:
                    rest[k] = sample[k]

        return {"image": image, **rest}


class PanopticFixedSizeCrop(object):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """

    def __init__(
        self,
        crop_size: Tuple[int],
        pad: bool = True,
        pad_value: float = 128.0,
        seg_pad_value: int = 255,
    ):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value to the image.
            seg_pad_value: the padding value to the segmentation mask.
        """
        super().__init__()
        self.crop_size = crop_size
        self.pad = pad
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value

    def __call__(self, sample):
        img = sample["image"]
        w, h = img.size
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        img = np.array(img)
        max_offset = np.subtract((w, h), output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)

        img = img[
            offset[1] : offset[1] + output_size[1],
            offset[0] : offset[0] + output_size[0],
        ]

        if self.pad:
            # Add padding if the image is scaled down.
            h, w = img.shape[:2]
            pad_size = np.subtract(output_size, (w, h))
            pad_size = np.maximum(pad_size, 0)

            if img.ndim == 3:
                padding = ((0, pad_size[1]), (0, pad_size[0]), (0, 0))
            else:
                padding = ((0, pad_size[1]), (0, pad_size[0]))
            img = np.pad(
                img,
                padding,
                mode="constant",
                constant_values=self.pad_value,
            )

        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    im_flag = False
                    if isinstance(sample[k], Image.Image):
                        im_flag = True
                        sample[k] = np.array(sample[k])

                    curr = sample[k][
                        offset[1] : offset[1] + output_size[1],
                        offset[0] : offset[0] + output_size[0],
                    ]
                    if self.pad:

                        if len(curr.shape) == 3:
                            padding = ((0, pad_size[1]), (0, pad_size[0]), (0, 0))
                        else:
                            padding = ((0, pad_size[1]), (0, pad_size[0]))
                        rest[k] = np.pad(
                            curr,
                            padding,
                            mode="constant",
                            constant_values=(
                                self.seg_pad_value
                                if k in ["semantic", "semantic_original", "instance"]
                                else self.pad_value
                            ),
                        )
                    else:
                        rest[k] = curr
                    if im_flag:
                        rest[k] = Image.fromarray(rest[k])
                else:
                    rest[k] = sample[k]

        return {"image": Image.fromarray(img), **rest}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample["image"]

        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = np.array(sample[k]).astype(np.float32)
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


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

    def __call__(self, sample):
        img = sample["image"]
        mask = sample[self.semantic_key]

        # original_mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        try:
            original_mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        except Exception as e:
            print(e)   # Maybe a race condition happens here. This avoids the buffer exception.
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
            if k != "image" and k != self.semantic_key:
                rest[k] = sample[k]

        if self.ood_classes is not None and len(self.ood_classes) > 0:
            return {
                "image": img,
                self.semantic_key: mask,
                "ood": Image.fromarray(mask_copy),
                f"{self.semantic_key}_original": original_mask,
                **rest,
            }
        else:
            return {
                "image": img,
                self.semantic_key: mask,
                f"{self.semantic_key}_original": original_mask,
                **rest,
            }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample["image"]
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()

        rest = {}
        for k in sample.keys():
            if k != "image":
                if isinstance(sample[k], (Image.Image, np.ndarray)):
                    rest[k] = torch.from_numpy(np.array(sample[k])).long()
                else:
                    rest[k] = sample[k]

        return {"image": img, **rest}


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
