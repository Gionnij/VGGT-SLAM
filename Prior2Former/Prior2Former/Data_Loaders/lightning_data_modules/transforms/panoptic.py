from typing import Dict

import numpy as np
import torch
from utils.logging_utils.log_writers import (
    gl_info,
)


class PanopticTargetGenerator(object):
    """
    Generates panoptic training target for Panoptic-DeepLab.
    Annotation is assumed to have Cityscapes format.
    Arguments:
        rgb2id: Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
        thing_list: List, a list of thing classes
        sigma: the sigma for Gaussian kernel.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
        ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in semantic segmentation branch,
            crowd region is ignored in the original TensorFlow implementation.
    """

    def __init__(
        self,
        thing_list,
        label_divisor,
        sigma=8,
        ignore_stuff_in_offset=False,
        small_instance_area=0,
        small_instance_weight=1,
        max_instances=400,
        void_instance=False,
    ):
        self.thing_list = thing_list
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight

        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

        self.label_divisor = label_divisor
        self.max_instances = max_instances
        self.void_instance = void_instance

    def __call__(self, sample):
        semantic_label = sample["semantic"]
        o_semantic_label = sample["semantic_original"]
        instance = sample["instance"]

        instance_clone = remap_instance(instance)

        unique_counting_instance = instance_clone.clone()
        unique_counting_instance[semantic_label == 255] = self.label_divisor
        unique_counting_instance = remap_instance(
            unique_counting_instance,
            ignore_instance=self.label_divisor if not self.void_instance else None,
        )
        instance_unique = unique_counting_instance.unique()
        instance_clss = torch.zeros(self.max_instances, dtype=torch.int32)
        for instance_id in instance_unique:
            if instance_id.item() != self.label_divisor:
                mask = unique_counting_instance == instance_id
                if semantic_label[mask].shape[0] > 0:
                    instance_clss[instance_id] = semantic_label[mask][0]
                else:
                    instance_clss[instance_id] = 255

        panoptic_id = o_semantic_label * self.label_divisor + instance_clone
        segments = panoptic_id.unique()

        height, width = semantic_label.shape[0], semantic_label.shape[1]
        foreground = np.zeros_like(semantic_label, dtype=np.uint8)
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(semantic_label, dtype=np.float32)
        x_coord = np.ones_like(semantic_label, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(semantic_label, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(semantic_label, dtype=np.uint8)
        offset_weights = np.zeros_like(semantic_label, dtype=np.uint8)
        for seg in segments:
            seg_id = seg

            cat_id = seg // self.label_divisor

            if cat_id in self.thing_list:
                foreground[panoptic_id == seg_id] = 1

            center_weights[panoptic_id == seg_id] = 1
            if self.ignore_stuff_in_offset:
                # Handle stuff region.
                if cat_id in self.thing_list:
                    offset_weights[panoptic_id == seg_id] = 1
            else:
                offset_weights[panoptic_id == seg_id] = 1
            if cat_id in self.thing_list:
                # find instance center
                mask_index = np.where(panoptic_id == seg_id)
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic_id == seg_id] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])

                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(
                    center[0, aa:bb, cc:dd], self.g[a:b, c:d]
                )

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset_y_index = (
                    np.zeros_like(mask_index[0]),
                    mask_index[0],
                    mask_index[1],
                )
                offset_x_index = (
                    np.ones_like(mask_index[0]),
                    mask_index[0],
                    mask_index[1],
                )
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]

        return dict(
            foreground=torch.as_tensor(foreground.astype("long")),
            center=torch.as_tensor(center.astype(np.float32)),
            # center_points=center_pts,  Are the center points needed anywhere?
            offset=torch.as_tensor(offset.astype(np.float32)),
            semantic_weights=torch.as_tensor(semantic_weights.astype(np.float32)),
            center_weights=torch.as_tensor(center_weights.astype(np.float32)),
            offset_weights=torch.as_tensor(offset_weights.astype(np.float32)),
            panoptic_id=semantic_label * self.label_divisor + instance_clone,
            instance_remapped=unique_counting_instance,
            instance_clss=instance_clss,
            **sample,
        )


def remap_instance(instance, ignore_instance=None):
    unique_instance = instance.unique()
    instance_clone = instance.clone()
    c = 0
    for i, x in enumerate(unique_instance):
        if ignore_instance is None or x.item() != ignore_instance:
            instance_clone[instance == x] = c
            c = c + 1
    return instance_clone


class MaxDeeplabTargetGenerator(object):
    def __init__(
        self,
        thing_list,
        label_divisor,
        small_instance_area=0,
        small_instance_weight=1,
        n_masks=50,
        to_train_id: Dict = {},
    ):
        self.thing_list = thing_list
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight

        self.label_divisor = label_divisor

        self.n_masks = n_masks
        self.to_train_id = to_train_id

    def __call__(self, sample):
        semantic_label = sample["semantic"]
        o_semantic_label = sample["semantic_original"]
        instance = sample["instance"]

        height, width = semantic_label.shape[0], semantic_label.shape[1]

        instance_clone = remap_instance(instance)

        unique_counting_instance = instance_clone.clone()
        unique_counting_instance[semantic_label == 255] = self.label_divisor

        panoptic_id = o_semantic_label * self.label_divisor + instance_clone
        segments = panoptic_id.unique()

        instance_mask = torch.zeros((self.n_masks, height, width), dtype=torch.uint8)
        pixel_instance = torch.zeros((height, width), dtype=torch.short)
        instance_class = torch.zeros((self.n_masks,), dtype=torch.long)

        foreground = np.zeros_like(semantic_label, dtype=np.uint8)

        semantic_weights = np.ones_like(semantic_label, dtype=np.uint8)

        instance_counter = 0

        for seg in segments:
            seg_id = seg

            cat_id = seg // self.label_divisor

            if cat_id.item() in self.to_train_id:
                if instance_counter >= self.n_masks:
                    raise Exception(
                        f"Set maximum number of masks to {self.n_masks}, but encountered sample with at least {instance_counter+1} unique segments"
                    )

                instance_class[instance_counter] = self.to_train_id[cat_id.item()]
                instance_mask[instance_counter] = panoptic_id == seg_id
                pixel_instance[panoptic_id == seg_id] = instance_counter

                instance_counter = instance_counter + 1

            if cat_id in self.thing_list:
                foreground[panoptic_id == seg_id] = 1

            if cat_id in self.thing_list:
                # find instance center
                mask_index = np.where(panoptic_id == seg_id)
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic_id == seg_id] = self.small_instance_weight

        return dict(
            foreground=torch.as_tensor(foreground.astype("long")),
            panoptic_id=semantic_label * self.label_divisor + instance_clone,
            instance_remapped=remap_instance(
                unique_counting_instance, ignore_instance=self.label_divisor
            ),
            n_instances=torch.tensor(instance_counter),
            instance_mask=instance_mask,
            pixel_instance=pixel_instance,
            instance_class=instance_class,
            **sample,
        )


class PrototypicalDeeplabTargetGenerator(object):
    def __init__(
        self,
        thing_list,
        stuff_list,
        label_divisor,
        small_instance_area=0,
        small_instance_weight=1,
        sigma=8,
        max_instances=150,
        fail_on_max_instances=True,
        per_class_centers=True,
        semantic_key="semantic",
    ):
        self.thing_list = thing_list
        self.stuff_list = stuff_list
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight

        self.thing_id = {thing: i for i, thing in enumerate(thing_list)}
        self.stuff_id = {stuff: i for i, stuff in enumerate(stuff_list)}

        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

        self.label_divisor = label_divisor
        self.max_instances = max_instances
        self.fail_on_max_instances = fail_on_max_instances

        self.per_class_centers = per_class_centers

        self.semantic_key = semantic_key

    def __call__(self, sample):
        semantic_label = sample[self.semantic_key]
        o_semantic_label = sample[f"{self.semantic_key}_original"]
        instance = sample["instance"]

        height, width = semantic_label.shape[0], semantic_label.shape[1]

        instance_clone = remap_instance(instance)

        unique_counting_instance = instance_clone.clone()
        unique_counting_instance[semantic_label == 255] = self.label_divisor

        panoptic_id = o_semantic_label * self.label_divisor + instance_clone
        segments = panoptic_id.unique()

        foreground = np.zeros_like(semantic_label, dtype=np.uint8)

        height, width = semantic_label.shape[0], semantic_label.shape[1]
        foreground = np.zeros_like(semantic_label, dtype=np.uint8)
        center = np.zeros(
            (len(self.thing_list) if self.per_class_centers else 1, height, width),
            dtype=np.float32,
        )
        center_pts = []
        center_onehot = np.ones((1, height, width), dtype=np.int8) * (-1)
        center_assoc = np.ones((1, height, width), dtype=np.int32) * (-1)
        center_positions = np.zeros((self.max_instances, 2), dtype=np.int32)

        semantic_weights = np.ones_like(semantic_label, dtype=np.uint8)
        center_weights = np.zeros_like(semantic_label, dtype=np.uint8)

        stuff_id = np.ones((1, height, width), dtype=np.int8) * 255

        instance_counter = 0

        for seg in segments:
            seg_id = seg

            # cat_id = seg // self.label_divisor
            cat_id = torch.div(seg, self.label_divisor, rounding_mode="trunc")

            center_weights[panoptic_id == seg_id] = 1

            if cat_id in self.thing_list:
                foreground[panoptic_id == seg_id] = 1

            if cat_id in self.thing_list and (
                self.fail_on_max_instances or instance_counter < self.max_instances
            ):
                # find instance center
                mask_index = np.where(panoptic_id == seg_id)
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic_id == seg_id] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])

                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center_clss_index = (
                    self.thing_id[cat_id.item()] if self.per_class_centers else 0
                )
                center[
                    center_clss_index,
                    aa:bb,
                    cc:dd,
                ] = np.maximum(
                    center[center_clss_index, aa:bb, cc:dd], self.g[a:b, c:d]
                )
                center_onehot[0, y, x] = instance_counter
                center_assoc[0, panoptic_id == seg_id] = instance_counter
                center_positions[instance_counter, 0] = y
                center_positions[instance_counter, 1] = x
                instance_counter += 1
            elif cat_id in self.stuff_list:
                stuff_id[:, panoptic_id == seg_id] = self.stuff_id[cat_id.item()]

        return dict(
            foreground=torch.as_tensor(foreground.astype("long")),
            center=torch.as_tensor(center),
            center_onehot=torch.as_tensor(center_onehot.astype("long")),
            center_positions=torch.as_tensor(center_positions.astype("long")),
            num_instances=instance_counter,
            center_assoc=torch.as_tensor(center_assoc.astype("long")),
            center_weights=torch.as_tensor(center_weights.astype(np.float32)),
            stuff_id=torch.as_tensor(stuff_id.astype("long")),
            panoptic_id=semantic_label * self.label_divisor + instance_clone,
            instance_remapped=remap_instance(
                unique_counting_instance, ignore_instance=self.label_divisor
            ),
            **sample,
        )


class Mask2FormerPanopticTargetGenerator(object):
    def __init__(
        self,
        thing_list,
        stuff_list,
        label_divisor,
        from_dataset_id,
        small_instance_area=0,
        small_instance_weight=1,
        max_instances=150,
        fail_on_max_instances=True,
        # per_class_centers=True,
        semantic_key="semantic",
    ):
        self.thing_list = thing_list
        self.stuff_list = stuff_list
        self.from_dataset_id = from_dataset_id
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight

        self.thing_id = {thing: i for i, thing in enumerate(thing_list)}
        self.stuff_id = {stuff: i for i, stuff in enumerate(stuff_list)}
        self.max_class = max(stuff_list)


        self.label_divisor = label_divisor
        self.max_instances = max_instances
        self.fail_on_max_instances = fail_on_max_instances

        self.semantic_key = semantic_key

    def __call__(self, sample):
        semantic_label = sample[self.semantic_key]
        o_semantic_label = sample[f"{self.semantic_key}_original"]
        instance = sample["instance"]

        height, width = semantic_label.shape[0], semantic_label.shape[1]

        instance_clone = remap_instance(instance)

        unique_counting_instance = instance_clone.clone()
        unique_counting_instance[semantic_label == 255] = self.label_divisor

        panoptic_id = o_semantic_label * self.label_divisor + instance_clone
        segments = panoptic_id.unique()

        semantic_weights = np.ones_like(semantic_label, dtype=np.uint8)


        instance_counter = 0
        masks = []
        labels = []
        for seg in segments:
            seg_id = seg

            cat_id = torch.div(seg, self.label_divisor, rounding_mode="trunc")


            if cat_id in self.thing_list and (
                self.fail_on_max_instances or instance_counter < self.max_instances
            ):
                # find instance center
                mask_index = np.where(panoptic_id == seg_id)
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic_id == seg_id] = self.small_instance_weight
                masks.append(panoptic_id == seg)
                labels.append(self.from_dataset_id[cat_id.item()])
                instance_counter += 1
            elif cat_id in self.stuff_list:
                masks.append(panoptic_id == seg_id)
                labels.append(self.from_dataset_id[cat_id.item()])
        if len(labels) == 0:
            gl_info("No Mask/Label in this crop")
            masks = torch.zeros((0, *semantic_label.shape[:2]))
            labels = torch.tensor([], dtype=int)
        else:
            masks = torch.stack(masks).to(int)

            labels = torch.tensor(labels, dtype=torch.int32)

        return dict(
            num_instances=instance_counter,
            panoptic_id=semantic_label * self.label_divisor + instance_clone,
            instance_remapped=remap_instance(
                unique_counting_instance, ignore_instance=self.label_divisor
            ),
            masks=masks,
            labels=labels,
            **sample,
        )


class Mask2FormerSemanticTargetGenerator(object):

    def __init__(
        self,
        thing_list,
        stuff_list,
        label_divisor,
        from_dataset_id,
        small_instance_area=0,
        small_instance_weight=1,
        max_instances=150,
        fail_on_max_instances=True,
        semantic_key="semantic",
    ):
        self.thing_list = thing_list
        self.stuff_list = stuff_list
        self.from_dataset_id = from_dataset_id
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight

        self.thing_id = {thing: i for i, thing in enumerate(thing_list)}
        self.stuff_id = {stuff: i for i, stuff in enumerate(stuff_list)}
        self.max_class = max(stuff_list)

        self.label_divisor = label_divisor
        self.max_instances = max_instances
        self.fail_on_max_instances = fail_on_max_instances

        self.semantic_key = semantic_key

    def __call__(self, sample):
        semantic_label = sample[self.semantic_key]
        o_semantic_label = sample[f"{self.semantic_key}_original"]

        unique_semantic_label = semantic_label.clone()
        unique_semantic_label[semantic_label == 255] = self.label_divisor

        panoptic_id = o_semantic_label * self.label_divisor
        segments = panoptic_id.unique()

        semantic_weights = np.ones_like(semantic_label, dtype=np.uint8)

        instance_counter = 0
        masks = []
        labels = []
        for seg in segments:
            seg_id = seg

            # cat_id = seg // self.label_divisor
            cat_id = torch.div(seg, self.label_divisor, rounding_mode="trunc")

            if cat_id in self.thing_list and (
                self.fail_on_max_instances or instance_counter < self.max_instances
            ):
                # find instance center
                mask_index = np.where(panoptic_id == seg_id)
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic_id == seg_id] = self.small_instance_weight
                masks.append(panoptic_id == seg)
                labels.append(self.from_dataset_id[cat_id.item()])
                instance_counter += 1
            elif cat_id in self.stuff_list:
                masks.append(panoptic_id == seg_id)
                labels.append(self.from_dataset_id[cat_id.item()])
        if len(labels) == 0:
            gl_info("No Mask/Label in this crop")
            masks = torch.zeros((0, *semantic_label.shape[:2]))
            labels = torch.tensor([], dtype=int)
        else:
            masks = torch.stack(masks).to(int)

            labels = torch.tensor(labels, dtype=torch.int32)

        return dict(
            num_instances=instance_counter,
            panoptic_id=semantic_label * self.label_divisor,
            instance_remapped=remap_instance(
                unique_semantic_label, ignore_instance=self.label_divisor
            ),
            masks=masks,
            labels=labels,
            **sample,
        )
