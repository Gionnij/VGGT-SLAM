#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from PIL import Image

from coco_utils import COCO_CATEGORIES, COCO_UNKNOWN_CATEGORIES


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    from panopticapi.utils import rgb2id
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(
    panoptic_json, panoptic_root, sem_seg_root, categories
):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.

    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]
    thing_ids = [k["id"] for k in categories if k["isthing"] == 1]
    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(stuff_ids) <= 254
    for i, stuff_id in enumerate(stuff_ids):
        id_map[stuff_id] = i + 1
    for thing_id in thing_ids:
        id_map[thing_id] = 0
    id_map[0] = 255

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


def main():
    from fvcore.common.download import download
    dataset_dir = "/Datasets/Coco"  # os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
    for s in ["val2017", "train2017"]:
        separate_coco_semantic_from_panoptic(
            os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
            os.path.join(dataset_dir, "annotations/panoptic_{}".format(s)),
            os.path.join(dataset_dir, "annotations/segmentation_{}".format(s)),
            COCO_CATEGORIES,
        )

    # Prepare val2017_100 for quick testing:

    dest_dir = os.path.join(dataset_dir, "annotations/")
    URL_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"
    download(URL_PREFIX + "annotations/coco/panoptic_val2017_100.json", dest_dir)
    with open(os.path.join(dest_dir, "panoptic_val2017_100.json")) as f:
        obj = json.load(f)

    def link_val100(dir_full, dir_100):
        print("Creating " + dir_100 + " ...")
        os.makedirs(dir_100, exist_ok=True)
        for img in obj["images"]:
            basename = os.path.splitext(img["file_name"])[0]
            src = os.path.join(dir_full, basename + ".jpg")
            dst = os.path.join(dir_100, basename + ".jpg")
            src = os.path.relpath(src, start=dir_100)
            os.symlink(src, dst)

    link_val100(
        os.path.join(dataset_dir, "panoptic_val2017"),
        os.path.join(dataset_dir, "panoptic_val2017_100"),
    )

    link_val100(
        os.path.join(dataset_dir, "segmentation_val2017"),
        os.path.join(dataset_dir, "segmentation_val2017_100"),
    )


def create_anomaly_set_json(anomaly_set_json, base_path, store_path):
    from PIL import Image

    unknown_ids = [C["id"] for C in COCO_UNKNOWN_CATEGORIES]
    for an in anomaly_set_json["annotations"]:
        pan_img = os.path.join(base_path, an["file_name"])
        if "302030" in an["file_name"]:
            print("found")
        image = np.array(Image.open(pan_img))
        image = rgb2id(image)
        for segment_info in an["segments_info"]:
            if segment_info["category_id"] in unknown_ids:
                segment_info["category_id"] = 254
                segment_info["name"] = "ood"
    anomaly_set_json["categories"] = [cat for cat in anomaly_set_json["categories"] if cat["id"] not in unknown_ids]
    with open(store_path, "w") as f:
        json.dump(anomaly_set_json, f)


if __name__ == "__main__":
    # main()
    format = "val2017_100"
    path = f"/Datasets/Coco/annotations/panoptic_{format}.json"
    basepath =  f"/Datasets/Coco/annotations/anomaly_full_panoptic_{format}.json"
    with open(path, "r") as f:
        anomaly_set_json = json.load(f)
    if len([c for c in anomaly_set_json["categories"] if c["name"] == "ood"]) == 0:
        print("add ood")
        anomaly_set_json["categories"].append(
            {"supercategory": "ood", "isthing": 1, "id": 254, "name": "ood"}
        )
    create_anomaly_set_json(
        anomaly_set_json, basepath, path.replace(".json", "_ood.json")
    )
