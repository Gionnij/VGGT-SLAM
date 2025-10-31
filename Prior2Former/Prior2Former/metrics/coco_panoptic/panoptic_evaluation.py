# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import time
import os
import tempfile
from collections import OrderedDict
from types import SimpleNamespace
from typing import Optional
from PIL import Image
from tabulate import tabulate
from easydict import EasyDict as edict

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator
from panopticapi.evaluation import pq_compute_multi_core

logger = logging.getLogger(__name__)

def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))
    
    gt_annotations = {el['image_id']: el for el in gt_json['annotations']}
    matched_annotations_list = []
    for pred in pred_json['annotations']:
        image_id = pred['image_id']
        if image_id not in gt_annotations:
            raise Exception('no ground truth for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_annotations[image_id], pred))
    # pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    # matched_annotations_list = []
    # for gt_ann in gt_json['annotations']:
    #     image_id = gt_ann['image_id']
    #     if image_id not in pred_annotations:
    #         raise Exception('no prediction for the image with id: {}'.format(image_id))
    #     matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results
class COCOPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(
        self,
        dataset_name: str,
        datamodul,
        output_dir: Optional[str] = None,
        resize: int = 0,
        closed_world=True,
        small: int = 0,
    ):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """

        self.dm = datamodul
        if dataset_name == "LandF":
            self.thing_dataset_id_to_contiguous_id = {
                0: 0
            }
            self.stuff_dataset_id_to_contiguous_id = {
                1: 1
            }
            self._thing_contiguous_id_to_dataset_id = {
                0: 0
            }
            self._stuff_contiguous_id_to_dataset_id = {
                1: 1
            }

            self._metadata = SimpleNamespace(
                ignore_label=255,
                stuff_dataset_id_to_contiguous_id=self.stuff_dataset_id_to_contiguous_id,
                thing_dataset_id_to_contiguous_id=self.thing_dataset_id_to_contiguous_id,
                panoptic_json=os.path.join(
                    os.path.expanduser(datamodul.data_dir),
                    "landf_panoptic_val.json" if not small else f"landf_panoptic_val_{small}.json",
                ),
                panoptic_root=os.path.join(
                    os.path.expanduser(datamodul.data_dir),
                    "gtFine",
                    "landf_panoptic_val" ,
                ),
            )
        elif dataset_name == "cityscapes":
            self.thing_dataset_id_to_contiguous_id = {
                k: v
                for k, v in self.dm.from_cityscapes_id.items()
                if k in self.dm.thing_list
            }
            self.stuff_dataset_id_to_contiguous_id = {
                k: v
                for k, v in self.dm.from_cityscapes_id.items()
                if k in self.dm.stuff_list
            }
            self._thing_contiguous_id_to_dataset_id = {
                v: k
                for k, v in self.thing_dataset_id_to_contiguous_id.items()
                if k in self.dm.thing_list
            }
            self._stuff_contiguous_id_to_dataset_id = {
                v: k
                for k, v in self.stuff_dataset_id_to_contiguous_id.items()
                if k in self.dm.stuff_list
            }

            self._metadata = SimpleNamespace(
                ignore_label=255,
                label_divisor=datamodul.label_divisor,
                stuff_dataset_id_to_contiguous_id=self.stuff_dataset_id_to_contiguous_id,
                thing_dataset_id_to_contiguous_id=self.thing_dataset_id_to_contiguous_id,
                panoptic_json=os.path.join(
                    os.path.expanduser(datamodul.data_dir),
                    "cityscapes_panoptic_val.json" if not small else f"cityscapes_panoptic_val_{small}.json",
                ),
                panoptic_root=os.path.join(
                    os.path.expanduser(datamodul.data_dir),
                    "gtFine",
                    "cityscapes_panoptic_val" ,
                ),
            )
        elif "coco_val" in dataset_name:
            from Data_Loaders.PanopticSegmentation.coco_utils import (
                get_metadata,
            )

            if "_100" in dataset_name:
                postfix = "_100"
            else:
                postfix = ""
            if  "anomaly" in dataset_name:
                remove_unknowns=True
            else:
                remove_unknowns = False
            meta = get_metadata(remove_unknowns)
            if not "anomaly" in dataset_name:
                self.thing_dataset_id_to_contiguous_id = meta[
                    "thing_dataset_id_to_contiguous_id"
                ]
                self.stuff_dataset_id_to_contiguous_id = meta[
                    "stuff_dataset_id_to_contiguous_id"
                ]
                self._thing_contiguous_id_to_dataset_id = {
                    v: k for k, v in self.thing_dataset_id_to_contiguous_id.items()
                }
                self._stuff_contiguous_id_to_dataset_id = {
                    v: k for k, v in self.stuff_dataset_id_to_contiguous_id.items()
                }
                self._metadata = SimpleNamespace(
                    ignore_label=255,
                    label_divisor=datamodul.label_divisor,
                    stuff_dataset_id_to_contiguous_id=self.stuff_dataset_id_to_contiguous_id,
                    thing_dataset_id_to_contiguous_id=self.thing_dataset_id_to_contiguous_id,
                    panoptic_json=os.path.join(
                        os.path.expanduser(datamodul.data_dir),
                        "annotations",
                        f"panoptic_val2017{postfix}.json",
                    ),
                    panoptic_root=os.path.join(
                        os.path.expanduser(datamodul.data_dir),
                        "annotations",
                        "panoptic_val2017",
                    ),
                )
            else:
                self.thing_dataset_id_to_contiguous_id = meta[
                    "thing_dataset_id_to_contiguous_id"
                ]
                self.thing_dataset_id_to_contiguous_id[254] = 254
                self.stuff_dataset_id_to_contiguous_id = meta[
                    "stuff_dataset_id_to_contiguous_id"
                ]
                self._thing_contiguous_id_to_dataset_id = {
                    v: k for k, v in self.thing_dataset_id_to_contiguous_id.items()
                }
                self._stuff_contiguous_id_to_dataset_id = {
                    v: k for k, v in self.stuff_dataset_id_to_contiguous_id.items()
                }
                if resize <= 0:
                    panoptic_root = os.path.join(
                            os.path.expanduser(datamodul.data_dir),
                            "annotations",
                            "panoptic_val2017",
                        )
                elif resize==512:
                    panoptic_root = os.path.join(
                            os.path.expanduser(datamodul.data_dir),
                            "annotations",
                            "panoptic_val2017_resized512",
                        )

                elif resize==640:
                    panoptic_root = os.path.join(
                            os.path.expanduser(datamodul.data_dir),
                            "annotations",
                            "panoptic_val2017_resized640",
                        )

                elif resize==1024:
                    panoptic_root = os.path.join(
                            os.path.expanduser(datamodul.data_dir),
                            "annotations",
                            "panoptic_val2017_resized1024",
                        )    
                if "InD" in dataset_name:
                    self._metadata = SimpleNamespace(
                        ignore_label=255,
                        label_divisor=datamodul.label_divisor,
                        stuff_dataset_id_to_contiguous_id=self.stuff_dataset_id_to_contiguous_id,
                        thing_dataset_id_to_contiguous_id=self.thing_dataset_id_to_contiguous_id,
                        panoptic_json=os.path.join(
                            os.path.expanduser(datamodul.data_dir),
                            "annotations",
                            f"anomaly_InD_panoptic_val2017{postfix}.json",

                        ),
                    panoptic_root=panoptic_root
                )
                elif "OOD" in dataset_name:
                    self._metadata = SimpleNamespace(
                        ignore_label=255,
                        label_divisor=datamodul.label_divisor,
                        stuff_dataset_id_to_contiguous_id=self.stuff_dataset_id_to_contiguous_id,
                        thing_dataset_id_to_contiguous_id=self.thing_dataset_id_to_contiguous_id,
                        panoptic_json=os.path.join(
                            os.path.expanduser(datamodul.data_dir),
                            "annotations",
                            f"anomaly_OOD_panoptic_val2017{postfix}.json",

                        ),
                    panoptic_root=panoptic_root
                    )
                else:
                    self._metadata = SimpleNamespace(
                        ignore_label=255,
                        label_divisor=datamodul.label_divisor,
                        stuff_dataset_id_to_contiguous_id=self.stuff_dataset_id_to_contiguous_id,
                        thing_dataset_id_to_contiguous_id=self.thing_dataset_id_to_contiguous_id,
                        panoptic_json=os.path.join(
                            os.path.expanduser(datamodul.data_dir),
                            "annotations",
                            f"anomaly_full_panoptic_val2017{postfix}.json",

                        ),
                        panoptic_root=panoptic_root
                    )
        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        # print(segment_info)
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for file_name, image_id, (panoptic_img, segments_info) in zip(
            inputs["file_name"], inputs["image_id"], outputs["panoptic_seg"]
        ):
            # panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class
                        in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            # file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            print(file_name_png)
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": image_id,  # input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def evaluate(self,org_cal=False):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            if org_cal:
                from panopticapi.evaluation import pq_compute as pq_compute_cal
            else:
                pq_compute_cal=pq_compute
            print(gt_json)
            print("pred")
            print(predictions_json)
            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute_cal(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
        res["per_class"] = pq_res["per_class"]
        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = (
            [name]
            + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]]
            + [pq_res[name]["n"]]
        )
        data.append(row)
    table = tabulate(
        data,
        headers=headers,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json,
            args.pred_json,
            gt_folder=args.gt_dir,
            pred_folder=args.pred_dir,
        )
        _print_panoptic_results(pq_res)
