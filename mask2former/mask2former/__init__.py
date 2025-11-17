# Copyright (c) Facebook, Inc. and its affiliates.
import os
import warnings

# Dataset registration can pull in heavy dependencies; allow skipping for smoke tests
# or environments with partial installs.
if os.getenv("MASK2FORMER_SKIP_DATA") == "1":
    data = None
else:
    try:
        from . import data  # register all new datasets
    except Exception as exc:
        warnings.warn(
            "[mask2former] Failed to register datasets (continuing without them): %s" % exc
        )
        data = None

from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
