from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


class EncodeCategories(object):
    def __init__(self, class_categories):
        self.class_categories = class_categories

    def __call__(self, sample):
        semantic = sample["semantic"]
        category = torch.ones_like(semantic) * 255
        for cat in self.class_categories:
            category[
                (semantic >= cat["min_ix"])
                & (semantic < (cat["min_ix"] + cat["num_classes"]))
            ] = cat["id"]

        return dict(
            category=category,
            **sample,
        )
