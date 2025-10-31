import argparse
import copy
import json
import os
from datetime import datetime

from typing import List


def update_dict(master, updated, expand=False):
    # master.update({k: updated[k] for k, v in master.items() if k in updated})
    merged = copy.deepcopy(master)
    for key, value in updated.items():
        if key in merged:
            if isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key]=update_dict(merged[key], value, expand=True)
            else:
                merged[key] = value
        else:
            if expand:
                merged[key] = value
    return merged



TRUE, FALSE = "True", "False"

def remove_defauts(params, keep :List[str]=None):
    if keep is None:
        keep=[]
    return {k: None if k not in keep else v for k, v in params.items()}


class StoreBooleanAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        assert values in (TRUE, FALSE)
        setattr(namespace, self.dest, values == TRUE)
