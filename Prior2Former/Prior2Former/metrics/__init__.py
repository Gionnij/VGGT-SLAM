from typing import Dict

from metrics.pq_u3hs import PanopticQualityTest, PanopticQualityUnknown


def get_metric(name: str, args: Dict):
    if name == "pq_test":
        return PanopticQualityTest(**args)
    elif name == "pq_unknown":
        return PanopticQualityUnknown(**args)
    else:
        raise Exception(f"Metric named {name} not known")
