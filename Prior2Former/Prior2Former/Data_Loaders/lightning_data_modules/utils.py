import torch
from torch.utils.data._utils.collate import default_collate


def mask2former_collate(batch):
    # Assuming each element in batch is a dictionary
    batch_dict = {}
    for key in batch[0]:
        if key in [
            "masks",
            "labels",
        ]:  # m2f has variable dimensions/length fro masks and labels
            batch_dict[key] = [d[key] for d in batch]
        else:
            # Use the default collate function for other types
            batch_dict[key] = default_collate([d[key] for d in batch])
    return batch_dict


def get_collate(name: str):
    if name == "mask2former":
        return mask2former_collate
    else:
        return default_collate
