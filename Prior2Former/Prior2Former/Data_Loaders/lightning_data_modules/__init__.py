from typing import Dict

from Data_Loaders.lightning_data_modules.bdd import BDD100KDataModule
from Data_Loaders.lightning_data_modules.cityscapes import (
    CityscapesDataModule,
)
from Data_Loaders.lightning_data_modules.coco import CocoDataModule
from Data_Loaders.lightning_data_modules.cityscapes_rain import (
    CityscapesOODDataModule,
)
from Data_Loaders.lightning_data_modules.lostandfound import (
    LostAndFoundDataModule,
)

def get_datamodule(name: str, args_raw: Dict, data_dir):
    # args.pop("data_dir")
    args = args_raw.copy()
    if "data_dir" in args:
        args.pop("data_dir")
    if name == "cityscapes":
        return CityscapesDataModule(data_dir=data_dir, **args)
    elif name == "cityscapes_rain" or name == "cityscapes_foggy":
        return CityscapesOODDataModule(data_dir=data_dir, **args)
    elif name == "bdd100kSeg":
        return BDD100KDataModule(data_dir=data_dir, **args)
    elif name == "bdd100kAnom":
        return BDD100KDataModule(data_dir=data_dir, **args, anomaly=True)
    elif name == "lostandfound":
        return LostAndFoundDataModule(data_dir=data_dir, **args)
    elif name == "coco":
        return CocoDataModule(**args, data_dir=data_dir)
    elif name == "cocoAnom":
        return CocoDataModule(**args, data_dir=data_dir, anomaly=True)
    raise Exception(f"Datamodule {name} unknown")
