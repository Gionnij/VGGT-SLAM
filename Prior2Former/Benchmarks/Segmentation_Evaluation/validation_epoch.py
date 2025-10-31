import argparse
import torch
torch.set_float32_matmul_precision("medium")
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from Prior2Former.lightning_utils.panoptic import (
    PanopticSegmentationModule,
)

def get_args():
    parser = argparse.ArgumentParser(description="Run validation with Mask2Former on Cityscapes")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt file)",
    )

    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="file:/Experiments/mlflow",
        help="MLflow tracking URI (default: local file path)",
    )

    parser.add_argument(
        "--val_size",
        type=str,
        default="2048,1024",
        help="Validation image size as WIDTH,HEIGHT (default: 2048,1024)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Datasets/Cityscapes",
        help="Path to dataset directory",
    )

    return parser.parse_args()

def setup_module(ckpt_path, data_dir, val_size):
    print(f"Loading checkpoint from {ckpt_path} ...")
    modul = PanopticSegmentationModule.load_from_checkpoint(ckpt_path)
    modul = modul.to("cuda")

    # Configure datamodule settings
    modul.dm.data_dir = data_dir
    modul.dm.base_size_val = val_size

    return modul

def main():
    args = get_args()
    val_size = tuple(map(int, args.val_size.split(",")))

    modul = setup_module(
        ckpt_path=args.ckpt_path,
        data_dir=args.data_dir,
        val_size=val_size,
    )

    logger = MLFlowLogger(
        experiment_name="val",
        tracking_uri=args.tracking_uri,
    )

    trainer = pl.Trainer(
        devices=1,
        logger=logger,
    )

    trainer.validate(modul, modul.dm.val_dataloader())

if __name__ == "__main__":
    main()
