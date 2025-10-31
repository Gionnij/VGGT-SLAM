import json
from pathlib import Path
from random import Random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def rgb_to_consecutive_ids(rgb_label: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W, 3) RGB label image to (H, W) with consecutive integer IDs.
    
    Parameters
    ----------
    rgb_label : np.ndarray
        An array of shape (H, W, 3), dtype uint8.
    
    Returns
    -------
    label_ids : np.ndarray
        An array of shape (H, W) with consecutive integer class IDs.
    """
    assert rgb_label.ndim == 3 and rgb_label.shape[2] == 3, "Expected shape (H, W, 3)"
    assert rgb_label.dtype == np.uint8, "Expected dtype uint8"

    # Flatten HxWx3 to Nx3 and compute a unique value for each RGB triplet
    flat_rgb = rgb_label.reshape(-1, 3)
    rgb_as_int = (flat_rgb[:, 0].astype(np.uint32) << 16) | \
                (flat_rgb[:, 1].astype(np.uint32) << 8)  | \
                (flat_rgb[:, 2].astype(np.uint32))

    # Map unique RGB ints to consecutive class IDs
    unique_rgb, inverse = np.unique(rgb_as_int, return_inverse=True)
    label_ids = inverse.reshape(rgb_label.shape[:2])

    return label_ids

class A2D2CameraDataset(Dataset):
    """
    Quick loader for Audi A2D2 camera-lidar-semantic dataset
    -------------------------------------------------------
    Expects the original folder tree:

      camera_lidar_semantic/
        20180807_145028/
            camera/cam_front_center/*.png
            label/cam_front_center/*.png   (uint8 semantic ids)
            lidar/ ...

    Args
    ----
    root        : Path to `camera_lidar_semantic`
    camera_name : Which sub-camera to load (default: 'cam_front_center')
    transform   : Callable applied to RGB
    target_tf   : Callable applied to mask
    """

    def __init__(
        self,
        root: str | Path,
        camera_name: str = "cam_front_center",
        max_size: int = -1,
        transforms=None,
        transform=None,
        target_tf=None,
        seed: int = 42,  # <--- New argument
    ):
        self.root = Path(root).expanduser()
        self.camera_name = camera_name
        self.transform = transforms if transforms else transform
        self.target_tf = target_tf
        self.max_size = max_size
        self.seed = seed

        self.samples: list[Tuple[Path, Path]] = []
        self._index_dataset()

        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root} for {camera_name}")

    # ------------------------------------------------------------------ #
    def _index_dataset(self) -> None:
        seq_dirs = [d for d in self.root.iterdir() if d.is_dir()]

        all_samples = []
        for seq_dir in sorted(seq_dirs):
            rgb_dir = seq_dir / "camera" / self.camera_name
            lbl_dir = seq_dir / "label" / self.camera_name
            if not rgb_dir.is_dir() or not lbl_dir.is_dir():
                continue

            for rgb_path in sorted(rgb_dir.glob("*.png")):
                lbl_path = lbl_dir / rgb_path.name.replace("_camera_", "_label_")
                if lbl_path.exists():
                    all_samples.append((rgb_path, lbl_path))
                else:
                    print(f"no label found at {lbl_path} for image {rgb_path}")

        

        if self.max_size > 0:
            # Deterministic shuffle
            rng = Random(self.seed)
            rng.shuffle(all_samples)
            all_samples = all_samples[:self.max_size]

        self.samples = all_samples


    
    

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        rgb_path, lbl_path = self.samples[idx]

        # --- load ---
        img = Image.open(rgb_path).convert("RGB")
        lbl = Image.open(lbl_path)  # stay in uint8 mode (no convert)

        # --- optional transforms ---
        
        # if self.target_tf
        #     lbl = self.target_tf(lbl)
        # else:  # default: convert mask to tensor of ints
        lbl = np.array(lbl).astype(np.uint8)
        lbl = torch.from_numpy(rgb_to_consecutive_ids(lbl)) # convert RGB to consecutive IDs

        meta = {
            "sequence": rgb_path.parts[-4],          # e.g. 20180807_145028
            "camera": self.camera_name,
            "frame_id": rgb_path.stem.split("_")[-1] # e.g. 000001546
        }
        res = {"image": img, "semantic": lbl, "meta": meta,"original_image": img.copy(),}
        if self.transform:
            res = self.transform(res)
        return res
    
if __name__ == "__main__":
    from torchvision.transforms import ToTensor, Compose, Resize

    root = "/Datasets/A2D2/camera_lidar_semantic"
    ds = A2D2CameraDataset(
            root,
            camera_name="cam_front_center",
            transform=None  # example
        )
    print(len(ds))
    sample = ds[0]
    for k,v in sample.items():
        print(k)
        if isinstance(v, torch.Tensor):
            print(v.shape)
