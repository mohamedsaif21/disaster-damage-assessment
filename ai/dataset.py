from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class XBDDataset(Dataset):
    """
    PyTorch Dataset for the xBD disaster damage dataset.

    Each sample contains:
        - pre-disaster RGB image
        - post-disaster RGB image
        - post-disaster damage mask

    Input:
        6 channels = 3 pre + 3 post

    Target:
        5 classes
        0 = background
        1 = no-damage
        2 = minor-damage
        3 = major-damage
        4 = destroyed
    """

    def __init__(
        self,
        dataset_root,
        split_file,
        image_size=256,
    ):
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / "train" / "images"
        self.target_dir = self.dataset_root / "train" / "targets"

        self.samples = pd.read_csv(split_file)
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = self.samples.iloc[index]["sample_id"]

        # File paths
        pre_path = self.image_dir / f"{sample_id}_pre_disaster.png"
        post_path = self.image_dir / f"{sample_id}_post_disaster.png"
        target_path = (
            self.target_dir / f"{sample_id}_post_disaster_target.png"
        )

        # Load images
        pre_image = Image.open(pre_path).convert("RGB")
        post_image = Image.open(post_path).convert("RGB")

        # Load target as single-channel image
        target = Image.open(target_path)

        # Resize images
        pre_image = pre_image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        post_image = post_image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        # IMPORTANT:
        # Masks must use nearest-neighbor interpolation.
        # Otherwise class IDs can become corrupted.
        target = target.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )

        # Convert images to NumPy
        pre_array = np.array(pre_image, dtype=np.float32)
        post_array = np.array(post_image, dtype=np.float32)

        # Convert target to integer class IDs
        target_array = np.array(target, dtype=np.int64)

        # Normalize RGB images to [0, 1]
        pre_array = pre_array / 255.0
        post_array = post_array / 255.0

        # HWC -> CHW
        pre_array = np.transpose(pre_array, (2, 0, 1))
        post_array = np.transpose(post_array, (2, 0, 1))

        # Combine:
        # [3, H, W] + [3, H, W]
        # = [6, H, W]
        image = np.concatenate(
            [pre_array, post_array],
            axis=0,
        )

        # NumPy -> PyTorch
        image = torch.from_numpy(image).float()
        target = torch.from_numpy(target_array).long()

        return image, target