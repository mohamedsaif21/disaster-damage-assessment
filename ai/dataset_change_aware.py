import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class XBDChangeAwareDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        split_file,
        image_size=256
    ):
        self.dataset_root = dataset_root
        self.image_size = image_size

        self.image_dir = os.path.join(
            dataset_root,
            "train",
            "images"
        )

        self.target_dir = os.path.join(
            dataset_root,
            "train",
            "targets"
        )

        self.df = pd.read_csv(split_file)

    def __len__(self):
        return len(self.df)

    def load_rgb(self, path):

        image = Image.open(path).convert("RGB")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = torch.from_numpy(
            image
        ).permute(2, 0, 1)

        return image

    def load_mask(self, path):

        mask = Image.open(path)

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        mask = np.asarray(
            mask,
            dtype=np.int64
        )

        return torch.from_numpy(mask)

    def __getitem__(self, index):

        sample_id = str(
            self.df.iloc[index, 0]
        )

        pre_path = os.path.join(
            self.image_dir,
            f"{sample_id}_pre_disaster.png"
        )

        post_path = os.path.join(
            self.image_dir,
            f"{sample_id}_post_disaster.png"
        )

        target_path = os.path.join(
            self.target_dir,
            f"{sample_id}_post_disaster_target.png"
        )

        pre = self.load_rgb(pre_path)
        post = self.load_rgb(post_path)

        # Explicit visual change
        difference = torch.abs(
            post - pre
        )

        # 3 + 3 + 3 = 9 channels
        image = torch.cat(
            [
                pre,
                post,
                difference
            ],
            dim=0
        )

        target = self.load_mask(
            target_path
        )

        return image, target