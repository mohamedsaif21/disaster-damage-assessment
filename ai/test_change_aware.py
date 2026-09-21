import os
import torch
from torch.utils.data import DataLoader

from dataset_change_aware import XBDChangeAwareDataset
from unet_change_aware import UNetChangeAware


DATASET_ROOT = r"D:\Projects\Datasets\xBD"

TRAIN_CSV = os.path.join(
    DATASET_ROOT,
    "splits",
    "train.csv"
)


def main():

    print("=" * 70)
    print("CHANGE-AWARE PIPELINE TEST")
    print("=" * 70)

    dataset = XBDChangeAwareDataset(
        DATASET_ROOT,
        TRAIN_CSV,
        image_size=256
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    images, targets = next(
        iter(loader)
    )

    print()
    print("Dataset size :", len(dataset))
    print("Images shape :", images.shape)
    print("Target shape :", targets.shape)
    print("Image dtype  :", images.dtype)
    print("Target dtype :", targets.dtype)

    print()
    print(
        "Target classes:",
        torch.unique(targets).tolist()
    )

    model = UNetChangeAware(
        in_channels=9,
        num_classes=5
    )

    output = model(images)

    print()
    print("Model input  :", images.shape)
    print("Model output :", output.shape)
    print("Output dtype :", output.dtype)

    print()
    print("=" * 70)
    print("CHANGE-AWARE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()