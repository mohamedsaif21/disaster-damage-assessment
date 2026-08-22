from pathlib import Path

import numpy as np

from dataset import XBDDataset


# CHANGE THIS if your dataset is somewhere else
DATASET_ROOT = r"D:\Projects\Datasets\xBD"

TRAIN_SPLIT = Path(DATASET_ROOT) / "splits" / "train.csv"


dataset = XBDDataset(
    dataset_root=DATASET_ROOT,
    split_file=TRAIN_SPLIT,
    image_size=256,
)

print("======================================")
print("xBD PYTORCH DATASET TEST")
print("======================================")

print("Dataset size:", len(dataset))

image, target = dataset[0]

print()
print("First sample")
print("-----------------------------")

print("Input shape:", tuple(image.shape))
print("Input dtype:", image.dtype)

print("Target shape:", tuple(target.shape))
print("Target dtype:", target.dtype)

print()
print("Input range:")
print("Min:", image.min().item())
print("Max:", image.max().item())

print()
print("Target classes:")
print(np.unique(target.numpy()))

print()
print("Target class counts:")

values, counts = np.unique(
    target.numpy(),
    return_counts=True,
)

for value, count in zip(values, counts):
    print(f"Class {value}: {count:,}")

print()
print("Dataset loader test complete.")