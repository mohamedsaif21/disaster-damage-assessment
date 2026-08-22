import sys

sys.path.insert(0, "ai")

import torch
from torch.utils.data import DataLoader

from dataset import XBDDataset


DATASET_ROOT = r"D:\Projects\Datasets\xBD"
TRAIN_SPLIT = r"D:\Projects\Datasets\xBD\splits\train.csv"


dataset = XBDDataset(
    dataset_root=DATASET_ROOT,
    split_file=TRAIN_SPLIT,
    image_size=256,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)


images, targets = next(iter(loader))

print("======================================")
print("xBD DATALOADER TEST")
print("======================================")
print("Dataset size :", len(dataset))
print("Batch images :", images.shape)
print("Batch targets:", targets.shape)
print("Image dtype  :", images.dtype)
print("Target dtype :", targets.dtype)
print("Image range  :", images.min().item(), "to", images.max().item())
print("Target classes:", torch.unique(targets).tolist())
print("======================================")