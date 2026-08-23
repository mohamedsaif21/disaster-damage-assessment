import sys

sys.path.insert(0, "ai")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import XBDDataset
from unet import UNet


DATASET_ROOT = r"D:\Projects\Datasets\xBD"
TRAIN_CSV = r"D:\Projects\Datasets\xBD\splits\train.csv"


print("======================================")
print("U-NET TRAINING STEP TEST")
print("======================================")


# Load dataset
dataset = XBDDataset(
    DATASET_ROOT,
    TRAIN_CSV,
    image_size=256,
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

images, targets = next(iter(loader))

print("Images :", images.shape)
print("Targets:", targets.shape)


# Create model
model = UNet(
    in_channels=6,
    num_classes=5,
)


# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)


# Forward pass
outputs = model(images)

print("Outputs:", outputs.shape)


# Calculate loss
loss = criterion(outputs, targets)

print("Loss:", loss.item())


# Backward pass
optimizer.zero_grad()

loss.backward()

print("Backward pass: OK")


# Update model weights
optimizer.step()

print("Optimizer step: OK")


print("======================================")
print("TRAINING STEP PASSED")
print("======================================")