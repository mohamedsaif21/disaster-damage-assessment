import sys

sys.path.insert(0, "ai")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import XBDDataset
from unet import UNet


# ======================================
# Configuration
# ======================================

DATASET_ROOT = r"D:\Projects\Datasets\xBD"

VAL_CSV = r"D:\Projects\Datasets\xBD\splits\val.csv"

IMAGE_SIZE = 256
BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================
# Setup
# ======================================

print("======================================")
print("xBD U-NET VALIDATION TEST")
print("======================================")

print("Device:", DEVICE)


dataset = XBDDataset(
    DATASET_ROOT,
    VAL_CSV,
    image_size=IMAGE_SIZE,
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


print("Validation samples:", len(dataset))
print("Batch size:", BATCH_SIZE)


# ======================================
# Model
# ======================================

model = UNet(
    in_channels=6,
    num_classes=5,
)

model = model.to(DEVICE)


# ======================================
# Loss
# ======================================

criterion = nn.CrossEntropyLoss()


# ======================================
# Validation
# ======================================

model.eval()

total_loss = 0.0
processed_batches = 0


with torch.no_grad():

    for batch_index, (images, targets) in enumerate(loader):

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, targets)

        batch_loss = loss.item()

        total_loss += batch_loss
        processed_batches += 1

        if processed_batches <= 5:
            print(
                f"Batch {batch_index + 1} "
                f"Loss: {batch_loss:.4f}"
            )

        # Validation sanity test
        if processed_batches == 5:
            break


# ======================================
# Average validation loss
# ======================================

average_loss = total_loss / processed_batches

print()
print("Processed batches:", processed_batches)
print(f"Average validation loss: {average_loss:.4f}")


# ======================================
# Complete
# ======================================

print()
print("======================================")
print("VALIDATION TEST COMPLETE")
print("======================================")