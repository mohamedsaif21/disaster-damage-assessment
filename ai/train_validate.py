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

TRAIN_CSV = r"D:\Projects\Datasets\xBD\splits\train.csv"
VAL_CSV = r"D:\Projects\Datasets\xBD\splits\val.csv"

IMAGE_SIZE = 256
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================
# Setup
# ======================================

print("======================================")
print("xBD U-NET TRAINING + VALIDATION")
print("======================================")

print("Device:", DEVICE)


# ======================================
# Datasets
# ======================================

train_dataset = XBDDataset(
    DATASET_ROOT,
    TRAIN_CSV,
    image_size=IMAGE_SIZE,
)

val_dataset = XBDDataset(
    DATASET_ROOT,
    VAL_CSV,
    image_size=IMAGE_SIZE,
)


# ======================================
# DataLoaders
# ======================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
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
# Loss + Optimizer
# ======================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ======================================
# Training + Validation
# ======================================

for epoch in range(EPOCHS):

    # ----------------------------------
    # Training
    # ----------------------------------

    model.train()

    train_total_loss = 0.0
    train_batches = 0

    for images, targets in train_loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record
        train_total_loss += loss.item()
        train_batches += 1

    train_loss = train_total_loss / train_batches


    # ----------------------------------
    # Validation
    # ----------------------------------

    model.eval()

    val_total_loss = 0.0
    val_batches = 0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward
            outputs = model(images)

            # Loss
            loss = criterion(outputs, targets)

            # Record
            val_total_loss += loss.item()
            val_batches += 1

    val_loss = val_total_loss / val_batches


    # ----------------------------------
    # Epoch Results
    # ----------------------------------

    print()
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss:   {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")


# ======================================
# Complete
# ======================================

print()
print("======================================")
print("TRAINING + VALIDATION COMPLETE")
print("======================================")