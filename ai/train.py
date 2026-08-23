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

IMAGE_SIZE = 256
BATCH_SIZE = 2
LEARNING_RATE = 1e-4

EPOCHS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================
# Setup
# ======================================

print("======================================")
print("xBD U-NET TRAINING")
print("======================================")

print("Device:", DEVICE)

dataset = XBDDataset(
    DATASET_ROOT,
    TRAIN_CSV,
    image_size=IMAGE_SIZE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


print("Training samples:", len(dataset))
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
# Training
# ======================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    print()
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    for batch_index, (images, targets) in enumerate(loader):

        if batch_index >= 5:
            break

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

        batch_loss = loss.item()

        total_loss += batch_loss

        print(
        f"Batch {batch_index + 1} "
        f"Loss: {batch_loss:.4f}"
            )

    processed_batches = batch_index + 1
    average_loss = total_loss / processed_batches

    print()
    print(f"Average loss: {average_loss:.4f}")


print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")