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
EPOCHS = 5

CHECKPOINT_DIR = r"ai\checkpoints"
BEST_MODEL_PATH = r"ai\checkpoints\best_model.pth"

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

class_weights = torch.tensor(
    [
        0.0078,  # Class 0: Background
        0.1653,  # Class 1: No Damage
        1.4010,  # Class 2: Minor Damage
        1.0366,  # Class 3: Major Damage
        2.3893,  # Class 4: Destroyed
    ],
    dtype=torch.float32,
).to(DEVICE)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ======================================
# Best Model Tracking
# ======================================

best_val_loss = float("inf")


# ======================================
# Training + Validation
# ======================================

for epoch in range(EPOCHS):

    print()
    print("--------------------------------------")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("--------------------------------------")


    # ==================================
    # TRAINING
    # ==================================

    model.train()

    train_total_loss = 0.0
    train_batches = 0


    for batch_index, (images, targets) in enumerate(train_loader):

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)


        # Forward

        outputs = model(images)


        # Loss

        loss = criterion(
            outputs,
            targets
        )


        if batch_index == 0:
            print(
                f"First training batch completed | "
                f"Loss: {loss.item():.4f}"
            )


        # Backward

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # Record loss

        train_total_loss += loss.item()

        train_batches += 1


        # Progress

        if (batch_index + 1) % 10 == 0:

            print(
                f"Training Batch "
                f"{batch_index + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )


    train_loss = train_total_loss / train_batches


    # ==================================
    # VALIDATION
    # ==================================

    model.eval()

    val_total_loss = 0.0
    val_batches = 0


    with torch.no_grad():

        for batch_index, (images, targets) in enumerate(val_loader):

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)


            # Forward

            outputs = model(images)


            # Loss

            loss = criterion(
                outputs,
                targets
            )


            # Record loss

            val_total_loss += loss.item()

            val_batches += 1


            # Progress

            if (batch_index + 1) % 50 == 0:

                print(
                    f"Validation Batch "
                    f"{batch_index + 1}/{len(val_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )


    val_loss = val_total_loss / val_batches


    # ==================================
    # Epoch Results
    # ==================================

    print()

    print(f"Epoch {epoch + 1} Results")

    print(f"Training Loss:   {train_loss:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")


    # ==================================
    # Save Best Model
    # ==================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }


        torch.save(
            checkpoint,
            BEST_MODEL_PATH,
        )


        print()

        print("Best model saved!")

        print(f"Path: {BEST_MODEL_PATH}")

        print(f"Validation Loss: {val_loss:.4f}")


# ======================================
# Complete
# ======================================

print()

print("======================================")
print("TRAINING + VALIDATION COMPLETE")
print("======================================")

print(f"Best Validation Loss: {best_val_loss:.4f}")

print(f"Best Model: {BEST_MODEL_PATH}")