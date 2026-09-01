import sys
from pathlib import Path

sys.path.insert(0, "ai")

import torch
import torch.nn as nn
import torch.nn.functional as F
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
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)


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
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ======================================
# Model
# ======================================

model = UNet(
    in_channels=6,
    num_classes=5,
)

model = model.to(DEVICE)


# ======================================
# Loss Function
# ======================================

# Softer class weights.
#
# 0 = Background
# 1 = No Damage
# 2 = Minor Damage
# 3 = Major Damage
# 4 = Destroyed

class_weights = torch.tensor(
    [
        0.25,
        0.75,
        1.50,
        1.75,
        2.00,
    ],
    dtype=torch.float32,
).to(DEVICE)


print()
print("Class weights:")
print("Class 0 (Background):", class_weights[0].item())
print("Class 1 (No Damage):", class_weights[1].item())
print("Class 2 (Minor Damage):", class_weights[2].item())
print("Class 3 (Major Damage):", class_weights[3].item())
print("Class 4 (Destroyed):", class_weights[4].item())


# ======================================
# Weighted Cross Entropy
# ======================================

cross_entropy = nn.CrossEntropyLoss(
    weight=class_weights
)


# ======================================
# Dice Loss
# ======================================

def dice_loss(outputs, targets, num_classes=5):

    probabilities = F.softmax(outputs, dim=1)

    targets_one_hot = F.one_hot(
        targets,
        num_classes=num_classes
    )

    targets_one_hot = targets_one_hot.permute(
        0, 3, 1, 2
    ).float()

    smooth = 1e-6

    intersection = (
        probabilities * targets_one_hot
    ).sum(dim=(0, 2, 3))

    denominator = (
        probabilities.sum(dim=(0, 2, 3))
        + targets_one_hot.sum(dim=(0, 2, 3))
    )

    dice = (
        (2.0 * intersection + smooth)
        / (denominator + smooth)
    )

    return 1.0 - dice.mean()


# ======================================
# Combined Loss
# ======================================

def combined_loss(outputs, targets):

    ce_loss = cross_entropy(
        outputs,
        targets
    )

    d_loss = dice_loss(
        outputs,
        targets,
        num_classes=5
    )

    total_loss = (
        0.7 * ce_loss
        + 0.3 * d_loss
    )

    return total_loss


# ======================================
# Optimizer
# ======================================

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

    for batch_index, (images, targets) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)


        # Forward pass

        outputs = model(images)


        # Combined loss

        loss = combined_loss(
            outputs,
            targets
        )


        # Backward pass

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # Record loss

        train_total_loss += loss.item()
        train_batches += 1


        # First batch

        if batch_index == 0:

            print(
                f"First training batch completed | "
                f"Loss: {loss.item():.4f}"
            )


        # Progress

        if (batch_index + 1) % 10 == 0:

            print(
                f"Training Batch "
                f"{batch_index + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )


    train_loss = (
        train_total_loss
        / train_batches
    )


    # ==================================
    # VALIDATION
    # ==================================

    model.eval()

    val_total_loss = 0.0
    val_batches = 0

    with torch.no_grad():

        for batch_index, (images, targets) in enumerate(
            val_loader
        ):

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)


            # Forward pass

            outputs = model(images)


            # Combined loss

            loss = combined_loss(
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


    val_loss = (
        val_total_loss
        / val_batches
    )


    # ==================================
    # Epoch Results
    # ==================================

    print()

    print(f"Epoch {epoch + 1} Results")

    print(
        f"Training Loss:   {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )


    # ==================================
    # Save Best Model
    # ==================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        Path(CHECKPOINT_DIR).mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint = {

            "epoch": epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,
        }


        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )


        print()

        print("Best model saved!")

        print(
            f"Path: {BEST_MODEL_PATH}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )


# ======================================
# Complete
# ======================================

print()

print("======================================")
print("TRAINING + VALIDATION COMPLETE")
print("======================================")

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best Model: "
    f"{BEST_MODEL_PATH}"
)