import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset_change_aware import XBDChangeAwareDataset
from unet_change_aware import UNetChangeAware


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"D:\Projects\Datasets\xBD"

TRAIN_CSV = os.path.join(
    DATASET_ROOT,
    "splits",
    "train.csv"
)

VAL_CSV = os.path.join(
    DATASET_ROOT,
    "splits",
    "val.csv"
)

CHECKPOINT_DIR = "ai/checkpoints"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model_change_aware.pth"
)

IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-4

NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS WEIGHTS
# Same weights as Experiment 1
# ============================================================

CLASS_WEIGHTS = torch.tensor(
    [
        0.25,
        0.75,
        1.50,
        1.75,
        2.00
    ],
    dtype=torch.float32
)


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    logits,
    targets,
    num_classes=5
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    targets_one_hot = F.one_hot(
        targets,
        num_classes=num_classes
    )

    targets_one_hot = targets_one_hot.permute(
        0,
        3,
        1,
        2
    ).float()

    smooth = 1e-6

    intersection = (
        probabilities * targets_one_hot
    ).sum(
        dim=(0, 2, 3)
    )

    denominator = (
        probabilities.sum(
            dim=(0, 2, 3)
        )
        +
        targets_one_hot.sum(
            dim=(0, 2, 3)
        )
    )

    dice = (
        (2 * intersection + smooth)
        /
        (denominator + smooth)
    )

    return 1 - dice.mean()


# ============================================================
# COMBINED LOSS
# 70% Weighted CE + 30% Dice
# ============================================================

def combined_loss(
    logits,
    targets,
    class_weights
):

    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights
    )

    dice = dice_loss(
        logits,
        targets,
        NUM_CLASSES
    )

    return (
        0.70 * ce
        +
        0.30 * dice
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("EXPERIMENT 4 — CHANGE-AWARE U-NET")
    print("=" * 75)

    print()
    print("Device       :", DEVICE)
    print("Epochs       :", EPOCHS)
    print("Batch size   :", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Image size   :", IMAGE_SIZE)
    print("Input        : 9 channels")
    print("Output       : 5 classes")
    print()
    print("Input channels:")
    print("  3 × Pre-disaster RGB")
    print("  3 × Post-disaster RGB")
    print("  3 × |Post - Pre|")
    print()
    print("Loss:")
    print("  70% Weighted Cross Entropy")
    print("  30% Dice Loss")
    print()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = XBDChangeAwareDataset(
        DATASET_ROOT,
        TRAIN_CSV,
        image_size=IMAGE_SIZE
    )

    val_dataset = XBDChangeAwareDataset(
        DATASET_ROOT,
        VAL_CSV,
        image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Training samples  :", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print(
        "Training batches   :",
        len(train_loader)
    )
    print(
        "Validation batches :",
        len(val_loader)
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = UNetChangeAware(
        in_channels=9,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    class_weights = CLASS_WEIGHTS.to(
        DEVICE
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        print()
        print(
            "=" * 75
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            "=" * 75
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0

        for batch_index, (
            images,
            targets
        ) in enumerate(train_loader):

            images = images.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = combined_loss(
                outputs,
                targets,
                class_weights
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if (
                batch_index + 1
            ) % 100 == 0:

                print(
                    f"  Batch "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)}"
                    f" | Loss: "
                    f"{loss.item():.4f}"
                )

        train_loss /= len(
            train_loader
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for images, targets in val_loader:

                images = images.to(
                    DEVICE
                )

                targets = targets.to(
                    DEVICE
                )

                outputs = model(
                    images
                )

                loss = combined_loss(
                    outputs,
                    targets,
                    class_weights
                )

                val_loss += loss.item()

        val_loss /= len(
            val_loader
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch + 1}"
        )

        print(
            f"Training Loss   : "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss : "
            f"{val_loss:.4f}"
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "train_loss":
                        train_loss,
                    "val_loss":
                        val_loss
                },
                CHECKPOINT_PATH
            )

            print()
            print(
                "✓ Best model saved:"
            )

            print(
                CHECKPOINT_PATH
            )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("EXPERIMENT 4 TRAINING COMPLETE")
    print("=" * 75)

    print()
    print(
        "Best Validation Loss:",
        f"{best_val_loss:.4f}"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


if __name__ == "__main__":
    main()