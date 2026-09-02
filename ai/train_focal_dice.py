import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from ai.dataset import XBDDataset
from ai.unet import UNet


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

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "ai",
    "checkpoints"
)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model_focal_dice.pth"
)

IMAGE_SIZE = 256

EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 1e-4

NUM_CLASSES = 5

# Focal Loss parameters
FOCAL_GAMMA = 2.0

# Loss combination
FOCAL_WEIGHT = 0.70
DICE_WEIGHT = 0.30

# Class weights
CLASS_WEIGHTS = torch.tensor(
    [
        0.25,   # Background
        0.75,   # No Damage
        1.50,   # Minor Damage
        1.75,   # Major Damage
        2.00    # Destroyed
    ],
    dtype=torch.float32
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        weight=None,
        gamma=2.0
    ):
        super().__init__()

        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):

        # Standard cross entropy per pixel
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none"
        )

        # Probability of the correct class
        pt = torch.exp(-ce_loss)

        # Focal weighting
        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        num_classes=5,
        smooth=1e-6
    ):
        super().__init__()

        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        total_dice_loss = 0.0

        for class_id in range(self.num_classes):

            prediction = probabilities[:, class_id]

            target = (
                targets == class_id
            ).float()

            prediction = prediction.contiguous().view(
                prediction.size(0),
                -1
            )

            target = target.contiguous().view(
                target.size(0),
                -1
            )

            intersection = (
                prediction * target
            ).sum(dim=1)

            denominator = (
                prediction.sum(dim=1)
                +
                target.sum(dim=1)
            )

            dice = (
                (2.0 * intersection + self.smooth)
                /
                (denominator + self.smooth)
            )

            total_dice_loss += (
                1.0 - dice
            ).mean()

        return (
            total_dice_loss
            /
            self.num_classes
        )


# ============================================================
# COMBINED LOSS
# ============================================================

class FocalDiceLoss(nn.Module):

    def __init__(
        self,
        num_classes=5,
        class_weights=None,
        gamma=2.0,
        focal_weight=0.70,
        dice_weight=0.30
    ):
        super().__init__()

        self.focal_loss = FocalLoss(
            weight=class_weights,
            gamma=gamma
        )

        self.dice_loss = DiceLoss(
            num_classes=num_classes
        )

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        focal = self.focal_loss(
            logits,
            targets
        )

        dice = self.dice_loss(
            logits,
            targets
        )

        total = (
            self.focal_weight * focal
            +
            self.dice_weight * dice
        )

        return total


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 70)
print("xBD FOCAL + DICE TRAINING")
print("=" * 70)

print(f"Device          : {DEVICE}")
print(f"Epochs          : {EPOCHS}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"Image size      : {IMAGE_SIZE}")
print()

print("Loss:")
print(f"  Focal weight  : {FOCAL_WEIGHT}")
print(f"  Dice weight   : {DICE_WEIGHT}")
print(f"  Focal gamma   : {FOCAL_GAMMA}")
print()

print("Class weights:")

for class_id, weight in enumerate(CLASS_WEIGHTS):

    if class_id == 0:
        name = "Background"
    elif class_id == 1:
        name = "No Damage"
    elif class_id == 2:
        name = "Minor Damage"
    elif class_id == 3:
        name = "Major Damage"
    else:
        name = "Destroyed"

    print(
        f"  Class {class_id} "
        f"{name:15s}: {weight.item():.2f}"
    )

print()


train_dataset = XBDDataset(
    DATASET_ROOT,
    TRAIN_CSV,
    image_size=IMAGE_SIZE
)

val_dataset = XBDDataset(
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


print(f"Training samples : {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Training batches : {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print()


# ============================================================
# MODEL
# ============================================================

model = UNet(
    in_channels=6,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

class_weights = CLASS_WEIGHTS.to(DEVICE)

criterion = FocalDiceLoss(
    num_classes=NUM_CLASSES,
    class_weights=class_weights,
    gamma=FOCAL_GAMMA,
    focal_weight=FOCAL_WEIGHT,
    dice_weight=DICE_WEIGHT
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    for batch_idx, (images, targets) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

        if (
            batch_idx + 1
        ) % 100 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = (
        running_train_loss
        /
        len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

            running_val_loss += loss.item()

    val_loss = (
        running_val_loss
        /
        len(val_loader)
    )


    # --------------------------------------------------------
    # PRINT EPOCH RESULTS
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss   : {train_loss:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_type": "Focal + Dice",
            "focal_gamma": FOCAL_GAMMA,
            "focal_weight": FOCAL_WEIGHT,
            "dice_weight": DICE_WEIGHT,
            "class_weights": CLASS_WEIGHTS.tolist()
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )

        print(
            f"Best model saved: {BEST_MODEL_PATH}"
        )

    print("-" * 70)
    print()


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("FOCAL + DICE TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Loss: {best_val_loss:.4f}"
)

print(
    f"Best Model: {BEST_MODEL_PATH}"
)

print()
print(
    "Next step: evaluate this model using the same "
    "validation metrics as the baseline."
)