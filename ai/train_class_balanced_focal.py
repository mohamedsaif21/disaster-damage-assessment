import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.dataset import XBDDataset
from ai.unet import UNet


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

NUM_CLASSES = 5
IMAGE_SIZE = 256

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

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "ai", "checkpoints")
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model_class_balanced_focal.pth"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Class weights
# ---------------------------------------------------------

# Background      = 0.25
# No Damage       = 0.75
# Minor Damage    = 1.50
# Major Damage    = 1.75
# Destroyed       = 2.00

CLASS_WEIGHTS = torch.tensor(
    [0.25, 0.75, 1.50, 1.75, 2.00],
    dtype=torch.float32
)


# ---------------------------------------------------------
# Corrected Class-Balanced Focal Loss
# ---------------------------------------------------------

class ClassBalancedFocalLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        gamma=2.0,
        ignore_index=None
    ):
        super().__init__()

        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):

        # -------------------------------------------------
        # logits:
        # [B, C, H, W]
        #
        # targets:
        # [B, H, W]
        # -------------------------------------------------

        # Calculate standard CE WITHOUT class weights.
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index
            if self.ignore_index is not None else -100
        )

        # -------------------------------------------------
        # Calculate pt from UNWEIGHTED CE
        #
        # pt = probability assigned to the correct class
        # -------------------------------------------------

        pt = torch.exp(-ce_loss)

        # -------------------------------------------------
        # Focal factor
        # -------------------------------------------------

        focal_factor = (1.0 - pt) ** self.gamma

        # -------------------------------------------------
        # Get class weight for every target pixel
        # -------------------------------------------------

        pixel_weights = self.class_weights[targets]

        # -------------------------------------------------
        # Apply class balancing separately
        # -------------------------------------------------

        focal_loss = (
            pixel_weights
            * focal_factor
            * ce_loss
        )

        return focal_loss.mean()


# ---------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------

class DiceLoss(nn.Module):

    def __init__(self, num_classes, smooth=1.0):
        super().__init__()

        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):

        # Convert logits → probabilities
        probabilities = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes
        )

        # [B, H, W, C] → [B, C, H, W]
        targets_one_hot = targets_one_hot.permute(
            0, 3, 1, 2
        ).float()

        # Flatten spatial dimensions
        probabilities = probabilities.contiguous().view(
            probabilities.size(0),
            probabilities.size(1),
            -1
        )

        targets_one_hot = targets_one_hot.contiguous().view(
            targets_one_hot.size(0),
            targets_one_hot.size(1),
            -1
        )

        # Intersection
        intersection = (
            probabilities * targets_one_hot
        ).sum(dim=2)

        # Total areas
        probability_sum = probabilities.sum(dim=2)
        target_sum = targets_one_hot.sum(dim=2)

        # Dice score
        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probability_sum
            + target_sum
            + self.smooth
        )

        # Dice loss
        dice_loss = 1.0 - dice.mean()

        return dice_loss


# ---------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------

class CombinedLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        num_classes,
        gamma=2.0,
        focal_weight=0.70,
        dice_weight=0.30
    ):
        super().__init__()

        self.focal_loss = ClassBalancedFocalLoss(
            class_weights=class_weights,
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
            + self.dice_weight * dice
        )

        return total, focal, dice


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate(
    model,
    dataloader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0

    with torch.no_grad():

        for images, targets in dataloader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)

            loss, focal, dice = criterion(
                outputs,
                targets
            )

            total_loss += loss.item()
            total_focal += focal.item()
            total_dice += dice.item()

    num_batches = len(dataloader)

    return (
        total_loss / num_batches,
        total_focal / num_batches,
        total_dice / num_batches
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("EXPERIMENT 3")
    print("Class-Balanced Focal + Dice Training")
    print("=" * 70)

    print()
    print(f"Device       : {DEVICE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Image size   : {IMAGE_SIZE}")
    print()

    print("Class weights:")
    print("  Background   : 0.25")
    print("  No Damage    : 0.75")
    print("  Minor Damage : 1.50")
    print("  Major Damage : 1.75")
    print("  Destroyed    : 2.00")
    print()

    print("Loss:")
    print("  70% Corrected Class-Balanced Focal")
    print("  30% Dice")
    print("  Focal gamma = 2.0")
    print()

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print("Loading datasets...")

    train_dataset = XBDDataset(
        dataset_root=DATASET_ROOT,
        split_file=TRAIN_CSV,
        image_size=IMAGE_SIZE
    )

    val_dataset = XBDDataset(
        dataset_root=DATASET_ROOT,
        split_file=VAL_CSV,
        image_size=IMAGE_SIZE
    )

    print(f"Training samples  : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

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

    print(f"Training batches  : {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print()

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print("Creating U-Net...")

    model = UNet(
        in_channels=6,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    class_weights = CLASS_WEIGHTS.to(DEVICE)

    criterion = CombinedLoss(
        class_weights=class_weights,
        num_classes=NUM_CLASSES,
        gamma=2.0,
        focal_weight=0.70,
        dice_weight=0.30
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_val_loss = float("inf")

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0
        running_focal = 0.0
        running_dice = 0.0

        total_batches = len(train_loader)

        for batch_idx, (images, targets) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # ---------------------------------------------
            # Clear gradients
            # ---------------------------------------------

            optimizer.zero_grad()

            # ---------------------------------------------
            # Forward
            # ---------------------------------------------

            outputs = model(images)

            # ---------------------------------------------
            # Loss
            # ---------------------------------------------

            loss, focal, dice = criterion(
                outputs,
                targets
            )

            # ---------------------------------------------
            # Backpropagation
            # ---------------------------------------------

            loss.backward()

            optimizer.step()

            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            running_loss += loss.item()
            running_focal += focal.item()
            running_dice += dice.item()

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if batch_idx % 10 == 0:

                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} "
                    f"| Batch {batch_idx}/{total_batches} "
                    f"| Loss {loss.item():.4f}"
                )

        # -------------------------------------------------
        # Average training loss
        # -------------------------------------------------

        train_loss = running_loss / total_batches
        train_focal = running_focal / total_batches
        train_dice = running_dice / total_batches

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        val_loss, val_focal, val_dice = validate(
            model,
            val_loader,
            criterion
        )

        # -------------------------------------------------
        # Epoch summary
        # -------------------------------------------------

        print()
        print("-" * 70)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        print(
            f"Training Loss : {train_loss:.4f}"
        )

        print(
            f"  Focal       : {train_focal:.4f}"
        )

        print(
            f"  Dice        : {train_dice:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"  Focal        : {val_focal:.4f}"
        )

        print(
            f"  Dice         : {val_dice:.4f}"
        )

        # -------------------------------------------------
        # Save best checkpoint
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_focal": train_focal,
                    "train_dice": train_dice,
                    "val_focal": val_focal,
                    "val_dice": val_dice,
                    "class_weights": CLASS_WEIGHTS,
                    "gamma": 2.0
                },
                BEST_MODEL_PATH
            )

            print()
            print("BEST MODEL SAVED")
            print(
                f"Path: {BEST_MODEL_PATH}"
            )
            print(
                f"Best Validation Loss: {best_val_loss:.4f}"
            )

        print("-" * 70)
        print()

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best Validation Loss: {best_val_loss:.4f}"
    )

    print(
        f"Best Model: {BEST_MODEL_PATH}"
    )

    print()


if __name__ == "__main__":
    main()