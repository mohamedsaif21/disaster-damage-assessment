import sys

sys.path.insert(0, "ai")

import torch
import numpy as np
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
MODEL_PATH = r"ai\checkpoints\best_model.pth"

NUM_CLASSES = 5

CLASS_NAMES = [
    "Background",
    "No Damage",
    "Minor Damage",
    "Major Damage",
    "Destroyed",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================
# Setup
# ======================================

print("======================================")
print("xBD U-NET MODEL EVALUATION")
print("======================================")

print("Device:", DEVICE)
print("Model:", MODEL_PATH)


# ======================================
# Dataset
# ======================================

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


# ======================================
# Model
# ======================================

model = UNet(
    in_channels=6,
    num_classes=NUM_CLASSES,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ======================================
# Confusion Matrix
# ======================================

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64,
)


# ======================================
# Evaluation
# ======================================

print()
print("======================================")
print("EVALUATING VALIDATION DATA")
print("======================================")


with torch.no_grad():

    for batch_index, (images, targets) in enumerate(loader):

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        # Move to CPU
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        # Update confusion matrix
        for target, prediction in zip(
            targets,
            predictions,
        ):

            target = target.flatten()
            prediction = prediction.flatten()

            for true_class in range(NUM_CLASSES):

                mask = target == true_class

                predicted_classes = prediction[mask]

                for predicted_class in range(NUM_CLASSES):

                    confusion_matrix[
                        true_class,
                        predicted_class
                    ] += np.sum(
                        predicted_classes == predicted_class
                    )

        if (batch_index + 1) % 25 == 0:

            print(
                f"Processed Batch "
                f"{batch_index + 1}/{len(loader)}"
            )


# ======================================
# Metrics
# ======================================

print()
print("======================================")
print("EVALUATION RESULTS")
print("======================================")


# --------------------------------------
# Pixel Accuracy
# --------------------------------------

correct_pixels = np.trace(confusion_matrix)

total_pixels = np.sum(confusion_matrix)

pixel_accuracy = (
    correct_pixels / total_pixels
    if total_pixels > 0
    else 0.0
)


print()
print(
    f"Pixel Accuracy: "
    f"{pixel_accuracy:.4f}"
)


# --------------------------------------
# Per-class IoU
# --------------------------------------

ious = []

print()
print("Per-Class IoU:")
print("--------------------------------------")

for class_id in range(NUM_CLASSES):

    true_positive = confusion_matrix[
        class_id,
        class_id
    ]

    false_positive = (
        np.sum(confusion_matrix[:, class_id])
        - true_positive
    )

    false_negative = (
        np.sum(confusion_matrix[class_id, :])
        - true_positive
    )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    if union > 0:

        iou = true_positive / union
        ious.append(iou)

    else:

        iou = float("nan")

    print(
        f"Class {class_id} "
        f"({CLASS_NAMES[class_id]}): "
        f"{iou:.4f}"
    )


# --------------------------------------
# Mean IoU
# --------------------------------------

mean_iou = np.nanmean(ious)


print()
print("--------------------------------------")

print(
    f"Mean IoU: {mean_iou:.4f}"
)


# ======================================
# Confusion Matrix
# ======================================

print()
print("======================================")
print("CONFUSION MATRIX")
print("======================================")

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(confusion_matrix)


# ======================================
# Complete
# ======================================

print()
print("======================================")
print("MODEL EVALUATION COMPLETE")
print("======================================")