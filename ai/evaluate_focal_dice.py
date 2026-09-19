import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from ai.dataset import XBDDataset
from ai.unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"D:\Projects\Datasets\xBD"

VAL_CSV = os.path.join(
    DATASET_ROOT,
    "splits",
    "val.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "checkpoints",
    "best_model_focal_dice.pth"
)

BATCH_SIZE = 2
IMAGE_SIZE = 256
NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "Background",
    "No Damage",
    "Minor Damage",
    "Major Damage",
    "Destroyed"
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FOCAL + DICE MODEL EVALUATION")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Validation   : {VAL_CSV}")
print(f"Model        : {MODEL_PATH}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Image size   : {IMAGE_SIZE}")
print()


# ============================================================
# DATASET
# ============================================================

val_dataset = XBDDataset(
    DATASET_ROOT,
    VAL_CSV,
    image_size=IMAGE_SIZE
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"Validation samples: {len(val_dataset)}"
)

print(
    f"Validation batches: {len(val_loader)}"
)

print()


# ============================================================
# MODEL
# ============================================================

model = UNet(
    in_channels=6,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# ============================================================
# EVALUATION
# ============================================================

total_correct = 0
total_pixels = 0

print("Running evaluation...")
print()


with torch.no_grad():

    for batch_idx, (images, targets) in enumerate(
        val_loader
    ):

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        predictions_np = (
            predictions.cpu().numpy()
        )

        targets_np = (
            targets.numpy()
        )

        # Pixel accuracy
        total_correct += np.sum(
            predictions_np == targets_np
        )

        total_pixels += targets_np.size

        # Confusion matrix
        confusion += confusion_matrix(
            targets_np.flatten(),
            predictions_np.flatten(),
            labels=list(range(NUM_CLASSES))
        )

        if (
            batch_idx + 1
        ) % 20 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1}/"
                f"{len(val_loader)} batches"
            )


# ============================================================
# PIXEL ACCURACY
# ============================================================

pixel_accuracy = (
    total_correct /
    total_pixels
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

ious = []
precisions = []
recalls = []
f1_scores = []


for class_id in range(NUM_CLASSES):

    true_positive = confusion[
        class_id,
        class_id
    ]

    false_positive = (
        confusion[:, class_id].sum()
        -
        true_positive
    )

    false_negative = (
        confusion[class_id, :].sum()
        -
        true_positive
    )

    # IoU
    denominator_iou = (
        true_positive
        +
        false_positive
        +
        false_negative
    )

    if denominator_iou > 0:

        iou = (
            true_positive /
            denominator_iou
        )

    else:

        iou = 0.0

    # Precision
    precision_denominator = (
        true_positive
        +
        false_positive
    )

    if precision_denominator > 0:

        precision = (
            true_positive /
            precision_denominator
        )

    else:

        precision = 0.0

    # Recall
    recall_denominator = (
        true_positive
        +
        false_negative
    )

    if recall_denominator > 0:

        recall = (
            true_positive /
            recall_denominator
        )

    else:

        recall = 0.0

    # F1
    if (
        precision + recall
    ) > 0:

        f1 = (
            2 *
            precision *
            recall
            /
            (precision + recall)
        )

    else:

        f1 = 0.0

    ious.append(iou)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)


# ============================================================
# MEAN IoU
# ============================================================

mean_iou = np.mean(
    ious
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print()
print(
    f"Pixel Accuracy: "
    f"{pixel_accuracy:.4f}"
)

print()

print("Per-Class Metrics:")

print()

print(
    f"{'Class':<20}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'IoU':>12}"
)

print("-" * 68)


for class_id in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[class_id]:<20}"
        f"{precisions[class_id]:>12.4f}"
        f"{recalls[class_id]:>12.4f}"
        f"{f1_scores[class_id]:>12.4f}"
        f"{ious[class_id]:>12.4f}"
    )


print()

print(
    f"Mean IoU: {mean_iou:.4f}"
)


# ============================================================
# PREDICTED CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

actual_total = confusion.sum(axis=1)
predicted_total = confusion.sum(axis=0)

print()

print(
    f"{'Class':<20}"
    f"{'Actual %':>15}"
    f"{'Predicted %':>15}"
)

print("-" * 50)

for class_id in range(NUM_CLASSES):

    actual_percentage = (
        actual_total[class_id]
        /
        total_pixels
        *
        100
    )

    predicted_percentage = (
        predicted_total[class_id]
        /
        total_pixels
        *
        100
    )

    print(
        f"{CLASS_NAMES[class_id]:<20}"
        f"{actual_percentage:>14.4f}%"
        f"{predicted_percentage:>14.4f}%"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print(
    f"{'Actual / Pred':<18}",
    end=""
)

for name in CLASS_NAMES:

    print(
        f"{name[:12]:>14}",
        end=""
    )

print()

print("-" * 88)

for row in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[row]:<18}",
        end=""
    )

    for col in range(NUM_CLASSES):

        print(
            f"{confusion[row, col]:>14,}",
            end=""
        )

    print()


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "focal_dice_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# Save confusion matrix

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

np.savetxt(
    confusion_path,
    confusion,
    delimiter=",",
    fmt="%d"
)


# Save metrics

metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.csv"
)

with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "class,precision,recall,f1,iou\n"
    )

    for class_id in range(NUM_CLASSES):

        f.write(
            f"{CLASS_NAMES[class_id]},"
            f"{precisions[class_id]:.6f},"
            f"{recalls[class_id]:.6f},"
            f"{f1_scores[class_id]:.6f},"
            f"{ious[class_id]:.6f}\n"
        )

    f.write(
        f"Mean IoU,"
        f",,,{mean_iou:.6f}\n"
    )

    f.write(
        f"Pixel Accuracy,"
        f"{pixel_accuracy:.6f},,,\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()
print(
    f"Metrics saved: {metrics_path}"
)

print(
    f"Confusion matrix saved: {confusion_path}"
)

print()