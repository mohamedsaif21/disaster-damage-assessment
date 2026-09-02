import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# =========================================================
# IMPORT PROJECT FILES
# =========================================================

from ai.dataset import XBDDataset
from ai.unet import UNet


# =========================================================
# CONFIGURATION
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------
# Your actual xBD dataset
# ---------------------------------------------------------

DATASET_ROOT = r"D:\Projects\Datasets\xBD"

# ---------------------------------------------------------
# Validation split
# ---------------------------------------------------------

VAL_CSV = r"D:\Projects\Datasets\xBD\splits\val.csv"

# ---------------------------------------------------------
# Trained model
# ---------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "checkpoints",
    "best_model.pth"
)

# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "class_analysis"
)

# ---------------------------------------------------------
# Model / Data settings
# ---------------------------------------------------------

BATCH_SIZE = 2
NUM_CLASSES = 5
IMAGE_SIZE = 256

CLASS_NAMES = [
    "Background",
    "No Damage",
    "Minor Damage",
    "Major Damage",
    "Destroyed"
]


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# HEADER
# =========================================================

print("=" * 60)
print("xBD CLASS PREDICTION ANALYSIS")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Dataset Root: {DATASET_ROOT}")
print(f"Validation CSV: {VAL_CSV}")


# =========================================================
# CHECK REQUIRED PATHS
# =========================================================

print("\n")
print("=" * 60)
print("CHECKING PATHS")
print("=" * 60)


if not os.path.exists(DATASET_ROOT):

    print("\nERROR: Dataset root not found!")
    print(DATASET_ROOT)
    sys.exit(1)

print("Dataset root: OK")


if not os.path.exists(VAL_CSV):

    print("\nERROR: Validation CSV not found!")
    print(VAL_CSV)
    sys.exit(1)

print("Validation CSV: OK")


if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model checkpoint not found!")
    print(MODEL_PATH)
    sys.exit(1)

print("Model checkpoint: OK")


# =========================================================
# LOAD VALIDATION DATASET
# =========================================================

print("\n")
print("=" * 60)
print("LOADING XBD VALIDATION DATASET")
print("=" * 60)


print("\nDataset root:")
print(DATASET_ROOT)

print("\nSplit file:")
print(VAL_CSV)

print("\nImage size:")
print(IMAGE_SIZE)


# ---------------------------------------------------------
# IMPORTANT:
#
# Your actual XBDDataset constructor is:
#
# XBDDataset(
#     dataset_root,
#     split_file,
#     image_size=256
# )
#
# ---------------------------------------------------------

val_dataset = XBDDataset(
    dataset_root=DATASET_ROOT,
    split_file=VAL_CSV,
    image_size=IMAGE_SIZE
)


print("\nValidation samples:")
print(len(val_dataset))


# =========================================================
# CREATE DATALOADER
# =========================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("\nValidation batches:")
print(len(val_loader))


# =========================================================
# LOAD MODEL
# =========================================================

print("\n")
print("=" * 60)
print("LOADING MODEL")
print("=" * 60)


model = UNet(
    in_channels=6,
    num_classes=NUM_CLASSES
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ---------------------------------------------------------
# Your training script saves:
#
# checkpoint["model_state_dict"]
#
# ---------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print("Loaded model_state_dict successfully.")

    else:

        # Fallback if checkpoint itself is state_dict
        model.load_state_dict(
            checkpoint
        )

        print("Loaded checkpoint directly.")

else:

    model.load_state_dict(
        checkpoint
    )

    print("Loaded checkpoint directly.")


model.to(DEVICE)
model.eval()


print("Model loaded successfully.")


# =========================================================
# INITIALIZE STATISTICS
# =========================================================

actual_pixels = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)


predicted_pixels = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)


# ---------------------------------------------------------
# Confusion matrix
#
# Rows    = Actual
# Columns = Predicted
# ---------------------------------------------------------

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# =========================================================
# ANALYZE VALIDATION DATA
# =========================================================

print("\n")
print("=" * 60)
print("ANALYZING VALIDATION PREDICTIONS")
print("=" * 60)


with torch.no_grad():

    for batch_index, (images, targets) in enumerate(
        val_loader
    ):

        # -------------------------------------------------
        # Move data to device
        # -------------------------------------------------

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)


        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        outputs = model(images)


        # -------------------------------------------------
        # Convert logits to predicted class
        # -------------------------------------------------

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # -------------------------------------------------
        # Convert tensors to NumPy
        # -------------------------------------------------

        targets_np = (
            targets
            .cpu()
            .numpy()
        )


        predictions_np = (
            predictions
            .cpu()
            .numpy()
        )


        # =================================================
        # COUNT ACTUAL / PREDICTED PIXELS
        # =================================================

        for class_id in range(NUM_CLASSES):

            actual_pixels[class_id] += np.sum(
                targets_np == class_id
            )


            predicted_pixels[class_id] += np.sum(
                predictions_np == class_id
            )


        # =================================================
        # BUILD CONFUSION MATRIX
        # =================================================

        target_flat = targets_np.reshape(-1)

        prediction_flat = predictions_np.reshape(-1)


        for actual_class in range(NUM_CLASSES):

            actual_mask = (
                target_flat == actual_class
            )


            if np.any(actual_mask):

                predicted_for_actual = (
                    prediction_flat[actual_mask]
                )


                for predicted_class in range(
                    NUM_CLASSES
                ):

                    confusion_matrix[
                        actual_class,
                        predicted_class
                    ] += np.sum(
                        predicted_for_actual
                        == predicted_class
                    )


        # =================================================
        # PROGRESS
        # =================================================

        if (batch_index + 1) % 25 == 0:

            print(
                f"Processed Batch "
                f"{batch_index + 1}/"
                f"{len(val_loader)}"
            )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

total_actual_pixels = actual_pixels.sum()

total_predicted_pixels = predicted_pixels.sum()


actual_percentage = (
    actual_pixels /
    total_actual_pixels
) * 100


predicted_percentage = (
    predicted_pixels /
    total_predicted_pixels
) * 100


# =========================================================
# CALCULATE METRICS
# =========================================================

precision = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)


recall = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)


f1_score = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)


iou = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)


for class_id in range(NUM_CLASSES):

    true_positive = confusion_matrix[
        class_id,
        class_id
    ]


    false_positive = (
        confusion_matrix[:, class_id].sum()
        - true_positive
    )


    false_negative = (
        confusion_matrix[class_id, :].sum()
        - true_positive
    )


    # -----------------------------------------------------
    # Precision
    # -----------------------------------------------------

    if (
        true_positive
        + false_positive
    ) > 0:

        precision[class_id] = (
            true_positive /
            (
                true_positive
                + false_positive
            )
        )


    # -----------------------------------------------------
    # Recall
    # -----------------------------------------------------

    if (
        true_positive
        + false_negative
    ) > 0:

        recall[class_id] = (
            true_positive /
            (
                true_positive
                + false_negative
            )
        )


    # -----------------------------------------------------
    # F1 Score
    # -----------------------------------------------------

    if (
        precision[class_id]
        + recall[class_id]
    ) > 0:

        f1_score[class_id] = (
            2
            * precision[class_id]
            * recall[class_id]
            /
            (
                precision[class_id]
                + recall[class_id]
            )
        )


    # -----------------------------------------------------
    # IoU
    # -----------------------------------------------------

    union = (
        true_positive
        + false_positive
        + false_negative
    )


    if union > 0:

        iou[class_id] = (
            true_positive /
            union
        )


# =========================================================
# CLASS DISTRIBUTION REPORT
# =========================================================

print("\n")
print("=" * 90)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 90)


print(
    f"{'Class':<18}"
    f"{'Actual %':>12}"
    f"{'Predicted %':>15}"
    f"{'Actual Pixels':>20}"
    f"{'Predicted Pixels':>20}"
)


print("-" * 90)


for class_id in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[class_id]:<18}"
        f"{actual_percentage[class_id]:>11.2f}%"
        f"{predicted_percentage[class_id]:>14.2f}%"
        f"{actual_pixels[class_id]:>20,}"
        f"{predicted_pixels[class_id]:>20,}"
    )


# =========================================================
# PER-CLASS METRICS
# =========================================================

print("\n")
print("=" * 75)
print("PER-CLASS METRICS")
print("=" * 75)


print(
    f"{'Class':<18}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'IoU':>12}"
)


print("-" * 75)


for class_id in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[class_id]:<18}"
        f"{precision[class_id]:>12.4f}"
        f"{recall[class_id]:>12.4f}"
        f"{f1_score[class_id]:>12.4f}"
        f"{iou[class_id]:>12.4f}"
    )


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n")
print("=" * 90)
print("CONFUSION MATRIX")
print("=" * 90)


print("Rows = Actual")
print("Columns = Predicted")
print()


print(
    f"{'Actual / Pred':<18}"
    + "".join(
        f"{name[:10]:>14}"
        for name in CLASS_NAMES
    )
)


print("-" * 90)


for actual_class in range(NUM_CLASSES):

    row_values = ""

    for predicted_class in range(NUM_CLASSES):

        value = confusion_matrix[
            actual_class,
            predicted_class
        ]

        row_values += f"{value:>14,}"

    print(
        f"{CLASS_NAMES[actual_class]:<18}"
        + row_values
    )


# =========================================================
# DAMAGE CLASS ANALYSIS
# =========================================================

print("\n")
print("=" * 90)
print("DAMAGE CLASS ANALYSIS")
print("=" * 90)


damage_classes = [
    2,  # Minor Damage
    3,  # Major Damage
    4   # Destroyed
]


for class_id in damage_classes:

    print("\n" + CLASS_NAMES[class_id])

    print(
        f"  Actual pixels:        "
        f"{actual_pixels[class_id]:,}"
    )


    print(
        f"  Predicted pixels:     "
        f"{predicted_pixels[class_id]:,}"
    )


    print(
        f"  Actual percentage:    "
        f"{actual_percentage[class_id]:.4f}%"
    )


    print(
        f"  Predicted percentage: "
        f"{predicted_percentage[class_id]:.4f}%"
    )


    print(
        f"  Precision:            "
        f"{precision[class_id]:.4f}"
    )


    print(
        f"  Recall:               "
        f"{recall[class_id]:.4f}"
    )


    print(
        f"  F1 Score:             "
        f"{f1_score[class_id]:.4f}"
    )


    print(
        f"  IoU:                  "
        f"{iou[class_id]:.4f}"
    )


# =========================================================
# MINOR DAMAGE ERROR DISTRIBUTION
# =========================================================

print("\n")
print("=" * 90)
print("MINOR DAMAGE ERROR DISTRIBUTION")
print("=" * 90)


minor_row = confusion_matrix[2]

minor_total = minor_row.sum()


for predicted_class in range(NUM_CLASSES):

    count = minor_row[predicted_class]


    percentage = (
        count / minor_total * 100
        if minor_total > 0
        else 0
    )


    print(
        f"Actual Minor Damage → "
        f"{CLASS_NAMES[predicted_class]:<18}"
        f"{count:>12,} "
        f"({percentage:.2f}%)"
    )


# =========================================================
# MAJOR DAMAGE ERROR DISTRIBUTION
# =========================================================

print("\n")
print("=" * 90)
print("MAJOR DAMAGE ERROR DISTRIBUTION")
print("=" * 90)


major_row = confusion_matrix[3]

major_total = major_row.sum()


for predicted_class in range(NUM_CLASSES):

    count = major_row[predicted_class]


    percentage = (
        count / major_total * 100
        if major_total > 0
        else 0
    )


    print(
        f"Actual Major Damage → "
        f"{CLASS_NAMES[predicted_class]:<18}"
        f"{count:>12,} "
        f"({percentage:.2f}%)"
    )


# =========================================================
# SAVE CLASS ANALYSIS CSV
# =========================================================

results = []


for class_id in range(NUM_CLASSES):

    results.append({

        "class_id":
            class_id,

        "class_name":
            CLASS_NAMES[class_id],

        "actual_pixels":
            int(actual_pixels[class_id]),

        "predicted_pixels":
            int(predicted_pixels[class_id]),

        "actual_percentage":
            float(actual_percentage[class_id]),

        "predicted_percentage":
            float(predicted_percentage[class_id]),

        "precision":
            float(precision[class_id]),

        "recall":
            float(recall[class_id]),

        "f1_score":
            float(f1_score[class_id]),

        "iou":
            float(iou[class_id])
    })


results_df = pd.DataFrame(
    results
)


class_results_path = os.path.join(
    OUTPUT_DIR,
    "class_prediction_analysis.csv"
)


results_df.to_csv(
    class_results_path,
    index=False
)


# =========================================================
# SAVE CONFUSION MATRIX CSV
# =========================================================

confusion_df = pd.DataFrame(
    confusion_matrix,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)


confusion_df.to_csv(
    confusion_path
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)


print("\nClass analysis saved to:")
print(class_results_path)


print("\nConfusion matrix saved to:")
print(confusion_path)


# =========================================================
# KEY DAMAGE RESULTS
# =========================================================

print("\n")
print("=" * 90)
print("KEY DAMAGE CLASS RESULTS")
print("=" * 90)


for class_id in damage_classes:

    print(
        f"{CLASS_NAMES[class_id]:<18}"
        f"IoU: {iou[class_id]:.4f} | "
        f"Recall: {recall[class_id]:.4f} | "
        f"F1: {f1_score[class_id]:.4f} | "
        f"Predicted: "
        f"{predicted_percentage[class_id]:.2f}%"
    )


# =========================================================
# DONE
# =========================================================

print("\n")
print("=" * 90)
print("DONE")
print("=" * 90)