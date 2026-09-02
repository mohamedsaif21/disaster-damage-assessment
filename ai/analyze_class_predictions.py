import os
import sys
import inspect
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

# Your actual xBD validation CSV
VAL_CSV = r"D:\Projects\Datasets\xBD\splits\val.csv"

# Your trained model
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "checkpoints",
    "best_model.pth"
)

# Output directory
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "class_analysis"
)

BATCH_SIZE = 2
NUM_CLASSES = 5

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
print(f"Validation CSV: {VAL_CSV}")


# =========================================================
# CHECK PATHS
# =========================================================

if not os.path.exists(VAL_CSV):

    print("\nERROR: Validation CSV not found!")
    print(VAL_CSV)
    sys.exit(1)


if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model checkpoint not found!")
    print(MODEL_PATH)
    sys.exit(1)


# =========================================================
# DETECT XBD DATASET CONSTRUCTOR
# =========================================================

print("\n")
print("=" * 60)
print("CHECKING XBD DATASET")
print("=" * 60)

signature = inspect.signature(
    XBDDataset.__init__
)

parameters = list(
    signature.parameters.values()
)[1:]  # Remove self

print("\nXBDDataset constructor parameters:")

for parameter in parameters:

    print(
        f"  {parameter.name}: "
        f"default={parameter.default}"
    )


# =========================================================
# CREATE VALIDATION DATASET
# =========================================================

print("\nCreating validation dataset...")


def create_validation_dataset():

    """
    Automatically adapts to the constructor
    used by the existing dataset.py.

    We intentionally do not modify dataset.py.
    """

    parameter_names = [
        parameter.name.lower()
        for parameter in parameters
    ]

    # -----------------------------------------------------
    # Case 1:
    # Constructor accepts csv_file
    # -----------------------------------------------------

    if "csv_file" in parameter_names:

        print("Using parameter: csv_file")

        return XBDDataset(
            csv_file=VAL_CSV
        )

    # -----------------------------------------------------
    # Case 2:
    # Constructor accepts csv
    # -----------------------------------------------------

    if "csv" in parameter_names:

        print("Using parameter: csv")

        return XBDDataset(
            csv=VAL_CSV
        )

    # -----------------------------------------------------
    # Case 3:
    # Constructor accepts csv_path
    # -----------------------------------------------------

    if "csv_path" in parameter_names:

        print("Using parameter: csv_path")

        return XBDDataset(
            csv_path=VAL_CSV
        )

    # -----------------------------------------------------
    # Case 4:
    # Constructor accepts csv_file_path
    # -----------------------------------------------------

    if "csv_file_path" in parameter_names:

        print("Using parameter: csv_file_path")

        return XBDDataset(
            csv_file_path=VAL_CSV
        )

    # -----------------------------------------------------
    # Case 5:
    # Constructor accepts split_file
    # -----------------------------------------------------

    if "split_file" in parameter_names:

        print("Using parameter: split_file")

        return XBDDataset(
            split_file=VAL_CSV
        )

    # -----------------------------------------------------
    # Case 6:
    # Constructor accepts annotation_file
    # -----------------------------------------------------

    if "annotation_file" in parameter_names:

        print("Using parameter: annotation_file")

        return XBDDataset(
            annotation_file=VAL_CSV
        )

    # -----------------------------------------------------
    # Case 7:
    # Constructor accepts a single required argument
    # -----------------------------------------------------

    required_parameters = [
        parameter
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
    ]

    if len(required_parameters) == 1:

        print(
            "Using the single required "
            "dataset argument."
        )

        return XBDDataset(
            VAL_CSV
        )

    # -----------------------------------------------------
    # Could not automatically determine constructor
    # -----------------------------------------------------

    print("\nERROR: Could not determine how to")
    print("create XBDDataset from your dataset.py.")

    print("\nDetected constructor:")

    print(signature)

    print("\nPlease send me the output above.")

    sys.exit(1)


val_dataset = create_validation_dataset()


# =========================================================
# DATASET INFORMATION
# =========================================================

print(
    f"\nValidation samples: "
    f"{len(val_dataset)}"
)


# =========================================================
# DATALOADER
# =========================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"Validation batches: "
    f"{len(val_loader)}"
)


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


# Your training script saves a dictionary containing
# model_state_dict.
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        # Fallback if checkpoint itself is a state_dict
        model.load_state_dict(
            checkpoint
        )

else:

    model.load_state_dict(
        checkpoint
    )


model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# =========================================================
# STATISTICS
# =========================================================

actual_pixels = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

predicted_pixels = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# =========================================================
# EVALUATION
# =========================================================

print("\n")
print("=" * 60)
print("ANALYZING VALIDATION PREDICTIONS")
print("=" * 60)


with torch.no_grad():

    for batch_index, batch in enumerate(val_loader):

        # -------------------------------------------------
        # Dataset returns:
        #
        # images, targets
        # -------------------------------------------------

        images, targets = batch

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # -------------------------------------------------
        # Convert to NumPy
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

        # -------------------------------------------------
        # Count actual and predicted pixels
        # -------------------------------------------------

        for class_id in range(NUM_CLASSES):

            actual_pixels[class_id] += np.sum(
                targets_np == class_id
            )

            predicted_pixels[class_id] += np.sum(
                predictions_np == class_id
            )

        # -------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

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
# METRICS
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
    # F1
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
# PRINT CLASS DISTRIBUTION
# =========================================================

print("\n")
print("=" * 80)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 80)

print(
    f"{'Class':<18}"
    f"{'Actual %':>12}"
    f"{'Predicted %':>15}"
    f"{'Actual Pixels':>20}"
    f"{'Predicted Pixels':>20}"
)

print("-" * 80)


for class_id in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[class_id]:<18}"
        f"{actual_percentage[class_id]:>11.2f}%"
        f"{predicted_percentage[class_id]:>14.2f}%"
        f"{actual_pixels[class_id]:>20,}"
        f"{predicted_pixels[class_id]:>20,}"
    )


# =========================================================
# PRINT METRICS
# =========================================================

print("\n")
print("=" * 70)
print("PER-CLASS METRICS")
print("=" * 70)

print(
    f"{'Class':<18}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'IoU':>12}"
)

print("-" * 70)


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
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

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


for actual_class in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[actual_class]:<18}"
        + "".join(
            f"{confusion_matrix[actual_class, predicted_class]:>14,}"
            for predicted_class in range(NUM_CLASSES)
        )
    )


# =========================================================
# DAMAGE CLASS ANALYSIS
# =========================================================

print("\n")
print("=" * 80)
print("DAMAGE CLASS ANALYSIS")
print("=" * 80)


damage_classes = [
    2,  # Minor
    3,  # Major
    4   # Destroyed
]


for class_id in damage_classes:

    print("\n" + CLASS_NAMES[class_id])

    print(
        f"  Actual pixels:       "
        f"{actual_pixels[class_id]:,}"
    )

    print(
        f"  Predicted pixels:    "
        f"{predicted_pixels[class_id]:,}"
    )

    print(
        f"  Actual percentage:   "
        f"{actual_percentage[class_id]:.4f}%"
    )

    print(
        f"  Predicted percentage:"
        f" {predicted_percentage[class_id]:.4f}%"
    )

    print(
        f"  Precision:           "
        f"{precision[class_id]:.4f}"
    )

    print(
        f"  Recall:              "
        f"{recall[class_id]:.4f}"
    )

    print(
        f"  F1 Score:            "
        f"{f1_score[class_id]:.4f}"
    )

    print(
        f"  IoU:                 "
        f"{iou[class_id]:.4f}"
    )


# =========================================================
# WHERE MINOR DAMAGE GOES
# =========================================================

print("\n")
print("=" * 80)
print("MINOR DAMAGE ERROR DISTRIBUTION")
print("=" * 80)

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
        f"{CLASS_NAMES[predicted_class]:<15}"
        f"{count:>12,} "
        f"({percentage:.2f}%)"
    )


# =========================================================
# WHERE MAJOR DAMAGE GOES
# =========================================================

print("\n")
print("=" * 80)
print("MAJOR DAMAGE ERROR DISTRIBUTION")
print("=" * 80)

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
        f"{CLASS_NAMES[predicted_class]:<15}"
        f"{count:>12,} "
        f"({percentage:.2f}%)"
    )


# =========================================================
# SAVE CLASS RESULTS
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
            actual_percentage[class_id],

        "predicted_percentage":
            predicted_percentage[class_id],

        "precision":
            precision[class_id],

        "recall":
            recall[class_id],

        "f1_score":
            f1_score[class_id],

        "iou":
            iou[class_id]
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
# SAVE CONFUSION MATRIX
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
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    "\nClass analysis saved to:"
)

print(
    class_results_path
)

print(
    "\nConfusion matrix saved to:"
)

print(
    confusion_path
)


print("\n")
print("=" * 80)
print("KEY DAMAGE CLASS RESULTS")
print("=" * 80)


for class_id in damage_classes:

    print(
        f"{CLASS_NAMES[class_id]:<18}"
        f"IoU: {iou[class_id]:.4f} | "
        f"Recall: {recall[class_id]:.4f} | "
        f"F1: {f1_score[class_id]:.4f} | "
        f"Predicted: "
        f"{predicted_percentage[class_id]:.2f}%"
    )


print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)