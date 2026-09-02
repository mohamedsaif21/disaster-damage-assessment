import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from ai.dataset import XBDDataset
from ai.unet import UNet


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_CSV = r"D:\Projects\Datasets\xBD\splits\val.csv"
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "checkpoints",
    "best_model.pth"
)

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


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
print("=" * 50)
print("xBD CLASS PREDICTION ANALYSIS")
print("=" * 50)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Validation CSV: {VAL_CSV}")


# ---------------------------------------------------------
# Load validation dataset
# ---------------------------------------------------------
print("\nLoading validation dataset...")

val_dataset = XBDDataset(
    csv_path=VAL_CSV
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Validation samples: {len(val_dataset)}")
print(f"Validation batches: {len(val_loader)}")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
print("\nLoading model...")

model = UNet(
    in_channels=6,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Your checkpoint contains model_state_dict
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

actual_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
predicted_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)

# Confusion matrix
confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
print("\nAnalyzing predictions...")

with torch.no_grad():

    for batch_index, (images, targets) in enumerate(val_loader):

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Move to CPU
        targets_np = targets.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        # -------------------------------------------------
        # Class pixel counts
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

            mask = target_flat == actual_class

            if np.any(mask):

                predicted_for_actual = prediction_flat[mask]

                for predicted_class in range(NUM_CLASSES):

                    confusion_matrix[
                        actual_class,
                        predicted_class
                    ] += np.sum(
                        predicted_for_actual == predicted_class
                    )

        # Progress
        if (batch_index + 1) % 25 == 0:

            print(
                f"Processed Batch "
                f"{batch_index + 1}/{len(val_loader)}"
            )


# ---------------------------------------------------------
# Calculate percentages
# ---------------------------------------------------------
total_actual_pixels = actual_pixels.sum()
total_predicted_pixels = predicted_pixels.sum()

actual_percentage = (
    actual_pixels / total_actual_pixels
) * 100

predicted_percentage = (
    predicted_pixels / total_predicted_pixels
) * 100


# ---------------------------------------------------------
# Precision / Recall / F1 / IoU
# ---------------------------------------------------------
precision = np.zeros(NUM_CLASSES)
recall = np.zeros(NUM_CLASSES)
f1_score = np.zeros(NUM_CLASSES)
iou = np.zeros(NUM_CLASSES)

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

    # Precision
    if true_positive + false_positive > 0:
        precision[class_id] = (
            true_positive /
            (true_positive + false_positive)
        )

    # Recall
    if true_positive + false_negative > 0:
        recall[class_id] = (
            true_positive /
            (true_positive + false_negative)
        )

    # F1
    if precision[class_id] + recall[class_id] > 0:
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

    # IoU
    union = (
        true_positive
        + false_positive
        + false_negative
    )

    if union > 0:
        iou[class_id] = (
            true_positive / union
        )


# ---------------------------------------------------------
# Print class analysis
# ---------------------------------------------------------
print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 70)

print(
    f"{'Class':<15}"
    f"{'Actual %':>12}"
    f"{'Predicted %':>15}"
    f"{'Actual Pixels':>18}"
    f"{'Predicted Pixels':>20}"
)

print("-" * 70)

for class_id in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[class_id]:<15}"
        f"{actual_percentage[class_id]:>11.2f}%"
        f"{predicted_percentage[class_id]:>14.2f}%"
        f"{actual_pixels[class_id]:>18,}"
        f"{predicted_pixels[class_id]:>20,}"
    )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
print("\n")
print("=" * 70)
print("PER-CLASS METRICS")
print("=" * 70)

print(
    f"{'Class':<15}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'IoU':>12}"
)

print("-" * 70)

for class_id in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[class_id]:<15}"
        f"{precision[class_id]:>11.4f}"
        f"{recall[class_id]:>12.4f}"
        f"{f1_score[class_id]:>12.4f}"
        f"{iou[class_id]:>12.4f}"
    )


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("Rows = Actual")
print("Columns = Predicted\n")

print(
    f"{'':<15}"
    + "".join(
        f"{name[:10]:>12}"
        for name in CLASS_NAMES
    )
)

for actual_class in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[actual_class]:<15}"
        + "".join(
            f"{confusion_matrix[actual_class, predicted_class]:>12,}"
            for predicted_class in range(NUM_CLASSES)
        )
    )


# ---------------------------------------------------------
# Important damage-class analysis
# ---------------------------------------------------------
print("\n")
print("=" * 70)
print("DAMAGE CLASS ANALYSIS")
print("=" * 70)

damage_classes = [2, 3, 4]

for class_id in damage_classes:

    actual_count = actual_pixels[class_id]
    predicted_count = predicted_pixels[class_id]

    print(f"\n{CLASS_NAMES[class_id]}:")

    print(
        f"  Actual pixels:     {actual_count:,}"
    )

    print(
        f"  Predicted pixels:  {predicted_count:,}"
    )

    print(
        f"  Actual percentage:  {actual_percentage[class_id]:.4f}%"
    )

    print(
        f"  Predicted percentage: "
        f"{predicted_percentage[class_id]:.4f}%"
    )

    print(
        f"  Precision: {precision[class_id]:.4f}"
    )

    print(
        f"  Recall:    {recall[class_id]:.4f}"
    )

    print(
        f"  F1 Score:  {f1_score[class_id]:.4f}"
    )

    print(
        f"  IoU:       {iou[class_id]:.4f}"
    )


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------
results = []

for class_id in range(NUM_CLASSES):

    results.append({
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "actual_pixels": int(actual_pixels[class_id]),
        "predicted_pixels": int(predicted_pixels[class_id]),
        "actual_percentage": actual_percentage[class_id],
        "predicted_percentage": predicted_percentage[class_id],
        "precision": precision[class_id],
        "recall": recall[class_id],
        "f1_score": f1_score[class_id],
        "iou": iou[class_id]
    })

results_df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "class_prediction_analysis.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ---------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------
print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"Class analysis saved to:")
print(csv_path)

print(f"\nConfusion matrix saved to:")
print(confusion_path)

print("\nKey classes:")

for class_id in damage_classes:

    print(
        f"{CLASS_NAMES[class_id]} "
        f"→ IoU: {iou[class_id]:.4f}, "
        f"Recall: {recall[class_id]:.4f}, "
        f"Predicted: {predicted_percentage[class_id]:.2f}%"
    )

print("\n" + "=" * 70)