import sys
from pathlib import Path

sys.path.insert(0, "ai")

import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import DataLoader

from dataset import XBDDataset
from unet import UNet


# ======================================
# CONFIGURATION
# ======================================

DATASET_ROOT = Path(r"D:\Projects\Datasets\xBD")

VAL_CSV = DATASET_ROOT / "splits" / "val.csv"

MODEL_PATH = Path(
    r"ai\checkpoints\best_model.pth"
)

OUTPUT_DIR = Path(
    "outputs/validation_results"
)

IMAGE_SIZE = 256

BATCH_SIZE = 2

# Number of validation samples to test
NUM_SAMPLES = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================
# CLASS INFORMATION
# ======================================

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}


NUM_CLASSES = 5


# ======================================
# SETUP
# ======================================

print("======================================")
print("xBD MULTI-SAMPLE VALIDATION")
print("======================================")

print("Device:", DEVICE)
print("Model:", MODEL_PATH)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================
# LOAD DATASET
# ======================================

print()
print("Loading validation dataset...")

dataset = XBDDataset(
    DATASET_ROOT,
    VAL_CSV,
    image_size=IMAGE_SIZE,
)

print(
    "Total validation samples:",
    len(dataset)
)


# ======================================
# LIMIT NUMBER OF SAMPLES
# ======================================

num_samples = min(
    NUM_SAMPLES,
    len(dataset)
)

print(
    "Samples to evaluate:",
    num_samples
)


# ======================================
# LOAD MODEL
# ======================================

print()
print("Loading model...")

model = UNet(
    in_channels=6,
    num_classes=5,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ======================================
# IoU FUNCTION
# ======================================

def calculate_iou(
    prediction,
    target,
    class_id
):

    prediction_mask = (
        prediction == class_id
    )

    target_mask = (
        target == class_id
    )

    intersection = np.logical_and(
        prediction_mask,
        target_mask
    ).sum()

    union = np.logical_or(
        prediction_mask,
        target_mask
    ).sum()

    if union == 0:

        return None

    return intersection / union


# ======================================
# RESULTS STORAGE
# ======================================

results = []


# ======================================
# EVALUATION
# ======================================

print()
print("======================================")
print("EVALUATING SAMPLES")
print("======================================")


with torch.no_grad():

    for index in range(num_samples):

        print()
        print(
            f"Sample {index + 1}/{num_samples}"
        )

        # ----------------------------------
        # Load sample
        # ----------------------------------

        image, target = dataset[index]

        # Add batch dimension

        image_batch = image.unsqueeze(0)

        image_batch = image_batch.to(
            DEVICE
        )

        # ----------------------------------
        # Prediction
        # ----------------------------------

        output = model(
            image_batch
        )

        prediction = torch.argmax(
            output,
            dim=1
        )

        prediction = prediction.squeeze(
            0
        )

        prediction = prediction.cpu().numpy()

        target = target.numpy()

        # ----------------------------------
        # Sample ID
        # ----------------------------------

        sample_id = dataset.samples.iloc[
            index
        ]["sample_id"]

        # ----------------------------------
        # Damage percentages
        # ----------------------------------

        total_pixels = target.size

        actual_damage_pixels = np.sum(
            target >= 2
        )

        predicted_damage_pixels = np.sum(
            prediction >= 2
        )

        actual_damage_percentage = (
            actual_damage_pixels /
            total_pixels
        ) * 100

        predicted_damage_percentage = (
            predicted_damage_pixels /
            total_pixels
        ) * 100

        # ----------------------------------
        # Per-class IoU
        # ----------------------------------

        class_ious = {}

        for class_id in range(
            NUM_CLASSES
        ):

            iou = calculate_iou(
                prediction,
                target,
                class_id
            )

            class_ious[class_id] = iou

        # ----------------------------------
        # Mean IoU
        # ----------------------------------

        valid_ious = [
            value
            for value in class_ious.values()
            if value is not None
        ]

        if len(valid_ious) > 0:

            mean_iou = np.mean(
                valid_ious
            )

        else:

            mean_iou = 0.0

        # ----------------------------------
        # Store result
        # ----------------------------------

        results.append({

            "sample_id": sample_id,

            "actual_damage_percentage":
                actual_damage_percentage,

            "predicted_damage_percentage":
                predicted_damage_percentage,

            "damage_difference":
                abs(
                    actual_damage_percentage -
                    predicted_damage_percentage
                ),

            "background_iou":
                class_ious[0],

            "no_damage_iou":
                class_ious[1],

            "minor_damage_iou":
                class_ious[2],

            "major_damage_iou":
                class_ious[3],

            "destroyed_iou":
                class_ious[4],

            "mean_iou":
                mean_iou,
        })

        # ----------------------------------
        # Print sample result
        # ----------------------------------

        print(
            "Sample ID:",
            sample_id
        )

        print(
            f"Actual Damage: "
            f"{actual_damage_percentage:.2f}%"
        )

        print(
            f"Predicted Damage: "
            f"{predicted_damage_percentage:.2f}%"
        )

        print(
            f"Mean IoU: "
            f"{mean_iou:.4f}"
        )

        print(
            f"Minor IoU: "
            f"{class_ious[2] if class_ious[2] is not None else 0:.4f}"
        )

        print(
            f"Major IoU: "
            f"{class_ious[3] if class_ious[3] is not None else 0:.4f}"
        )

        print(
            f"Destroyed IoU: "
            f"{class_ious[4] if class_ious[4] is not None else 0:.4f}"
        )


# ======================================
# CREATE DATAFRAME
# ======================================

results_df = pd.DataFrame(
    results
)


# ======================================
# OVERALL RESULTS
# ======================================

print()
print("======================================")
print("OVERALL RESULTS")
print("======================================")


print()

print(
    f"Samples evaluated: "
    f"{len(results_df)}"
)


print()

print(
    f"Average Actual Damage: "
    f"{results_df['actual_damage_percentage'].mean():.2f}%"
)


print(
    f"Average Predicted Damage: "
    f"{results_df['predicted_damage_percentage'].mean():.2f}%"
)


print(
    f"Average Damage Difference: "
    f"{results_df['damage_difference'].mean():.2f}%"
)


print()

print(
    f"Background IoU: "
    f"{results_df['background_iou'].mean():.4f}"
)


print(
    f"No Damage IoU: "
    f"{results_df['no_damage_iou'].mean():.4f}"
)


print(
    f"Minor Damage IoU: "
    f"{results_df['minor_damage_iou'].mean():.4f}"
)


print(
    f"Major Damage IoU: "
    f"{results_df['major_damage_iou'].mean():.4f}"
)


print(
    f"Destroyed IoU: "
    f"{results_df['destroyed_iou'].mean():.4f}"
)


print()

print(
    f"Mean IoU: "
    f"{results_df['mean_iou'].mean():.4f}"
)


# ======================================
# SAVE CSV REPORT
# ======================================

report_path = (
    OUTPUT_DIR /
    "validation_report.csv"
)

results_df.to_csv(
    report_path,
    index=False
)


print()
print("Report saved:")
print(report_path)


# ======================================
# BEST / WORST SAMPLES
# ======================================

best_sample = results_df.loc[
    results_df["mean_iou"].idxmax()
]

worst_sample = results_df.loc[
    results_df["mean_iou"].idxmin()
]


print()
print("======================================")
print("BEST SAMPLE")
print("======================================")


print(
    "Sample:",
    best_sample["sample_id"]
)

print(
    f"Mean IoU: "
    f"{best_sample['mean_iou']:.4f}"
)


print()
print("======================================")
print("WORST SAMPLE")
print("======================================")


print(
    "Sample:",
    worst_sample["sample_id"]
)

print(
    f"Mean IoU: "
    f"{worst_sample['mean_iou']:.4f}"
)


# ======================================
# COMPLETE
# ======================================

print()
print("======================================")
print("MULTI-SAMPLE VALIDATION COMPLETE")
print("======================================")