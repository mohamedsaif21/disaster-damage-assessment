import sys

sys.path.insert(0, "ai")

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


# ======================================
# Configuration
# ======================================

DATASET_ROOT = Path(r"D:\Projects\Datasets\xBD")
TRAIN_CSV = DATASET_ROOT / "splits" / "train.csv"

TARGET_DIR = DATASET_ROOT / "train" / "targets"

NUM_CLASSES = 5

CLASS_NAMES = [
    "Background",
    "No Damage",
    "Minor Damage",
    "Major Damage",
    "Destroyed",
]


# ======================================
# Setup
# ======================================

print("======================================")
print("xBD CLASS DISTRIBUTION ANALYSIS")
print("======================================")

samples = pd.read_csv(TRAIN_CSV)

print("Training samples:", len(samples))


# ======================================
# Count pixels
# ======================================

class_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.int64,
)


print()
print("Scanning target masks...")


for index, row in samples.iterrows():

    sample_id = row["sample_id"]

    target_path = (
        TARGET_DIR
        / f"{sample_id}_post_disaster_target.png"
    )

    target = np.array(
        Image.open(target_path),
        dtype=np.int64,
    )

    for class_id in range(NUM_CLASSES):

        class_counts[class_id] += np.sum(
            target == class_id
        )

    if (index + 1) % 100 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(samples)}"
        )


# ======================================
# Statistics
# ======================================

total_pixels = np.sum(class_counts)


print()
print("======================================")
print("CLASS DISTRIBUTION")
print("======================================")

for class_id in range(NUM_CLASSES):

    count = class_counts[class_id]

    percentage = (
        count / total_pixels
    ) * 100

    print(
        f"Class {class_id} "
        f"({CLASS_NAMES[class_id]}): "
        f"{count:,} pixels "
        f"({percentage:.4f}%)"
    )


# ======================================
# Class weights
# ======================================

print()
print("======================================")
print("SUGGESTED CLASS WEIGHTS")
print("======================================")

frequencies = class_counts / total_pixels

weights = 1.0 / (
    frequencies + 1e-8
)

# Normalize weights
weights = weights / weights.mean()

for class_id in range(NUM_CLASSES):

    print(
        f"Class {class_id} "
        f"({CLASS_NAMES[class_id]}): "
        f"{weights[class_id]:.4f}"
    )


# ======================================
# Complete
# ======================================

print()
print("======================================")
print("ANALYSIS COMPLETE")
print("======================================")