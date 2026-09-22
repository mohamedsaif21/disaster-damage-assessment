import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_change_aware import XBDChangeAwareDataset
from unet_change_aware import UNetChangeAware


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"D:\Projects\Datasets\xBD"

VAL_CSV = os.path.join(
    DATASET_ROOT,
    "splits",
    "val.csv"
)

CHECKPOINT_PATH = (
    r"ai\checkpoints\best_model_change_aware.pth"
)

OUTPUT_DIR = (
    r"outputs\hierarchical_analysis"
)

IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_CLASSES = 5

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed"
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HIERARCHICAL CLASS MAPPINGS
# ============================================================

# LEVEL 1
# 0 = Background
# 1 = Building
#
# Original:
# 0 -> Background
# 1,2,3,4 -> Building

BUILDING_MAP = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1
}


# LEVEL 2
# 0 = No Damage
# 1 = Damage
#
# Background is excluded from this analysis.
#
# Original:
# 1 -> No Damage
# 2,3,4 -> Damage

DAMAGE_MAP = {
    1: 0,
    2: 1,
    3: 1,
    4: 1
}


# LEVEL 3
# Damage severity:
#
# 2 = Minor
# 3 = Major
# 4 = Destroyed


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    targets,
    predictions,
    num_classes
):

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    for actual, predicted in zip(
        targets,
        predictions
    ):

        confusion[
            actual,
            predicted
        ] += 1

    metrics = []

    for class_id in range(
        num_classes
    ):

        tp = confusion[
            class_id,
            class_id
        ]

        fp = (
            confusion[:, class_id].sum()
            - tp
        )

        fn = (
            confusion[class_id, :].sum()
            - tp
        )

        precision_denominator = (
            tp + fp
        )

        recall_denominator = (
            tp + fn
        )

        union = (
            tp + fp + fn
        )

        precision = (
            tp / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            tp / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_denominator = (
            precision + recall
        )

        f1 = (
            2 * precision * recall
            / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

        iou = (
            tp / union
            if union > 0
            else 0.0
        )

        metrics.append(
            {
                "class_id": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou
            }
        )

    return (
        pd.DataFrame(metrics),
        confusion
    )


# ============================================================
# CONVERT ORIGINAL LABELS TO HIERARCHICAL LABELS
# ============================================================

def convert_building_labels(labels):

    converted = np.zeros_like(
        labels
    )

    for original, new in BUILDING_MAP.items():

        converted[
            labels == original
        ] = new

    return converted


def convert_damage_labels(labels):

    # Only labels 1-4 are relevant.
    mask = labels != 0

    filtered = labels[
        mask
    ]

    converted = np.zeros_like(
        filtered
    )

    for original, new in DAMAGE_MAP.items():

        converted[
            filtered == original
        ] = new

    return (
        filtered,
        converted
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 75)
    print("HIERARCHICAL DAMAGE ANALYSIS")
    print("=" * 75)

    print()
    print("Device    :", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Validation:", VAL_CSV)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = XBDChangeAwareDataset(
        DATASET_ROOT,
        VAL_CSV,
        image_size=IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print()
    print(
        "Validation samples:",
        len(dataset)
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = UNetChangeAware(
        in_channels=9,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint epoch:",
        checkpoint["epoch"]
    )

    # --------------------------------------------------------
    # COLLECT ALL PIXELS
    # --------------------------------------------------------

    all_targets = []
    all_predictions = []

    with torch.no_grad():

        for batch_index, (
            images,
            targets
        ) in enumerate(loader):

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            targets_np = (
                targets
                .numpy()
                .reshape(-1)
            )

            predictions_np = (
                predictions
                .cpu()
                .numpy()
                .reshape(-1)
            )

            all_targets.append(
                targets_np
            )

            all_predictions.append(
                predictions_np
            )

            if (
                batch_index + 1
            ) % 25 == 0:

                print(
                    f"Processed "
                    f"{batch_index + 1}/"
                    f"{len(loader)} batches"
                )

    targets = np.concatenate(
        all_targets
    )

    predictions = np.concatenate(
        all_predictions
    )

    # ========================================================
    # LEVEL 1 — BUILDING DETECTION
    # ========================================================

    print()
    print("=" * 75)
    print("LEVEL 1 — BACKGROUND vs BUILDING")
    print("=" * 75)

    building_targets = (
        convert_building_labels(
            targets
        )
    )

    building_predictions = (
        convert_building_labels(
            predictions
        )
    )

    building_metrics, building_confusion = (
        calculate_metrics(
            building_targets,
            building_predictions,
            2
        )
    )

    building_metrics["class_name"] = [
        "Background",
        "Building"
    ]

    print()

    for _, row in building_metrics.iterrows():

        print(
            f"{row['class_name']:<15}"
            f" Precision {row['precision']:.4f}"
            f" Recall {row['recall']:.4f}"
            f" F1 {row['f1']:.4f}"
            f" IoU {row['iou']:.4f}"
        )

    print()
    print("Confusion Matrix:")
    print(building_confusion)

    # ========================================================
    # LEVEL 2 — NO DAMAGE vs DAMAGE
    # ========================================================

    print()
    print("=" * 75)
    print("LEVEL 2 — NO DAMAGE vs DAMAGE")
    print("=" * 75)

    (
        damage_original_targets,
        damage_targets
    ) = convert_damage_labels(
        targets
    )

    (
        damage_original_predictions,
        damage_predictions
    ) = convert_damage_labels(
        predictions
    )

    # Important:
    # Both target and prediction must refer to
    # the same pixels.
    #
    # We therefore use the original target's
    # non-background mask for both.

    building_pixel_mask = (
        targets != 0
    )

    damage_targets = np.where(
        targets[building_pixel_mask] == 1,
        0,
        1
    )

    damage_predictions = np.where(
        predictions[building_pixel_mask] == 1,
        0,
        1
    )

    damage_metrics, damage_confusion = (
        calculate_metrics(
            damage_targets,
            damage_predictions,
            2
        )
    )

    damage_metrics["class_name"] = [
        "No Damage",
        "Damage"
    ]

    print()

    for _, row in damage_metrics.iterrows():

        print(
            f"{row['class_name']:<15}"
            f" Precision {row['precision']:.4f}"
            f" Recall {row['recall']:.4f}"
            f" F1 {row['f1']:.4f}"
            f" IoU {row['iou']:.4f}"
        )

    print()
    print("Confusion Matrix:")
    print(damage_confusion)

    # ========================================================
    # LEVEL 3 — MINOR vs MAJOR vs DESTROYED
    # ========================================================

    print()
    print("=" * 75)
    print("LEVEL 3 — DAMAGE SEVERITY")
    print("=" * 75)

    severity_mask = (
        targets >= 2
    )

    severity_targets_original = (
        targets[severity_mask]
    )

    severity_predictions_original = (
        predictions[severity_mask]
    )

    # Convert:
    # Minor     = 0
    # Major     = 1
    # Destroyed = 2

    severity_targets = (
        severity_targets_original - 2
    )

    severity_predictions = (
        severity_predictions_original - 2
    )

    # Predictions that are Background or No Damage
    # cannot be treated as valid severity classes.
    #
    # Therefore clamp them to a special handling class.
    #
    # We create a 4-class analysis:
    #
    # 0 Minor
    # 1 Major
    # 2 Destroyed
    # 3 Predicted No-Damage/Background

    severity_predictions = np.where(
        severity_predictions_original < 0,
        3,
        severity_predictions
    )

    severity_metrics, severity_confusion = (
        calculate_metrics(
            severity_targets,
            severity_predictions,
            4
        )
    )

    severity_metrics["class_name"] = [
        "Minor",
        "Major",
        "Destroyed",
        "Missed Damage"
    ]

    print()

    for _, row in severity_metrics.iterrows():

        print(
            f"{row['class_name']:<18}"
            f" Precision {row['precision']:.4f}"
            f" Recall {row['recall']:.4f}"
            f" F1 {row['f1']:.4f}"
            f" IoU {row['iou']:.4f}"
        )

    print()
    print("Confusion Matrix:")
    print(severity_confusion)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    building_metrics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "building_metrics.csv"
        ),
        index=False
    )

    damage_metrics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "damage_detection_metrics.csv"
        ),
        index=False
    )

    severity_metrics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "severity_metrics.csv"
        ),
        index=False
    )

    pd.DataFrame(
        building_confusion,
        index=[
            "Background",
            "Building"
        ],
        columns=[
            "Background",
            "Building"
        ]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "building_confusion_matrix.csv"
        )
    )

    pd.DataFrame(
        damage_confusion,
        index=[
            "No Damage",
            "Damage"
        ],
        columns=[
            "No Damage",
            "Damage"
        ]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "damage_confusion_matrix.csv"
        )
    )

    pd.DataFrame(
        severity_confusion,
        index=[
            "Minor",
            "Major",
            "Destroyed",
            "Missed Damage"
        ],
        columns=[
            "Minor",
            "Major",
            "Destroyed",
            "Missed Damage"
        ]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "severity_confusion_matrix.csv"
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 75)
    print("HIERARCHICAL ANALYSIS COMPLETE")
    print("=" * 75)

    print()
    print("Saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()