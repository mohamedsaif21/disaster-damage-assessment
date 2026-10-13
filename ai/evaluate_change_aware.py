import argparse
import json
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

# Split manifests. "val" is the split used for checkpoint
# selection and remains the default so the original validation
# workflow is unchanged. "test" is the held-out split.
SPLITS = {
    "val": os.path.join(
        DATASET_ROOT,
        "splits",
        "val.csv"
    ),
    "test": os.path.join(
        DATASET_ROOT,
        "splits",
        "test.csv"
    )
}

DEFAULT_SPLIT = "val"

# Each split writes to its own directory so held-out test results
# can never overwrite or be confused with validation results.
OUTPUT_DIRS = {
    "val": r"outputs\change_aware_evaluation",
    "test": r"outputs\change_aware_test_evaluation"
}

CHECKPOINT_PATH = (
    r"ai\checkpoints\best_model_change_aware.pth"
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
# CLI
# ============================================================

def parse_args(argv=None):
    """Parse command-line arguments.

    Only the evaluated split is configurable. The default ("val")
    preserves the original validation workflow; "test" selects the
    held-out split without duplicating this evaluator.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Change-Aware U-Net checkpoint on a "
            "dataset split."
        )
    )

    parser.add_argument(
        "--split",
        choices=sorted(SPLITS.keys()),
        default=DEFAULT_SPLIT,
        help=(
            "Split manifest to evaluate. Defaults to 'val' "
            "(backward compatible); use 'test' for the held-out "
            "test evaluation."
        )
    )

    return parser.parse_args(argv)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    split = args.split
    split_csv = SPLITS[split]
    output_dir = OUTPUT_DIRS[split]

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("=" * 75)
    print("EXPERIMENT 4 — CHANGE-AWARE U-NET EVALUATION")
    print("=" * 75)

    print()
    print("Device    :", DEVICE)
    print("Split     :", split)
    print("Split CSV :", split_csv)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Input     : 9 channels")
    print("Classes   : 5")
    print("Image size:", IMAGE_SIZE)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = XBDChangeAwareDataset(
        DATASET_ROOT,
        split_csv,
        image_size=IMAGE_SIZE
    )

    expected_samples = len(
        pd.read_csv(split_csv)
    )

    print()
    print(
        "Split CSV rows   :",
        expected_samples
    )

    print(
        "Evaluated samples:",
        len(dataset)
    )

    if len(dataset) != expected_samples:

        raise RuntimeError(
            "Dataset length "
            f"({len(dataset)}) does not match "
            f"the split manifest ({split_csv}, "
            f"{expected_samples} rows)"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        "Evaluated batches:",
        len(loader)
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

    print()
    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"]
    )

    print(
        "Checkpoint validation loss:",
        f"{checkpoint['val_loss']:.4f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    confusion_matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    total_pixels = 0
    correct_pixels = 0

    # --------------------------------------------------------
    # CLASS PIXEL COUNTS
    # --------------------------------------------------------

    actual_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    predicted_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print()
    print("Running evaluation...")

    with torch.no_grad():

        for batch_index, (
            images,
            targets
        ) in enumerate(loader):

            images = images.to(
                DEVICE
            )

            targets = targets.to(
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
                .cpu()
                .numpy()
                .reshape(-1)
            )

            predictions_np = (
                predictions
                .cpu()
                .numpy()
                .reshape(-1)
            )

            # Accuracy
            total_pixels += len(
                targets_np
            )

            correct_pixels += np.sum(
                predictions_np == targets_np
            )

            # Class counts
            for class_id in range(
                NUM_CLASSES
            ):

                actual_counts[class_id] += np.sum(
                    targets_np == class_id
                )

                predicted_counts[class_id] += np.sum(
                    predictions_np == class_id
                )

            # Confusion matrix
            for actual, predicted in zip(
                targets_np,
                predictions_np
            ):

                confusion_matrix[
                    actual,
                    predicted
                ] += 1

            if (
                batch_index + 1
            ) % 25 == 0:

                print(
                    f"Processed "
                    f"{batch_index + 1}/"
                    f"{len(loader)} batches"
                )

    # ========================================================
    # PIXEL ACCURACY
    # ========================================================

    pixel_accuracy = (
        correct_pixels /
        total_pixels
    )

    # ========================================================
    # PER-CLASS METRICS
    # ========================================================

    metrics = []

    for class_id in range(
        NUM_CLASSES
    ):

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

        precision_denominator = (
            true_positive +
            false_positive
        )

        recall_denominator = (
            true_positive +
            false_negative
        )

        union = (
            true_positive +
            false_positive +
            false_negative
        )

        precision = (
            true_positive /
            precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive /
            recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        if (
            precision + recall
        ) > 0:

            f1 = (
                2 *
                precision *
                recall /
                (precision + recall)
            )

        else:

            f1 = 0.0

        iou = (
            true_positive /
            union
            if union > 0
            else 0.0
        )

        metrics.append(
            {
                "class_id": class_id,
                "class_name":
                    CLASS_NAMES[class_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou
            }
        )

    metrics_df = pd.DataFrame(
        metrics
    )

    mean_iou = metrics_df[
        "iou"
    ].mean()

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 75)
    print("EXPERIMENT 4 RESULTS")
    print("=" * 75)

    print()
    print(
        f"Pixel Accuracy: "
        f"{pixel_accuracy:.4f}"
    )

    print()
    print("Per-Class Metrics:")

    for _, row in metrics_df.iterrows():

        print(
            f"{row['class_name']:<18}"
            f" Precision {row['precision']:.4f}"
            f" Recall {row['recall']:.4f}"
            f" F1 {row['f1']:.4f}"
            f" IoU {row['iou']:.4f}"
        )

    print()
    print(
        f"Mean IoU: {mean_iou:.4f}"
    )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()
    print("=" * 75)
    print("CLASS DISTRIBUTION")
    print("=" * 75)

    distribution = []

    for class_id in range(
        NUM_CLASSES
    ):

        actual_percentage = (
            actual_counts[class_id]
            /
            total_pixels
            *
            100
        )

        predicted_percentage = (
            predicted_counts[class_id]
            /
            total_pixels
            *
            100
        )

        print(
            f"{CLASS_NAMES[class_id]:<18}"
            f" Actual {actual_percentage:>8.4f}%"
            f" Predicted {predicted_percentage:>8.4f}%"
        )

        distribution.append(
            {
                "class_id": class_id,
                "class_name":
                    CLASS_NAMES[class_id],
                "actual_pixels":
                    int(actual_counts[class_id]),
                "predicted_pixels":
                    int(predicted_counts[class_id]),
                "actual_percentage":
                    actual_percentage,
                "predicted_percentage":
                    predicted_percentage
            }
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 75)
    print("CONFUSION MATRIX")
    print("=" * 75)

    print()
    print(
        "Actual / Predicted"
    )

    print(
        f"{'':<18}"
        f"{'Background':>14}"
        f"{'No Damage':>14}"
        f"{'Minor':>14}"
        f"{'Major':>14}"
        f"{'Destroyed':>14}"
    )

    for actual_id in range(
        NUM_CLASSES
    ):

        values = confusion_matrix[
            actual_id
        ]

        print(
            f"{CLASS_NAMES[actual_id]:<18}"
            f"{values[0]:>14,}"
            f"{values[1]:>14,}"
            f"{values[2]:>14,}"
            f"{values[3]:>14,}"
            f"{values[4]:>14,}"
        )

    # ========================================================
    # CONSISTENCY CHECKS
    # ========================================================

    confusion_total = int(
        confusion_matrix.sum()
    )

    actual_total = int(
        actual_counts.sum()
    )

    predicted_total = int(
        predicted_counts.sum()
    )

    print()
    print("=" * 75)
    print("CONSISTENCY CHECKS")
    print("=" * 75)

    print(
        "Total pixels          :",
        int(total_pixels)
    )

    print(
        "Confusion matrix total:",
        confusion_total
    )

    print(
        "Actual counts total   :",
        actual_total
    )

    print(
        "Predicted counts total:",
        predicted_total
    )

    consistency_errors = []

    if confusion_total != total_pixels:

        consistency_errors.append(
            "confusion matrix total != total pixels"
        )

    if actual_total != total_pixels:

        consistency_errors.append(
            "actual counts total != total pixels"
        )

    if predicted_total != total_pixels:

        consistency_errors.append(
            "predicted counts total != total pixels"
        )

    if int(correct_pixels) != int(
        np.trace(confusion_matrix)
    ):

        consistency_errors.append(
            "correct pixels != confusion matrix trace"
        )

    if consistency_errors:

        print()
        print("INCONSISTENCIES DETECTED:")

        for error in consistency_errors:

            print("  -", error)

        raise RuntimeError(
            "Consistency checks failed; outputs not saved"
        )

    print("All consistency checks passed.")

    # ========================================================
    # SAVE FILES
    # ========================================================

    metrics_df.to_csv(
        os.path.join(
            output_dir,
            "metrics.csv"
        ),
        index=False
    )

    distribution_df = pd.DataFrame(
        distribution
    )

    distribution_df.to_csv(
        os.path.join(
            output_dir,
            "class_distribution.csv"
        ),
        index=False
    )

    confusion_df = pd.DataFrame(
        confusion_matrix,
        index=[
            CLASS_NAMES[i]
            for i in range(NUM_CLASSES)
        ],
        columns=[
            CLASS_NAMES[i]
            for i in range(NUM_CLASSES)
        ]
    )

    confusion_df.to_csv(
        os.path.join(
            output_dir,
            "confusion_matrix.csv"
        )
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "step": "9.2",
        "split": split,
        "split_csv": split_csv,
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "checkpoint_val_loss": float(
            checkpoint["val_loss"]
        ),
        "device": str(DEVICE),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_classes": NUM_CLASSES,
        "class_mapping": {
            str(class_id): CLASS_NAMES[class_id]
            for class_id in range(NUM_CLASSES)
        },
        "evaluated_samples": len(dataset),
        "evaluated_pixels": int(total_pixels),
        "metrics_convention": {
            "pixel_accuracy":
                "correct_pixels / total_pixels",
            "precision":
                "TP / (TP + FP); 0.0 when denominator is 0",
            "recall":
                "TP / (TP + FN); 0.0 when denominator is 0",
            "f1":
                "2*P*R / (P + R); 0.0 when P + R is 0",
            "iou":
                "TP / (TP + FP + FN); 0.0 when union is 0",
            "mean_iou":
                "unweighted mean of per-class IoU over all "
                f"{NUM_CLASSES} classes (zero-union classes "
                "contribute 0.0)",
            "confusion_matrix":
                "rows = actual (ground truth), "
                "columns = predicted"
        }
    }

    metadata_path = os.path.join(
        output_dir,
        "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)

    print()
    print("Saved:")
    print(
        os.path.join(
            output_dir,
            "metrics.csv"
        )
    )

    print("Saved:")
    print(
        os.path.join(
            output_dir,
            "class_distribution.csv"
        )
    )

    print("Saved:")
    print(
        os.path.join(
            output_dir,
            "confusion_matrix.csv"
        )
    )

    print("Saved:")
    print(
        metadata_path
    )


if __name__ == "__main__":
    main()