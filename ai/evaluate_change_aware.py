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
    r"outputs\change_aware_evaluation"
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
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 75)
    print("EXPERIMENT 4 — CHANGE-AWARE U-NET EVALUATION")
    print("=" * 75)

    print()
    print("Device    :", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Input     : 9 channels")
    print("Classes   : 5")
    print("Image size:", IMAGE_SIZE)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    val_dataset = XBDChangeAwareDataset(
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

    print()
    print(
        "Validation samples:",
        len(val_dataset)
    )

    print(
        "Validation batches:",
        len(val_loader)
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
        ) in enumerate(val_loader):

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
                    f"{len(val_loader)} batches"
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
    # SAVE FILES
    # ========================================================

    metrics_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "metrics.csv"
        ),
        index=False
    )

    distribution_df = pd.DataFrame(
        distribution
    )

    distribution_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
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
            OUTPUT_DIR,
            "confusion_matrix.csv"
        )
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
            OUTPUT_DIR,
            "metrics.csv"
        )
    )

    print("Saved:")
    print(
        os.path.join(
            OUTPUT_DIR,
            "class_distribution.csv"
        )
    )

    print("Saved:")
    print(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.csv"
        )
    )


if __name__ == "__main__":
    main()