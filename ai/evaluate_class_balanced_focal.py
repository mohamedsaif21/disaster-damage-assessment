import os
import sys
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.dataset import XBDDataset
from ai.unet import UNet


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 5
IMAGE_SIZE = 256
BATCH_SIZE = 2

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
    "best_model_class_balanced_focal.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "class_balanced_focal_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Class names
# ---------------------------------------------------------

CLASS_NAMES = [
    "Background",
    "No Damage",
    "Minor Damage",
    "Major Damage",
    "Destroyed"
]


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

def load_model():

    print("Loading U-Net...")

    model = UNet(
        in_channels=6,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Model loaded from:\n{MODEL_PATH}"
    )

    if "epoch" in checkpoint:
        print(
            f"Checkpoint epoch: {checkpoint['epoch']}"
        )

    if "val_loss" in checkpoint:
        print(
            f"Checkpoint validation loss: "
            f"{checkpoint['val_loss']:.4f}"
        )

    return model


# ---------------------------------------------------------
# Build confusion matrix
# ---------------------------------------------------------

def calculate_confusion_matrix(
    model,
    dataloader
):

    confusion_matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    actual_pixels = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    predicted_pixels = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    total_correct = 0
    total_pixels = 0

    model.eval()

    with torch.no_grad():

        for batch_idx, (images, targets) in enumerate(
            dataloader,
            start=1
        ):

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            targets_np = targets.cpu().numpy().reshape(-1)
            predictions_np = predictions.cpu().numpy().reshape(-1)

            # ---------------------------------------------
            # Accuracy
            # ---------------------------------------------

            total_correct += np.sum(
                predictions_np == targets_np
            )

            total_pixels += len(targets_np)

            # ---------------------------------------------
            # Class distributions
            # ---------------------------------------------

            actual_counts = np.bincount(
                targets_np,
                minlength=NUM_CLASSES
            )

            predicted_counts = np.bincount(
                predictions_np,
                minlength=NUM_CLASSES
            )

            actual_pixels += actual_counts
            predicted_pixels += predicted_counts

            # ---------------------------------------------
            # Confusion matrix
            #
            # rows    = actual
            # columns = predicted
            # ---------------------------------------------

            combined = (
                targets_np * NUM_CLASSES
                + predictions_np
            )

            batch_matrix = np.bincount(
                combined,
                minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(
                NUM_CLASSES,
                NUM_CLASSES
            )

            confusion_matrix += batch_matrix

            if batch_idx % 20 == 0:

                print(
                    f"Processed "
                    f"{batch_idx}/{len(dataloader)} "
                    f"batches"
                )

    pixel_accuracy = (
        total_correct / total_pixels
    )

    return (
        confusion_matrix,
        actual_pixels,
        predicted_pixels,
        pixel_accuracy
    )


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

def calculate_metrics(
    confusion_matrix
):

    metrics = []

    for class_index in range(NUM_CLASSES):

        true_positive = confusion_matrix[
            class_index,
            class_index
        ]

        false_positive = (
            confusion_matrix[:, class_index].sum()
            - true_positive
        )

        false_negative = (
            confusion_matrix[class_index, :].sum()
            - true_positive
        )

        # ---------------------------------------------
        # Precision
        # ---------------------------------------------

        precision_denominator = (
            true_positive + false_positive
        )

        if precision_denominator > 0:

            precision = (
                true_positive
                / precision_denominator
            )

        else:

            precision = 0.0

        # ---------------------------------------------
        # Recall
        # ---------------------------------------------

        recall_denominator = (
            true_positive + false_negative
        )

        if recall_denominator > 0:

            recall = (
                true_positive
                / recall_denominator
            )

        else:

            recall = 0.0

        # ---------------------------------------------
        # F1
        # ---------------------------------------------

        f1_denominator = (
            precision + recall
        )

        if f1_denominator > 0:

            f1 = (
                2
                * precision
                * recall
                / f1_denominator
            )

        else:

            f1 = 0.0

        # ---------------------------------------------
        # IoU
        # ---------------------------------------------

        iou_denominator = (
            true_positive
            + false_positive
            + false_negative
        )

        if iou_denominator > 0:

            iou = (
                true_positive
                / iou_denominator
            )

        else:

            iou = 0.0

        metrics.append(
            {
                "class": CLASS_NAMES[class_index],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "true_positive": int(true_positive),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative)
            }
        )

    return metrics


# ---------------------------------------------------------
# Save metrics CSV
# ---------------------------------------------------------

def save_metrics_csv(metrics):

    path = os.path.join(
        OUTPUT_DIR,
        "metrics.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Class",
                "Precision",
                "Recall",
                "F1",
                "IoU",
                "True Positive",
                "False Positive",
                "False Negative"
            ]
        )

        for metric in metrics:

            writer.writerow(
                [
                    metric["class"],
                    f"{metric['precision']:.6f}",
                    f"{metric['recall']:.6f}",
                    f"{metric['f1']:.6f}",
                    f"{metric['iou']:.6f}",
                    metric["true_positive"],
                    metric["false_positive"],
                    metric["false_negative"]
                ]
            )

    print(
        f"\nSaved metrics:\n{path}"
    )


# ---------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------

def save_confusion_matrix_csv(
    confusion_matrix
):

    path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["Actual / Predicted"]
            + CLASS_NAMES
        )

        for index, row in enumerate(
            confusion_matrix
        ):

            writer.writerow(
                [CLASS_NAMES[index]]
                + row.tolist()
            )

    print(
        f"Saved confusion matrix:\n{path}"
    )


# ---------------------------------------------------------
# Save class distribution
# ---------------------------------------------------------

def save_class_distribution(
    actual_pixels,
    predicted_pixels
):

    path = os.path.join(
        OUTPUT_DIR,
        "class_distribution.csv"
    )

    total_actual = actual_pixels.sum()
    total_predicted = predicted_pixels.sum()

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Class",
                "Actual Pixels",
                "Actual %",
                "Predicted Pixels",
                "Predicted %"
            ]
        )

        for index in range(NUM_CLASSES):

            actual_percentage = (
                actual_pixels[index]
                / total_actual
                * 100
            )

            predicted_percentage = (
                predicted_pixels[index]
                / total_predicted
                * 100
            )

            writer.writerow(
                [
                    CLASS_NAMES[index],
                    int(actual_pixels[index]),
                    f"{actual_percentage:.4f}",
                    int(predicted_pixels[index]),
                    f"{predicted_percentage:.4f}"
                ]
            )

    print(
        f"Saved class distribution:\n{path}"
    )


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

def print_results(
    pixel_accuracy,
    metrics,
    actual_pixels,
    predicted_pixels
):

    print()
    print("=" * 75)
    print("EXPERIMENT 3 EVALUATION RESULTS")
    print("=" * 75)

    print()
    print(
        f"Pixel Accuracy: "
        f"{pixel_accuracy:.4f}"
    )

    print()
    print("Per-Class Metrics")
    print("-" * 75)

    print(
        f"{'Class':<18}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'IoU':>12}"
    )

    print("-" * 75)

    ious = []

    for metric in metrics:

        print(
            f"{metric['class']:<18}"
            f"{metric['precision']:>12.4f}"
            f"{metric['recall']:>12.4f}"
            f"{metric['f1']:>12.4f}"
            f"{metric['iou']:>12.4f}"
        )

        ious.append(
            metric["iou"]
        )

    mean_iou = np.mean(ious)

    print("-" * 75)

    print(
        f"{'Mean IoU':<18}"
        f"{mean_iou:>48.4f}"
    )

    # -----------------------------------------------------
    # Distribution
    # -----------------------------------------------------

    print()
    print("Class Distribution")
    print("-" * 75)

    total_actual = actual_pixels.sum()
    total_predicted = predicted_pixels.sum()

    for index in range(NUM_CLASSES):

        actual_percentage = (
            actual_pixels[index]
            / total_actual
            * 100
        )

        predicted_percentage = (
            predicted_pixels[index]
            / total_predicted
            * 100
        )

        print(
            f"{CLASS_NAMES[index]:<18}"
            f"Actual: {actual_percentage:>8.4f}%"
            f"   Predicted: {predicted_percentage:>8.4f}%"
        )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print()
    print("Confusion Matrix")
    print("-" * 90)

    print(
        f"{'Actual / Pred':<18}"
        + "".join(
            f"{name[:10]:>14}"
            for name in CLASS_NAMES
        )
    )

    for index, row in enumerate(
        confusion_matrix_global
    ):

        print(
            f"{CLASS_NAMES[index]:<18}"
            + "".join(
                f"{value:>14,}"
                for value in row
            )
        )

    print()
    print("=" * 75)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 75)
    print("EXPERIMENT 3")
    print("Class-Balanced Focal + Dice Evaluation")
    print("=" * 75)

    print()
    print(f"Device: {DEVICE}")
    print(f"Validation samples: loading...")
    print()

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    val_dataset = XBDDataset(
        dataset_root=DATASET_ROOT,
        split_file=VAL_CSV,
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

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = load_model()

    print()

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    (
        confusion_matrix,
        actual_pixels,
        predicted_pixels,
        pixel_accuracy
    ) = calculate_confusion_matrix(
        model,
        val_loader
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    metrics = calculate_metrics(
        confusion_matrix
    )

    # -----------------------------------------------------
    # Mean IoU
    # -----------------------------------------------------

    mean_iou = np.mean(
        [
            metric["iou"]
            for metric in metrics
        ]
    )

    # -----------------------------------------------------
    # Print main results
    # -----------------------------------------------------

    print()
    print("=" * 75)
    print("RESULTS")
    print("=" * 75)

    print(
        f"\nPixel Accuracy: {pixel_accuracy:.4f}"
    )

    print()
    print("Per-Class Metrics:")

    for metric in metrics:

        print(
            f"{metric['class']:<18}"
            f"Precision {metric['precision']:.4f} "
            f"Recall {metric['recall']:.4f} "
            f"F1 {metric['f1']:.4f} "
            f"IoU {metric['iou']:.4f}"
        )

    print(
        f"\nMean IoU: {mean_iou:.4f}"
    )

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    print()
    print("Class Distribution:")

    total_actual = actual_pixels.sum()
    total_predicted = predicted_pixels.sum()

    for index in range(NUM_CLASSES):

        actual_percentage = (
            actual_pixels[index]
            / total_actual
            * 100
        )

        predicted_percentage = (
            predicted_pixels[index]
            / total_predicted
            * 100
        )

        print(
            f"{CLASS_NAMES[index]:<18}"
            f"Actual {actual_percentage:>8.4f}% "
            f"Predicted {predicted_percentage:>8.4f}%"
        )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print()
    print("Confusion Matrix:")

    print(
        f"{'Actual / Pred':<18}"
        + "".join(
            f"{name[:10]:>14}"
            for name in CLASS_NAMES
        )
    )

    for index, row in enumerate(
        confusion_matrix
    ):

        print(
            f"{CLASS_NAMES[index]:<18}"
            + "".join(
                f"{value:>14,}"
                for value in row
            )
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_metrics_csv(
        metrics
    )

    save_confusion_matrix_csv(
        confusion_matrix
    )

    save_class_distribution(
        actual_pixels,
        predicted_pixels
    )

    print()
    print("=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )