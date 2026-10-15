import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SCRIPT_DIR)

from dataset_change_aware import XBDChangeAwareDataset
from unet_change_aware import UNetChangeAware


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_ROOT = r"D:\Projects\Datasets\xBD"

# "val" is the split used for checkpoint selection and is the
# default here so that failure inspection for improvement
# planning never touches the held-out test set. "test" is
# available to reproduce examples matching the Step 9.2
# held-out baseline (diagnosis only, never used for tuning).
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

# Each split writes to its own directory so results can never
# overwrite the Step 9.2 evaluation outputs.
OUTPUT_DIRS = {
    "val": os.path.join(
        PROJECT_ROOT,
        "outputs",
        "change_aware_failure_analysis",
        "val"
    ),
    "test": os.path.join(
        PROJECT_ROOT,
        "outputs",
        "change_aware_failure_analysis",
        "test"
    )
}

# Reference confusion matrix produced by Step 9.2 for the
# held-out test split. Used only as a consistency check.
REFERENCE_TEST_CONFUSION = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "change_aware_test_evaluation",
    "confusion_matrix.csv"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "checkpoints",
    "best_model_change_aware.pth"
)

IMAGE_SIZE = 256
NUM_CLASSES = 5

DAMAGE_CLASSES = [2, 3, 4]

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed"
}

# Same palette used by the existing failure-analysis scripts.
CLASS_COLORS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.65, 0.65, 0.65],
        [1.0, 1.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 0.0, 0.0]
    ],
    dtype=np.float32
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description=(
            "Analyze change-aware U-Net damage-severity "
            "failures on a dataset split (diagnosis only)."
        )
    )

    parser.add_argument(
        "--split",
        choices=sorted(SPLITS.keys()),
        default=DEFAULT_SPLIT,
        help=(
            "Split manifest to analyze. Defaults to 'val' "
            "(checkpoint-selection split, safe for planning); "
            "use 'test' to reproduce the held-out baseline."
        )
    )

    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=4,
        help="Number of example figures per damage class."
    )

    parser.add_argument(
        "--min-gt-pixels",
        type=int,
        default=200,
        help=(
            "Minimum ground-truth pixels of the focus class "
            "for a sample to be eligible as an example."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size."
    )

    return parser.parse_args(argv)


# ============================================================
# HELPERS
# ============================================================

def mask_to_rgb(mask):

    return CLASS_COLORS[mask]


def tensor_to_rgb(tensor):

    image = tensor.numpy()

    image = np.transpose(image, (1, 2, 0))

    image = np.clip(image, 0, 1)

    return image


def calculate_per_class_metrics(confusion_matrix):

    metrics = []

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

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        union = true_positive + false_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        iou = (
            true_positive / union
            if union > 0
            else 0.0
        )

        metrics.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou
            }
        )

    return pd.DataFrame(metrics)


def build_misclassification_summary(confusion_matrix):

    rows = []

    for class_id in DAMAGE_CLASSES:

        row = confusion_matrix[class_id]

        gt_pixels = int(row.sum())

        correct_pixels = int(row[class_id])

        recall = (
            correct_pixels / gt_pixels
            if gt_pixels > 0
            else 0.0
        )

        # Rank the destinations for wrongly classified
        # ground-truth pixels of this class.
        destinations = []

        for other_id in range(NUM_CLASSES):

            if other_id == class_id:
                continue

            count = int(row[other_id])

            destinations.append(
                {
                    "class_id": other_id,
                    "class_name": CLASS_NAMES[other_id],
                    "pixels": count,
                    "pct_of_gt": (
                        count / gt_pixels * 100
                        if gt_pixels > 0
                        else 0.0
                    )
                }
            )

        destinations.sort(
            key=lambda item: item["pixels"],
            reverse=True
        )

        top = destinations[0]

        background = row[0]
        no_damage = row[1]

        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "gt_pixels": gt_pixels,
                "correct_pixels": correct_pixels,
                "recall": recall,
                "top_confusion_class_id":
                    top["class_id"],
                "top_confusion_class":
                    top["class_name"],
                "top_confusion_pixels":
                    top["pixels"],
                "top_confusion_pct_of_gt":
                    top["pct_of_gt"],
                "predicted_background_pixels":
                    int(background),
                "predicted_background_pct_of_gt":
                    (
                        background / gt_pixels * 100
                        if gt_pixels > 0
                        else 0.0
                    ),
                "predicted_no_damage_pixels":
                    int(no_damage),
                "predicted_no_damage_pct_of_gt":
                    (
                        no_damage / gt_pixels * 100
                        if gt_pixels > 0
                        else 0.0
                    )
            }
        )

    return pd.DataFrame(rows)


def save_example_figure(
    output_path,
    pre_image,
    post_image,
    target,
    prediction,
    sample_id,
    focus_class,
    focus_iou,
    pixel_accuracy,
    misclassified_pixels
):

    ground_truth_rgb = mask_to_rgb(target)
    prediction_rgb = mask_to_rgb(prediction)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    axes[0].imshow(pre_image)
    axes[0].set_title("Before Disaster")
    axes[0].axis("off")

    axes[1].imshow(post_image)
    axes[1].set_title("After Disaster")
    axes[1].axis("off")

    axes[2].imshow(ground_truth_rgb)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    axes[3].imshow(prediction_rgb)
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    plt.suptitle(
        f"{sample_id} | focus: {focus_class} | "
        f"IoU(focus)={focus_iou:.4f} | "
        f"pixel_acc={pixel_accuracy:.4f} | "
        f"misclassified(focus)={misclassified_pixels:,} px",
        fontsize=12
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    split = args.split
    split_csv = SPLITS[split]
    output_dir = OUTPUT_DIRS[split]
    examples_dir = os.path.join(output_dir, "examples")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)

    print("=" * 75)
    print("CHANGE-AWARE U-NET DAMAGE-SEVERITY FAILURE ANALYSIS")
    print("=" * 75)

    print()
    print("Device           :", DEVICE)
    print("Split            :", split)
    print("Split CSV        :", split_csv)
    print("Checkpoint       :", CHECKPOINT_PATH)
    print("Output dir       :", output_dir)
    print("Examples/class   :", args.examples_per_class)
    print("Min GT pixels    :", args.min_gt_pixels)
    print("Input            : 9 channels")
    print("Classes          :", NUM_CLASSES)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = XBDChangeAwareDataset(
        DATASET_ROOT,
        split_csv,
        image_size=IMAGE_SIZE
    )

    sample_ids = [
        str(value)
        for value in dataset.df.iloc[:, 0].tolist()
    ]

    expected_samples = len(pd.read_csv(split_csv))

    print()
    print("Split CSV rows   :", expected_samples)
    print("Dataset samples  :", len(dataset))

    if len(dataset) != expected_samples:

        raise RuntimeError(
            "Dataset length does not match the split manifest "
            f"({len(dataset)} != {expected_samples})"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
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
        "Loaded checkpoint epoch:",
        checkpoint["epoch"]
    )

    print(
        "Checkpoint val loss    :",
        f"{checkpoint['val_loss']:.4f}"
    )
    print("Preprocessing          : 3+3+3 channels, "
          "masks resized NEAREST to 256")

    # --------------------------------------------------------
    # INFERENCE PASS
    # --------------------------------------------------------

    num_samples = len(dataset)

    confusion_matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    actual_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    predicted_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    per_image_confusion = np.zeros(
        (num_samples, NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    # uint8 is sufficient for class ids 0..4
    predictions_store = np.zeros(
        (num_samples, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    total_pixels = 0
    correct_pixels = 0

    print()
    print("Running inference...")

    with torch.no_grad():

        for batch_index, (
            images,
            targets
        ) in enumerate(loader):

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            ).cpu().numpy().astype(np.uint8)

            targets_np = targets.numpy().astype(np.int64)

            for local_index in range(predictions.shape[0]):

                global_index = (
                    batch_index * args.batch_size
                    + local_index
                )

                prediction = predictions[local_index]
                target = targets_np[local_index]

                predictions_store[global_index] = prediction

                pairs = (
                    target.reshape(-1) * NUM_CLASSES
                    + prediction.reshape(-1)
                )

                image_confusion = np.bincount(
                    pairs,
                    minlength=NUM_CLASSES * NUM_CLASSES
                ).reshape(NUM_CLASSES, NUM_CLASSES)

                per_image_confusion[global_index] = image_confusion
                confusion_matrix += image_confusion

                actual_counts += image_confusion.sum(axis=1)
                predicted_counts += image_confusion.sum(axis=0)

                total_pixels += target.size
                correct_pixels += int(
                    image_confusion.trace()
                )

            if (batch_index + 1) % 25 == 0:

                print(
                    f"Processed {batch_index + 1}/"
                    f"{len(loader)} batches"
                )

    # --------------------------------------------------------
    # CONSISTENCY CHECKS
    # --------------------------------------------------------

    consistency_errors = []

    if int(confusion_matrix.sum()) != total_pixels:

        consistency_errors.append(
            "confusion total != total pixels"
        )

    if int(actual_counts.sum()) != total_pixels:

        consistency_errors.append(
            "actual counts total != total pixels"
        )

    if int(predicted_counts.sum()) != total_pixels:

        consistency_errors.append(
            "predicted counts total != total pixels"
        )

    if int(correct_pixels) != int(confusion_matrix.trace()):

        consistency_errors.append(
            "correct pixels != confusion trace"
        )

    if consistency_errors:

        for error in consistency_errors:
            print("CONSISTENCY ERROR:", error)

        raise RuntimeError(
            "Consistency checks failed; refine the script."
        )

    print()
    print("All consistency checks passed.")
    print("Total pixels:", int(total_pixels))

    # --------------------------------------------------------
    # REPRODUCE STEP 9.2 (test split only)
    # --------------------------------------------------------

    if split == "test" and os.path.exists(
        REFERENCE_TEST_CONFUSION
    ):

        reference = pd.read_csv(
            REFERENCE_TEST_CONFUSION,
            index_col=0
        ).to_numpy()

        matches = np.array_equal(
            reference.astype(np.int64),
            confusion_matrix
        )

        print()
        print("Step 9.2 test confusion matrix match:", matches)

        if not matches:

            raise RuntimeError(
                "Aggregate confusion matrix does not match "
                "the Step 9.2 reference; aborting."
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics_df = calculate_per_class_metrics(
        confusion_matrix
    )

    pixel_accuracy = correct_pixels / total_pixels
    mean_iou = metrics_df["iou"].mean()

    print()
    print("=" * 75)
    print("PER-CLASS METRICS")
    print("=" * 75)

    for _, row in metrics_df.iterrows():

        print(
            f"{row['class_name']:<18}"
            f" Precision {row['precision']:.4f}"
            f" Recall {row['recall']:.4f}"
            f" F1 {row['f1']:.4f}"
            f" IoU {row['iou']:.4f}"
        )

    print()
    print(f"Pixel Accuracy: {pixel_accuracy:.6f}")
    print(f"Mean IoU      : {mean_iou:.6f}")

    # --------------------------------------------------------
    # MISCLASSIFICATION SUMMARY
    # --------------------------------------------------------

    summary_df = build_misclassification_summary(
        confusion_matrix
    )

    print()
    print("=" * 75)
    print("DAMAGE-SEVERITY MISCLASSIFICATION SUMMARY")
    print("=" * 75)

    for _, row in summary_df.iterrows():

        print()
        print(
            f"{row['class_name']}  "
            f"(GT pixels = {row['gt_pixels']:,})"
        )
        print(
            f"  correct (IoU recall) : "
            f"{row['correct_pixels']:,} "
            f"({row['recall'] * 100:.4f}%)"
        )
        print(
            f"  top confusion        : "
            f"{row['top_confusion_class']} "
            f"({row['top_confusion_pixels']:,} px, "
            f"{row['top_confusion_pct_of_gt']:.2f}% of GT)"
        )
        print(
            f"  -> Background        : "
            f"{row['predicted_background_pixels']:,} "
            f"({row['predicted_background_pct_of_gt']:.2f}%)"
        )
        print(
            f"  -> No Damage         : "
            f"{row['predicted_no_damage_pixels']:,} "
            f"({row['predicted_no_damage_pct_of_gt']:.2f}%)"
        )

    # --------------------------------------------------------
    # EXAMPLE SELECTION
    # --------------------------------------------------------
    # Deterministic: rank by misclassified ground-truth
    # pixels of the focus class (descending), tie-break by
    # sample id (ascending). Each sample appears at most once
    # across all focus classes.

    print()
    print("=" * 75)
    print("SELECTING REPRESENTATIVE FAILURES")
    print("=" * 75)

    selected_rows = []
    chosen_sample_ids = set()

    for class_id in DAMAGE_CLASSES:

        focus_name = CLASS_NAMES[class_id]

        candidates = []

        for index in range(num_samples):

            gt_pixels = int(
                per_image_confusion[index, class_id, :].sum()
            )

            if gt_pixels < args.min_gt_pixels:
                continue

            correct = int(
                per_image_confusion[index, class_id, class_id]
            )

            misclassified = gt_pixels - correct

            candidates.append(
                (misclassified, gt_pixels, correct, index)
            )

        # Sort by misclassified desc, then sample id asc.
        candidates.sort(
            key=lambda item: (
                -item[0],
                sample_ids[item[3]]
            )
        )

        chosen_for_class = 0

        for (
            misclassified,
            gt_pixels,
            correct,
            index
        ) in candidates:

            if chosen_for_class >= args.examples_per_class:
                break

            sample_id = sample_ids[index]

            if sample_id in chosen_sample_ids:
                continue

            chosen_sample_ids.add(sample_id)
            chosen_for_class += 1

            # Focus-class IoU for this image.
            tp = correct
            fp = int(
                per_image_confusion[index, :, class_id].sum()
                - tp
            )
            fn = misclassified
            union = tp + fp + fn

            focus_iou = tp / union if union > 0 else 0.0

            total_image_pixels = int(
                per_image_confusion[index].sum()
            )
            image_correct = int(
                per_image_confusion[index].trace()
            )

            image_pixel_accuracy = (
                image_correct / total_image_pixels
            )

            image_tensor, target_tensor = dataset[index]

            pre_image = tensor_to_rgb(image_tensor[:3])
            post_image = tensor_to_rgb(image_tensor[3:6])

            target = target_tensor.numpy()
            prediction = predictions_store[index]

            filename = (
                f"{class_id}_"
                f"{focus_name.lower().replace(' ', '_')}_"
                f"{sample_id}.png"
            )

            output_path = os.path.join(
                examples_dir,
                filename
            )

            save_example_figure(
                output_path,
                pre_image,
                post_image,
                target,
                prediction,
                sample_id,
                focus_name,
                focus_iou,
                image_pixel_accuracy,
                misclassified
            )

            selected_rows.append(
                {
                    "focus_class_id": class_id,
                    "focus_class_name": focus_name,
                    "sample_id": sample_id,
                    "dataset_index": index,
                    "gt_focus_pixels": gt_pixels,
                    "correct_focus_pixels": correct,
                    "misclassified_focus_pixels": misclassified,
                    "focus_iou": focus_iou,
                    "image_pixel_accuracy":
                        image_pixel_accuracy,
                    "example_file":
                        os.path.join("examples", filename)
                }
            )

            print(
                f"{focus_name:<15} {sample_id}  "
                f"GT {gt_pixels:>6,}  "
                f"misclassified {misclassified:>6,}  "
                f"IoU {focus_iou:.4f}  "
                f"-> {filename}"
            )

    selected_df = pd.DataFrame(selected_rows)

    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    metrics_df.to_csv(
        os.path.join(output_dir, "metrics.csv"),
        index=False
    )

    confusion_df = pd.DataFrame(
        confusion_matrix,
        index=[
            CLASS_NAMES[i] for i in range(NUM_CLASSES)
        ],
        columns=[
            CLASS_NAMES[i] for i in range(NUM_CLASSES)
        ]
    )

    confusion_df.to_csv(
        os.path.join(output_dir, "confusion_matrix.csv")
    )

    summary_df.to_csv(
        os.path.join(
            output_dir,
            "misclassification_summary.csv"
        ),
        index=False
    )

    selected_df.to_csv(
        os.path.join(
            output_dir,
            "selected_examples.csv"
        ),
        index=False
    )

    metadata = {
        "step": "9.3",
        "analysis": "damage-severity failure analysis",
        "split": split,
        "split_csv": split_csv,
        "checkpoint": os.path.relpath(
            CHECKPOINT_PATH, PROJECT_ROOT
        ),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_train_loss": float(
            checkpoint["train_loss"]
        ),
        "checkpoint_val_loss": float(
            checkpoint["val_loss"]
        ),
        "device": str(DEVICE),
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "class_mapping": {
            str(class_id): CLASS_NAMES[class_id]
            for class_id in range(NUM_CLASSES)
        },
        "class_colors_rgb": {
            CLASS_NAMES[class_id]: (
                CLASS_COLORS[class_id].tolist()
            )
            for class_id in range(NUM_CLASSES)
        },
        "evaluated_samples": num_samples,
        "evaluated_pixels": int(total_pixels),
        "pixel_accuracy": float(pixel_accuracy),
        "mean_iou": float(mean_iou),
        "example_selection": {
            "examples_per_class": args.examples_per_class,
            "min_gt_pixels": args.min_gt_pixels,
            "rule": (
                "For each damage class, rank samples with at "
                "least min_gt_pixels ground-truth pixels of "
                "that class by misclassified ground-truth "
                "pixels of the focus class (descending), "
                "tie-break by sample id (ascending). Each "
                "sample is used at most once across classes."
            )
        },
        "metrics_convention": {
            "iou": "TP / (TP + FP + FN); 0.0 when union is 0",
            "mean_iou": (
                "unweighted mean of per-class IoU over all "
                f"{NUM_CLASSES} classes"
            ),
            "confusion_matrix": (
                "rows = actual (ground truth), "
                "columns = predicted"
            )
        },
        "notes": [
            "Diagnosis only; the held-out test split is never "
            "used for tuning.",
            "Improvement experiments must be selected and "
            "validated on the validation split."
        ]
    }

    metadata_path = os.path.join(
        output_dir,
        "analysis_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(metadata, handle, indent=2)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FAILURE ANALYSIS COMPLETE")
    print("=" * 75)

    print()
    print("Saved:")
    print(" ", os.path.join(output_dir, "metrics.csv"))
    print(" ", os.path.join(output_dir, "confusion_matrix.csv"))
    print(" ", os.path.join(
        output_dir, "misclassification_summary.csv"))
    print(" ", os.path.join(output_dir, "selected_examples.csv"))
    print(" ", metadata_path)
    print(" ", examples_dir)


if __name__ == "__main__":
    main()
