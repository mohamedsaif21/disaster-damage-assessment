import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from ai.dataset import XBDDataset
from ai.unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

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
    "best_model_focal_dice.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "focal_confidence"
)

IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed"
}

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FOCAL + DICE CONFIDENCE ANALYSIS")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Validation   : {VAL_CSV}")
print(f"Model        : {MODEL_PATH}")
print(f"Output Dir   : {OUTPUT_DIR}")
print(f"Image Size   : {IMAGE_SIZE}")
print()


# ============================================================
# LOAD DATASET
# ============================================================

dataset = XBDDataset(
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

print(
    f"Validation samples: {len(dataset)}"
)

print(
    f"Validation batches: {len(loader)}"
)

print()


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet(
    in_channels=6,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# STORAGE
# ============================================================

# Confidence of the correct class for each actual class
correct_class_confidences = {
    class_id: []
    for class_id in range(NUM_CLASSES)
}

# Confidence when the prediction is correct
correct_prediction_confidences = {
    class_id: []
    for class_id in range(NUM_CLASSES)
}

# Confidence when prediction is incorrect
incorrect_prediction_confidences = {
    class_id: []
    for class_id in range(NUM_CLASSES)
}

# For every actual class:
# what class did the model predict?
confusion_counts = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

# Store pixel-level records for detailed analysis
records = []


# ============================================================
# EVALUATION
# ============================================================

print("Analyzing prediction probabilities...")
print()


with torch.no_grad():

    for batch_idx, (images, targets) in enumerate(loader):

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        probabilities_np = (
            probabilities
            .cpu()
            .numpy()
        )

        predictions_np = (
            predictions
            .cpu()
            .numpy()
        )

        targets_np = (
            targets
            .numpy()
        )

        # ----------------------------------------------------
        # PROCESS EACH IMAGE
        # ----------------------------------------------------

        for image_idx in range(
            images.shape[0]
        ):

            probs = probabilities_np[
                image_idx
            ]

            prediction = predictions_np[
                image_idx
            ]

            target = targets_np[
                image_idx
            ]

            # ------------------------------------------------
            # PROCESS EACH CLASS
            # ------------------------------------------------

            for class_id in range(NUM_CLASSES):

                actual_mask = (
                    target == class_id
                )

                if not np.any(actual_mask):
                    continue

                # Probability assigned to the ACTUAL class
                actual_class_probability = probs[
                    class_id
                ][actual_mask]

                correct_class_confidences[
                    class_id
                ].extend(
                    actual_class_probability.tolist()
                )

            # ------------------------------------------------
            # PREDICTION CONFIDENCE
            # ------------------------------------------------

            predicted_confidence = np.max(
                probs,
                axis=0
            )

            # ------------------------------------------------
            # CONFUSION COUNTS
            # ------------------------------------------------

            for actual_class in range(NUM_CLASSES):

                actual_mask = (
                    target == actual_class
                )

                if not np.any(actual_mask):
                    continue

                predicted_classes = prediction[
                    actual_mask
                ]

                for predicted_class in range(
                    NUM_CLASSES
                ):

                    count = np.sum(
                        predicted_classes
                        ==
                        predicted_class
                    )

                    confusion_counts[
                        actual_class,
                        predicted_class
                    ] += count

                # ------------------------------------------------
                # CORRECT / INCORRECT CONFIDENCE
                # ------------------------------------------------

                correct_mask = (
                    prediction[actual_mask]
                    ==
                    actual_class
                )

                incorrect_mask = (
                    prediction[actual_mask]
                    !=
                    actual_class
                )

                actual_class_probs = probs[
                    actual_class
                ][actual_mask]

                if np.any(correct_mask):

                    correct_prediction_confidences[
                        actual_class
                    ].extend(
                        actual_class_probs[
                            correct_mask
                        ].tolist()
                    )

                if np.any(incorrect_mask):

                    incorrect_prediction_confidences[
                        actual_class
                    ].extend(
                        predicted_confidence[
                            actual_mask
                        ][incorrect_mask]
                        .tolist()
                    )

            # ------------------------------------------------
            # IMAGE-LEVEL SUMMARY
            # ------------------------------------------------

            record = {
                "dataset_index":
                    batch_idx * BATCH_SIZE
                    + image_idx
            }

            for class_id in range(NUM_CLASSES):

                mask = (
                    target == class_id
                )

                if np.any(mask):

                    record[
                        f"actual_{class_id}_pixels"
                    ] = int(
                        np.sum(mask)
                    )

                    record[
                        f"actual_{class_id}_mean_probability"
                    ] = float(
                        np.mean(
                            probs[class_id][mask]
                        )
                    )

                    record[
                        f"actual_{class_id}_max_probability"
                    ] = float(
                        np.max(
                            probs[class_id][mask]
                        )
                    )

                    record[
                        f"actual_{class_id}_min_probability"
                    ] = float(
                        np.min(
                            probs[class_id][mask]
                        )
                    )

                else:

                    record[
                        f"actual_{class_id}_pixels"
                    ] = 0

                    record[
                        f"actual_{class_id}_mean_probability"
                    ] = np.nan

                    record[
                        f"actual_{class_id}_max_probability"
                    ] = np.nan

                    record[
                        f"actual_{class_id}_min_probability"
                    ] = np.nan

            records.append(record)

        if (
            batch_idx + 1
        ) % 20 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1}/"
                f"{len(loader)} batches"
            )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE SUMMARY")
print("=" * 70)
print()


print(
    f"{'Class':<20}"
    f"{'Mean P(actual)':>18}"
    f"{'Correct P':>18}"
    f"{'Wrong P':>18}"
)

print("-" * 74)


summary_rows = []


for class_id in range(NUM_CLASSES):

    actual_values = np.array(
        correct_class_confidences[
            class_id
        ]
    )

    correct_values = np.array(
        correct_prediction_confidences[
            class_id
        ]
    )

    incorrect_values = np.array(
        incorrect_prediction_confidences[
            class_id
        ]
    )

    mean_actual = (
        np.mean(actual_values)
        if len(actual_values) > 0
        else 0.0
    )

    mean_correct = (
        np.mean(correct_values)
        if len(correct_values) > 0
        else 0.0
    )

    mean_incorrect = (
        np.mean(incorrect_values)
        if len(incorrect_values) > 0
        else 0.0
    )

    print(
        f"{CLASS_NAMES[class_id]:<20}"
        f"{mean_actual:>17.4f}"
        f"{mean_correct:>17.4f}"
        f"{mean_incorrect:>17.4f}"
    )

    summary_rows.append({
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "mean_probability_for_actual_class":
            mean_actual,
        "mean_probability_when_correct":
            mean_correct,
        "mean_confidence_when_incorrect":
            mean_incorrect,
        "actual_pixels":
            len(actual_values),
        "correct_pixels":
            len(correct_values),
        "incorrect_pixels":
            len(incorrect_values)
    })


# ============================================================
# CLASS-SPECIFIC ANALYSIS
# ============================================================

print()
print("=" * 70)
print("IMPORTANT DAMAGE CLASS ANALYSIS")
print("=" * 70)
print()


for class_id in [2, 3, 4]:

    values = np.array(
        correct_class_confidences[
            class_id
        ]
    )

    correct_values = np.array(
        correct_prediction_confidences[
            class_id
        ]
    )

    incorrect_values = np.array(
        incorrect_prediction_confidences[
            class_id
        ]
    )

    print(
        f"{class_id} - "
        f"{CLASS_NAMES[class_id]}"
    )

    print(
        f"  Actual pixels analyzed : "
        f"{len(values):,}"
    )

    if len(values) > 0:

        print(
            f"  Mean actual probability: "
            f"{np.mean(values):.4f}"
        )

        print(
            f"  Median actual probability: "
            f"{np.median(values):.4f}"
        )

        print(
            f"  Max actual probability: "
            f"{np.max(values):.4f}"
        )

        print(
            f"  90th percentile: "
            f"{np.percentile(values, 90):.4f}"
        )

    print(
        f"  Correct predictions     : "
        f"{len(correct_values):,}"
    )

    if len(correct_values) > 0:

        print(
            f"  Mean correct confidence: "
            f"{np.mean(correct_values):.4f}"
        )

    print(
        f"  Incorrect predictions   : "
        f"{len(incorrect_values):,}"
    )

    if len(incorrect_values) > 0:

        print(
            f"  Mean incorrect confidence: "
            f"{np.mean(incorrect_values):.4f}"
        )

    print()


# ============================================================
# CONFUSION ANALYSIS
# ============================================================

print("=" * 70)
print("CONFUSION ANALYSIS")
print("=" * 70)
print()


print(
    f"{'Actual / Pred':<18}",
    end=""
)

for class_id in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[class_id][:12]:>14}",
        end=""
    )

print()

print("-" * 88)


for actual_class in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[actual_class]:<18}",
        end=""
    )

    for predicted_class in range(NUM_CLASSES):

        print(
            f"{confusion_counts[actual_class, predicted_class]:>14,}",
            end=""
        )

    print()


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "confidence_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# SAVE IMAGE-LEVEL ANALYSIS
# ============================================================

records_df = pd.DataFrame(
    records
)

records_path = os.path.join(
    OUTPUT_DIR,
    "image_confidence_analysis.csv"
)

records_df.to_csv(
    records_path,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

confusion_df = pd.DataFrame(
    confusion_counts,
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
    confusion_path
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE ANALYSIS COMPLETE")
print("=" * 70)

print()
print(
    f"Summary saved: {summary_path}"
)

print(
    f"Image analysis saved: {records_path}"
)

print(
    f"Confusion matrix saved: {confusion_path}"
)

print()
print(
    "Next step: use these confidence results "
    "to design Experiment 3."
)

print()