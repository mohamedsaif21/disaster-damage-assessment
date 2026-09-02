import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from ai.dataset import XBDDataset
from ai.unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"D:\Projects\Datasets\xBD"
VAL_CSV = os.path.join(DATASET_ROOT, "splits", "val.csv")

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "checkpoints",
    "best_model.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "damage_failures"
)

IMAGE_SIZE = 256

# Number of examples to save for each important damage class
SAMPLES_PER_CLASS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed"
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("xBD DAMAGE FAILURE ANALYSIS")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Dataset Root : {DATASET_ROOT}")
print(f"Validation   : {VAL_CSV}")
print(f"Model        : {MODEL_PATH}")
print(f"Output Dir   : {OUTPUT_DIR}")
print()


dataset = XBDDataset(
    DATASET_ROOT,
    VAL_CSV,
    image_size=IMAGE_SIZE
)

print(f"Validation samples: {len(dataset)}")
print()


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet(
    in_channels=6,
    num_classes=5
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Handle checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# STORAGE FOR SELECTED SAMPLES
# ============================================================

selected_samples = {
    2: [],
    3: [],
    4: []
}


# ============================================================
# FIND SAMPLES CONTAINING DAMAGE CLASSES
# ============================================================

print("Searching validation set for damage examples...")
print()

for index in range(len(dataset)):

    image_tensor, target_tensor = dataset[index]

    target = target_tensor.numpy()

    present_classes = np.unique(target)

    for class_id in [2, 3, 4]:

        if class_id in present_classes:

            if len(selected_samples[class_id]) < SAMPLES_PER_CLASS:
                selected_samples[class_id].append(index)

    # Stop once enough examples have been found
    if all(
        len(selected_samples[c]) >= SAMPLES_PER_CLASS
        for c in [2, 3, 4]
    ):
        break


# ============================================================
# PRINT SELECTED SAMPLES
# ============================================================

print("Selected samples:")
print()

for class_id in [2, 3, 4]:

    print(
        f"{CLASS_NAMES[class_id]:15s}: "
        f"{len(selected_samples[class_id])} samples"
    )

print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def tensor_to_rgb(tensor):
    """
    Convert a 3-channel tensor from [3,H,W]
    to numpy RGB image [H,W,3].
    """

    image = tensor.numpy()

    image = np.transpose(image, (1, 2, 0))

    image = np.clip(image, 0, 1)

    return image


def create_ground_truth_rgb(target):
    """
    Convert target mask to RGB visualization.

    Background  -> black
    No Damage   -> gray
    Minor       -> yellow
    Major       -> orange
    Destroyed   -> red
    """

    h, w = target.shape

    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Background
    rgb[target == 0] = [0.0, 0.0, 0.0]

    # No Damage
    rgb[target == 1] = [0.65, 0.65, 0.65]

    # Minor Damage
    rgb[target == 2] = [1.0, 1.0, 0.0]

    # Major Damage
    rgb[target == 3] = [1.0, 0.5, 0.0]

    # Destroyed
    rgb[target == 4] = [1.0, 0.0, 0.0]

    return rgb


def create_prediction_rgb(prediction):
    """
    Convert prediction mask to RGB visualization.
    """

    return create_ground_truth_rgb(prediction)


# ============================================================
# ANALYZE ONE SAMPLE
# ============================================================

def analyze_sample(index, target_class):

    image_tensor, target_tensor = dataset[index]

    target = target_tensor.numpy()

    # --------------------------------------------------------
    # IMAGE PREPARATION
    # --------------------------------------------------------

    image_input = image_tensor.unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(image_input)

        prediction = torch.argmax(
            output,
            dim=1
        )[0].cpu().numpy()

    # --------------------------------------------------------
    # SPLIT BEFORE / AFTER IMAGES
    # --------------------------------------------------------

    before_tensor = image_tensor[:3]
    after_tensor = image_tensor[3:]

    before_image = tensor_to_rgb(before_tensor)
    after_image = tensor_to_rgb(after_tensor)

    # --------------------------------------------------------
    # CLASS PIXEL COUNTS
    # --------------------------------------------------------

    actual_count = int(np.sum(target == target_class))
    predicted_count = int(np.sum(prediction == target_class))

    total_pixels = target.size

    actual_percentage = (
        actual_count / total_pixels
    ) * 100

    predicted_percentage = (
        predicted_count / total_pixels
    ) * 100

    # --------------------------------------------------------
    # DETERMINE SAMPLE ID
    # --------------------------------------------------------

    sample_id = None

    try:

        val_df = pd.read_csv(VAL_CSV)

        # Try common ID column names
        for column in ["id", "sample_id", "image_id"]:

            if column in val_df.columns:

                sample_id = str(
                    val_df.iloc[index][column]
                )

                break

    except Exception:
        pass

    if sample_id is None:
        sample_id = f"sample_{index:04d}"

    # --------------------------------------------------------
    # PRINT INFORMATION
    # --------------------------------------------------------

    print("-" * 70)

    print(
        f"Sample       : {sample_id}"
    )

    print(
        f"Index        : {index}"
    )

    print(
        f"Target class : {target_class} "
        f"({CLASS_NAMES[target_class]})"
    )

    print(
        f"Actual pixels: {actual_count:,} "
        f"({actual_percentage:.4f}%)"
    )

    print(
        f"Predicted    : {predicted_count:,} "
        f"({predicted_percentage:.4f}%)"
    )

    # --------------------------------------------------------
    # DAMAGE CLASS DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("Actual class distribution:")

    for class_id in range(5):

        count = int(np.sum(target == class_id))

        percentage = (
            count / total_pixels
        ) * 100

        print(
            f"  {class_id} "
            f"{CLASS_NAMES[class_id]:15s}: "
            f"{count:8,} pixels "
            f"({percentage:7.3f}%)"
        )

    print()
    print("Predicted class distribution:")

    for class_id in range(5):

        count = int(np.sum(prediction == class_id))

        percentage = (
            count / total_pixels
        ) * 100

        print(
            f"  {class_id} "
            f"{CLASS_NAMES[class_id]:15s}: "
            f"{count:8,} pixels "
            f"({percentage:7.3f}%)"
        )

    # --------------------------------------------------------
    # CREATE VISUALIZATION
    # --------------------------------------------------------

    ground_truth_rgb = create_ground_truth_rgb(target)
    prediction_rgb = create_prediction_rgb(prediction)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    axes[0].imshow(before_image)
    axes[0].set_title("Before Disaster")
    axes[0].axis("off")

    axes[1].imshow(after_image)
    axes[1].set_title("After Disaster")
    axes[1].axis("off")

    axes[2].imshow(ground_truth_rgb)
    axes[2].set_title(
        f"Ground Truth\n{CLASS_NAMES[target_class]}"
    )
    axes[2].axis("off")

    axes[3].imshow(prediction_rgb)
    axes[3].set_title(
        f"Prediction\n{CLASS_NAMES[target_class]}"
    )
    axes[3].axis("off")

    plt.suptitle(
        f"{sample_id} | Target: {CLASS_NAMES[target_class]}",
        fontsize=14
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    filename = (
        f"{target_class}_"
        f"{CLASS_NAMES[target_class].lower().replace(' ', '_')}_"
        f"{sample_id}.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(f"Saved: {output_path}")
    print()


# ============================================================
# RUN ANALYSIS
# ============================================================

for target_class in [2, 3, 4]:

    print()
    print("=" * 70)

    print(
        f"ANALYZING CLASS {target_class}: "
        f"{CLASS_NAMES[target_class]}"
    )

    print("=" * 70)

    for index in selected_samples[target_class]:

        analyze_sample(
            index,
            target_class
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Generated visualizations:")
print(
    f"  {OUTPUT_DIR}"
)

print()
print("Classes analyzed:")
print("  Class 2 - Minor Damage")
print("  Class 3 - Major Damage")
print("  Class 4 - Destroyed")

print()
print("Next step:")
print(
    "Inspect the generated images before changing "
    "the model, loss function, or resolution."
)

print()