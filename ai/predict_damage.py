import sys
from pathlib import Path

sys.path.insert(0, "ai")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from unet import UNet


# ======================================
# CONFIGURATION
# ======================================

DATASET_ROOT = Path(r"D:\Projects\Datasets\xBD")
VAL_CSV = DATASET_ROOT / "splits" / "val.csv"

MODEL_PATH = Path(r"ai\checkpoints\best_model.pth")

OUTPUT_DIR = Path("outputs/damage_predictions")

IMAGE_SIZE = 256

# Which validation sample to test
SAMPLE_INDEX = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================
# DAMAGE CLASSES
# ======================================

CLASS_NAMES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}


# ======================================
# SETUP
# ======================================

print("======================================")
print("xBD DAMAGE PREDICTION")
print("======================================")

print("Device:", DEVICE)
print("Model:", MODEL_PATH)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================
# LOAD VALIDATION CSV
# ======================================

print()
print("Loading validation data...")

samples = pd.read_csv(VAL_CSV)

print(
    "Validation samples:",
    len(samples)
)


# ======================================
# SELECT SAMPLE
# ======================================

sample_id = samples.iloc[
    SAMPLE_INDEX
]["sample_id"]


print()
print("Selected sample:", sample_id)


# ======================================
# IMAGE PATHS
# ======================================

IMAGE_DIR = (
    DATASET_ROOT /
    "train" /
    "images"
)

pre_path = (
    IMAGE_DIR /
    f"{sample_id}_pre_disaster.png"
)

post_path = (
    IMAGE_DIR /
    f"{sample_id}_post_disaster.png"
)


print()
print("Pre-disaster image:")
print(pre_path)

print()
print("Post-disaster image:")
print(post_path)


# ======================================
# CHECK FILES
# ======================================

if not pre_path.exists():

    raise FileNotFoundError(
        f"Pre-disaster image not found:\n{pre_path}"
    )


if not post_path.exists():

    raise FileNotFoundError(
        f"Post-disaster image not found:\n{post_path}"
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
# LOAD IMAGES
# ======================================

print()
print("Loading images...")

pre_image = Image.open(
    pre_path
).convert("RGB")


post_image = Image.open(
    post_path
).convert("RGB")


# ======================================
# RESIZE
# ======================================

pre_image_resized = pre_image.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.BILINEAR
)


post_image_resized = post_image.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.BILINEAR
)


# ======================================
# NUMPY CONVERSION
# ======================================

pre_array = np.array(
    pre_image_resized,
    dtype=np.float32
)


post_array = np.array(
    post_image_resized,
    dtype=np.float32
)


# ======================================
# NORMALIZATION
# ======================================

pre_array = pre_array / 255.0
post_array = post_array / 255.0


# ======================================
# HWC -> CHW
# ======================================

pre_array = np.transpose(
    pre_array,
    (2, 0, 1)
)


post_array = np.transpose(
    post_array,
    (2, 0, 1)
)


# ======================================
# COMBINE 6 CHANNELS
# ======================================

image = np.concatenate(
    [
        pre_array,
        post_array
    ],
    axis=0
)


# ======================================
# NUMPY -> TORCH
# ======================================

image = torch.from_numpy(
    image
).float()


# Add batch dimension

image = image.unsqueeze(0)

image = image.to(DEVICE)


print()
print("Input shape:", image.shape)


# ======================================
# MODEL PREDICTION
# ======================================

print()
print("Running prediction...")


with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(
        output,
        dim=1
    )


# ======================================
# REMOVE BATCH DIMENSION
# ======================================

prediction = prediction.squeeze(
    0
)


prediction = prediction.cpu().numpy()


print(
    "Prediction shape:",
    prediction.shape
)


# ======================================
# CLASS DISTRIBUTION
# ======================================

print()
print("======================================")
print("DAMAGE CLASS DISTRIBUTION")
print("======================================")


total_pixels = prediction.size

class_percentages = {}


for class_id, class_name in CLASS_NAMES.items():

    pixel_count = np.sum(
        prediction == class_id
    )

    percentage = (
        pixel_count /
        total_pixels
    ) * 100

    class_percentages[class_id] = percentage

    print(
        f"{class_name:15s}: "
        f"{pixel_count:8d} pixels "
        f"({percentage:.2f}%)"
    )


# ======================================
# DAMAGE SUMMARY
# ======================================

damage_pixels = np.sum(
    prediction >= 2
)


damage_percentage = (
    damage_pixels /
    total_pixels
) * 100


print()
print("======================================")
print("DAMAGE SUMMARY")
print("======================================")


print(
    f"Damaged pixels: "
    f"{damage_pixels}"
)


print(
    f"Damage percentage: "
    f"{damage_percentage:.2f}%"
)


# ======================================
# DAMAGE LEVEL
# ======================================

if class_percentages[4] > 5:

    damage_level = "SEVERE"

elif class_percentages[3] > 5:

    damage_level = "MAJOR"

elif class_percentages[2] > 5:

    damage_level = "MINOR"

else:

    damage_level = "LOW"


print(
    "Overall Damage Level:",
    damage_level
)


# ======================================
# SAVE RAW MASK
# ======================================

mask_path = (
    OUTPUT_DIR /
    f"{sample_id}_damage_mask.png"
)


mask_image = Image.fromarray(
    prediction.astype(np.uint8)
)


mask_image.save(
    mask_path
)


print()
print("Saved mask:")
print(mask_path)


# ======================================
# VISUALIZATION
# ======================================

print()
print("Generating visualization...")


plt.figure(
    figsize=(16, 5)
)


# --------------------------------------
# PRE-DISASTER
# --------------------------------------

plt.subplot(1, 4, 1)

plt.imshow(
    pre_image_resized
)

plt.title(
    "Pre-Disaster"
)

plt.axis("off")


# --------------------------------------
# POST-DISASTER
# --------------------------------------

plt.subplot(1, 4, 2)

plt.imshow(
    post_image_resized
)

plt.title(
    "Post-Disaster"
)

plt.axis("off")


# --------------------------------------
# PREDICTION
# --------------------------------------

plt.subplot(1, 4, 3)

plt.imshow(
    prediction,
    cmap="viridis",
    vmin=0,
    vmax=4
)

plt.title(
    "Predicted Damage"
)

plt.axis("off")


# --------------------------------------
# OVERLAY
# --------------------------------------

plt.subplot(1, 4, 4)

plt.imshow(
    post_image_resized
)

plt.imshow(
    prediction,
    cmap="viridis",
    alpha=0.45,
    vmin=0,
    vmax=4
)

plt.title(
    "Damage Overlay"
)

plt.axis("off")


# ======================================
# SAVE VISUALIZATION
# ======================================

figure_path = (
    OUTPUT_DIR /
    f"{sample_id}_damage_assessment.png"
)


plt.tight_layout()


plt.savefig(
    figure_path,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print()
print("Saved visualization:")
print(figure_path)


# ======================================
# COMPLETE
# ======================================

print()
print("======================================")
print("DAMAGE PREDICTION COMPLETE")
print("======================================")


print()
print("Sample:", sample_id)

print(
    "Overall Damage Level:",
    damage_level
)

print(
    f"Total Damage: "
    f"{damage_percentage:.2f}%"
)