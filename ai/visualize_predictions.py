import sys
from pathlib import Path

sys.path.insert(0, "ai")

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import XBDDataset
from unet import UNet


# ======================================
# Configuration
# ======================================

DATASET_ROOT = r"D:\Projects\Datasets\xBD"

VAL_CSV = r"D:\Projects\Datasets\xBD\splits\val.csv"

MODEL_PATH = r"ai\checkpoints\best_model.pth"

OUTPUT_DIR = Path("outputs/predictions")

IMAGE_SIZE = 256
BATCH_SIZE = 1

NUM_SAMPLES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================
# Setup
# ======================================

print("======================================")
print("xBD U-NET PREDICTION VISUALIZATION")
print("======================================")

print("Device:", DEVICE)
print("Model:", MODEL_PATH)
print("Validation CSV:", VAL_CSV)


# ======================================
# Create output directory
# ======================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================
# Dataset
# ======================================

dataset = XBDDataset(
    DATASET_ROOT,
    VAL_CSV,
    image_size=IMAGE_SIZE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Validation samples:", len(dataset))


# ======================================
# Load Model
# ======================================

model = UNet(
    in_channels=6,
    num_classes=5,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ======================================
# Class names
# ======================================

class_names = [
    "Background",
    "No Damage",
    "Minor Damage",
    "Major Damage",
    "Destroyed",
]


# ======================================
# Prediction
# ======================================

print()
print("======================================")
print("GENERATING PREDICTIONS")
print("======================================")

with torch.no_grad():

    for sample_index, (images, targets) in enumerate(loader):

        if sample_index >= NUM_SAMPLES:
            break

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # Model prediction
        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # ==================================
        # Convert tensors to NumPy
        # ==================================

        image = images[0].cpu().numpy()

        target = targets[0].cpu().numpy()

        prediction = predictions[0].cpu().numpy()

        # ==================================
        # Separate RGB images
        # ==================================

        pre_image = np.transpose(
            image[:3],
            (1, 2, 0)
        )

        post_image = np.transpose(
            image[3:],
            (1, 2, 0)
        )

        # Make sure display range is valid
        pre_image = np.clip(
            pre_image,
            0,
            1
        )

        post_image = np.clip(
            post_image,
            0,
            1
        )

        # ==================================
        # Print classes
        # ==================================

        target_classes = np.unique(target).tolist()

        predicted_classes = np.unique(
            prediction
        ).tolist()

        print()
        print(
            f"Sample {sample_index + 1}/{NUM_SAMPLES}"
        )

        print(
            "Actual classes:",
            [
                class_names[c]
                for c in target_classes
            ]
        )

        print(
            "Predicted classes:",
            [
                class_names[c]
                for c in predicted_classes
            ]
        )

        # ==================================
        # Create figure
        # ==================================

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4)
        )

        # ==================================
        # Pre-disaster
        # ==================================

        axes[0].imshow(pre_image)

        axes[0].set_title(
            "Pre-Disaster"
        )

        axes[0].axis("off")

        # ==================================
        # Post-disaster
        # ==================================

        axes[1].imshow(post_image)

        axes[1].set_title(
            "Post-Disaster"
        )

        axes[1].axis("off")

        # ==================================
        # Ground Truth
        # ==================================

        axes[2].imshow(
            target,
            vmin=0,
            vmax=4
        )

        axes[2].set_title(
            "Ground Truth"
        )

        axes[2].axis("off")

        # ==================================
        # Prediction
        # ==================================

        axes[3].imshow(
            prediction,
            vmin=0,
            vmax=4
        )

        axes[3].set_title(
            "Model Prediction"
        )

        axes[3].axis("off")

        # ==================================
        # Main title
        # ==================================

        fig.suptitle(
            f"xBD Sample {sample_index + 1}",
            fontsize=14
        )

        plt.tight_layout()

        # ==================================
        # Save
        # ==================================

        output_path = (
            OUTPUT_DIR
            / f"sample_{sample_index + 1:03d}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Saved:",
            output_path
        )


# ======================================
# Complete
# ======================================

print()
print("======================================")
print("VISUALIZATION COMPLETE")
print("======================================")

print(
    "Results saved to:",
    OUTPUT_DIR
)