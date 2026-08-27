import sys
from pathlib import Path

sys.path.insert(0, "ai")

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import XBDDataset
from unet import UNet


# ======================================
# Configuration
# ======================================

DATASET_ROOT = r"D:\Projects\Datasets\xBD"
VAL_CSV = r"D:\Projects\Datasets\xBD\splits\val.csv"

IMAGE_SIZE = 256
BATCH_SIZE = 2

MODEL_PATH = r"ai\checkpoints\best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================
# Setup
# ======================================

print("======================================")
print("xBD U-NET MODEL TEST")
print("======================================")

print("Device:", DEVICE)
print("Model:", MODEL_PATH)


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

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ======================================
# Test Prediction
# ======================================

with torch.no_grad():

    images, targets = next(iter(loader))

    images = images.to(DEVICE)
    targets = targets.to(DEVICE)

    outputs = model(images)

    predictions = torch.argmax(
        outputs,
        dim=1,
    )


# ======================================
# Results
# ======================================

print()
print("======================================")
print("PREDICTION RESULTS")
print("======================================")

print("Input shape :", images.shape)
print("Target shape:", targets.shape)
print("Output shape:", outputs.shape)
print("Prediction shape:", predictions.shape)

print()
print("Target classes:")

for i in range(len(targets)):
    classes = torch.unique(targets[i]).cpu().tolist()
    print(f"Sample {i + 1}:", classes)

print()
print("Predicted classes:")

for i in range(len(predictions)):
    classes = torch.unique(predictions[i]).cpu().tolist()
    print(f"Sample {i + 1}:", classes)

print()
print("======================================")
print("MODEL TEST COMPLETE")
print("======================================")