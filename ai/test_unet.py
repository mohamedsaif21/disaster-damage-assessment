import sys

sys.path.insert(0, "ai")

import torch

from unet import UNet


print("======================================")
print("U-NET FORWARD PASS TEST")
print("======================================")

# Create model
model = UNet(
    in_channels=6,
    num_classes=5,
)

# Fake batch matching our DataLoader
x = torch.randn(4, 6, 256, 256)

print("Input shape :", x.shape)

# Forward pass
with torch.no_grad():
    output = model(x)

print("Output shape:", output.shape)
print("Output dtype:", output.dtype)

print("======================================")