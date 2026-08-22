import sys
from pathlib import Path

sys.path.insert(0, "ai")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATASET_ROOT = Path(r"D:\Projects\Datasets\xBD")
SAMPLE_ID = "guatemala-volcano_00000001"

image_dir = DATASET_ROOT / "train" / "images"
target_dir = DATASET_ROOT / "train" / "targets"

pre_path = image_dir / f"{SAMPLE_ID}_pre_disaster.png"
post_path = image_dir / f"{SAMPLE_ID}_post_disaster.png"
target_path = target_dir / f"{SAMPLE_ID}_post_disaster_target.png"

pre = np.array(Image.open(pre_path).convert("RGB"))
post = np.array(Image.open(post_path).convert("RGB"))
target = np.array(Image.open(target_path))

print("Pre image :", pre.shape)
print("Post image:", post.shape)
print("Target    :", target.shape)
print("Classes   :", np.unique(target))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(pre)
axes[0].set_title("Pre-disaster")
axes[0].axis("off")

axes[1].imshow(post)
axes[1].set_title("Post-disaster")
axes[1].axis("off")

axes[2].imshow(target)
axes[2].set_title("Damage Target")
axes[2].axis("off")

plt.tight_layout()

output = Path("ai") / "sample_alignment.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", output.resolve())