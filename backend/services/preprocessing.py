import io

import numpy as np
import torch
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_IMAGE_SIZE = 256


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    image_bytes: bytes
) -> Image.Image:
    """
    Load image bytes and convert to RGB.
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    return image


# ============================================================
# IMAGE → TENSOR
# ============================================================

def image_to_tensor(
    image: Image.Image
) -> torch.Tensor:
    """
    Convert PIL RGB image to normalized
    CHW PyTorch tensor.
    """

    image = image.resize(
        (
            MODEL_IMAGE_SIZE,
            MODEL_IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array
    ).permute(
        2,
        0,
        1
    )

    return tensor


# ============================================================
# BUILD 9-CHANNEL INPUT
# ============================================================

def prepare_model_input(
    before_bytes: bytes,
    after_bytes: bytes
) -> torch.Tensor:
    """
    Convert before/after images into the
    9-channel input expected by the
    Change-Aware U-Net.

    Channels:

    0-2 : Before RGB
    3-5 : After RGB
    6-8 : Absolute difference
    """

    before_image = load_image(
        before_bytes
    )

    after_image = load_image(
        after_bytes
    )

    before = image_to_tensor(
        before_image
    )

    after = image_to_tensor(
        after_image
    )

    # Explicit visual change
    difference = torch.abs(
        after - before
    )

    # 3 + 3 + 3 = 9 channels
    combined = torch.cat(
        [
            before,
            after,
            difference
        ],
        dim=0
    )

    # Add batch dimension
    combined = combined.unsqueeze(
        dim=0
    )

    return combined


# ============================================================
# VALIDATION HELPER
# ============================================================

def get_model_input_info(
    tensor: torch.Tensor
) -> dict:

    return {
        "shape": list(
            tensor.shape
        ),
        "dtype": str(
            tensor.dtype
        ),
        "min": float(
            tensor.min().item()
        ),
        "max": float(
            tensor.max().item()
        )
    }