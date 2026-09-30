import io

import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MASK_WIDTH = 256

MASK_HEIGHT = 256

MIN_CLASS_ID = 0

MAX_CLASS_ID = 4

MASK_CONTENT_TYPE = "image/png"


# ============================================================
# PREDICTION MASK -> PNG
# ============================================================

def prediction_mask_to_png(mask) -> bytes:
    """
    Convert a 2D class-ID prediction mask into
    lossless PNG bytes.

    The mask is written as an 8-bit grayscale
    ("L") image where each pixel holds its class
    ID directly: 0 background, 1 no_damage,
    2 minor_damage, 3 major_damage, 4 destroyed.

    Values are deliberately not rescaled, so
    decoding the PNG reproduces the original
    class IDs exactly. PNG is lossless, so no
    information is lost.

    No temporary file is written to disk and the
    image data is never logged.
    """

    array = np.asarray(mask)

    # --------------------------------------------------------
    # SHAPE
    # --------------------------------------------------------

    if array.ndim != 2:
        raise ValueError(
            "Prediction mask must be 2D. "
            f"Received {array.ndim}D with shape "
            f"{array.shape}"
        )

    height, width = array.shape

    if (
        width != MASK_WIDTH
        or height != MASK_HEIGHT
    ):
        raise ValueError(
            f"Prediction mask must be "
            f"{MASK_WIDTH}x{MASK_HEIGHT}. "
            f"Received {width}x{height}"
        )

    # --------------------------------------------------------
    # DTYPE
    # --------------------------------------------------------

    if not np.issubdtype(
        array.dtype, np.integer
    ):
        raise ValueError(
            "Prediction mask must use an integer "
            "dtype. "
            f"Received {array.dtype}"
        )

    # --------------------------------------------------------
    # CLASS IDS
    # --------------------------------------------------------

    minimum = int(array.min())

    maximum = int(array.max())

    if (
        minimum < MIN_CLASS_ID
        or maximum > MAX_CLASS_ID
    ):
        raise ValueError(
            "Prediction mask must contain only "
            f"class IDs {MIN_CLASS_ID}-{MAX_CLASS_ID}. "
            f"Received range {minimum}-{maximum}"
        )

    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

    image = Image.fromarray(
        array.astype(np.uint8),
        mode="L"
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True
    )

    return buffer.getvalue()
