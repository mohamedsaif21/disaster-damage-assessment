import io
import os
import tempfile

import numpy as np
from PIL import Image

from backend.services.prediction_mask_service import (
    prediction_mask_to_png,
    MASK_WIDTH,
    MASK_HEIGHT,
    MIN_CLASS_ID,
    MAX_CLASS_ID
)


# ============================================================
# HELPERS
# ============================================================

def full_class_mask() -> np.ndarray:
    """
    Build a 256x256 mask that contains every
    class ID from 0 to 4.
    """

    mask = np.zeros(
        (MASK_HEIGHT, MASK_WIDTH),
        dtype=np.int64
    )

    band = MASK_HEIGHT // 5

    for class_id in range(
        MIN_CLASS_ID, MAX_CLASS_ID + 1
    ):
        top = class_id * band
        bottom = (
            top + band
            if class_id < MAX_CLASS_ID
            else MASK_HEIGHT
        )
        mask[top:bottom, :] = class_id

    return mask


def expect_raises(label, function, failures):
    """
    Assert that a call raises ValueError.
    """

    try:
        function()
    except ValueError as error:
        print(f"  [PASS] {label}")
        print(f"         {error}")
        return
    except Exception as error:
        print(
            f"  [FAIL] {label} "
            f"(raised {type(error).__name__})"
        )
        failures.append(label)
        return

    print(f"  [FAIL] {label} (no error raised)")
    failures.append(label)


# ============================================================
# TEST
# ============================================================

def main():
    failures = []

    def check(label, condition, detail=""):
        status = "PASS" if condition else "FAIL"

        print(f"  [{status}] {label}")

        if detail:
            print(f"         {detail}")

        if not condition:
            failures.append(label)

    print("=" * 66)
    print("STEP 7.11 PREDICTION MASK SERVICE TEST")
    print("=" * 66)
    print()

    # ---------------------------------------------------------
    # 1. Valid conversion
    # ---------------------------------------------------------

    print("[1] Valid 256x256 mask containing classes 0-4")

    mask = full_class_mask()

    png_bytes = prediction_mask_to_png(mask)

    print(f"     mask shape : {mask.shape}")
    print(f"     unique ids : {sorted(set(mask.flatten().tolist()))}")
    print(f"     png bytes  : {len(png_bytes)}")
    print()

    check(
        "conversion returns bytes",
        isinstance(png_bytes, bytes)
    )
    check(
        "png is non-empty",
        len(png_bytes) > 0
    )

    # ---------------------------------------------------------
    # 2. Valid PNG signature
    # ---------------------------------------------------------

    print("[2] Returned bytes are a valid PNG")

    check(
        "PNG magic signature present",
        png_bytes[:8] == b"\x89PNG\r\n\x1a\n",
        repr(png_bytes[:8])
    )
    check(
        "IEND chunk present",
        png_bytes[-12:-4] == b"\x00\x00\x00\x00IEND"
    )

    # ---------------------------------------------------------
    # 3. Dimensions
    # ---------------------------------------------------------

    print("[3] PNG dimensions are 256x256")

    with Image.open(io.BytesIO(png_bytes)) as image:
        width, height = image.size
        mode = image.mode

    print(f"     size : {width}x{height}")
    print(f"     mode : {mode}")
    print()

    check(
        "width is 256",
        width == MASK_WIDTH
    )
    check(
        "height is 256",
        height == MASK_HEIGHT
    )
    check(
        "mode is 8-bit grayscale (L)",
        mode == "L",
        f"mode={mode}"
    )

    # ---------------------------------------------------------
    # 4. Exact class ID round trip
    # ---------------------------------------------------------

    print("[4] Decoding reproduces the exact class IDs")

    with Image.open(io.BytesIO(png_bytes)) as image:
        decoded = np.array(image)

    print(f"     decoded unique ids : {sorted(set(decoded.flatten().tolist()))}")
    print()

    check(
        "decoded array has same shape",
        decoded.shape == mask.shape
    )
    check(
        "decoded values are exactly the original class IDs",
        np.array_equal(decoded, mask.astype(np.uint8))
    )
    check(
        "decoded contains only IDs 0-4",
        set(decoded.flatten().tolist()).issubset(
            {0, 1, 2, 3, 4}
        )
    )

    # ---------------------------------------------------------
    # 5. No lossy format
    # ---------------------------------------------------------

    print("[5] Encoding is lossless, not a lossy JPEG")

    check(
        "stored as PNG not JPEG",
        png_bytes[:3] != b"\xff\xd8\xff"
    )

    # ---------------------------------------------------------
    # 6. No rescaling
    # ---------------------------------------------------------

    print("[6] Class IDs are not rescaled")

    single = np.full(
        (MASK_HEIGHT, MASK_WIDTH), 4, dtype=np.int64
    )

    single_png = prediction_mask_to_png(single)

    with Image.open(io.BytesIO(single_png)) as image:
        single_decoded = np.array(image)

    check(
        "class 4 stays 4 (not 255)",
        int(single_decoded.max()) == 4,
        f"max={int(single_decoded.max())}"
    )
    check(
        "uniform class-4 mask round-trips exactly",
        np.array_equal(single_decoded, single.astype(np.uint8))
    )

    # ---------------------------------------------------------
    # 7. Validation
    # ---------------------------------------------------------

    print("[7] Invalid input is rejected")

    expect_raises(
        "rejects 1D mask",
        lambda: prediction_mask_to_png(
            np.zeros(256, dtype=np.int64)
        ),
        failures
    )

    expect_raises(
        "rejects 3D mask",
        lambda: prediction_mask_to_png(
            np.zeros((1, 256, 256), dtype=np.int64)
        ),
        failures
    )

    expect_raises(
        "rejects wrong dimensions (128x128)",
        lambda: prediction_mask_to_png(
            np.zeros((128, 128), dtype=np.int64)
        ),
        failures
    )

    expect_raises(
        "rejects wrong dimensions (256x512)",
        lambda: prediction_mask_to_png(
            np.zeros((256, 512), dtype=np.int64)
        ),
        failures
    )

    high = np.zeros(
        (MASK_HEIGHT, MASK_WIDTH), dtype=np.int64
    )
    high[0, 0] = 5

    expect_raises(
        "rejects class ID 5",
        lambda: prediction_mask_to_png(high),
        failures
    )

    negative = np.zeros(
        (MASK_HEIGHT, MASK_WIDTH), dtype=np.int64
    )
    negative[0, 0] = -1

    expect_raises(
        "rejects class ID -1",
        lambda: prediction_mask_to_png(negative),
        failures
    )

    expect_raises(
        "rejects float dtype",
        lambda: prediction_mask_to_png(
            np.zeros(
                (MASK_HEIGHT, MASK_WIDTH),
                dtype=np.float32
            )
        ),
        failures
    )

    # ---------------------------------------------------------
    # 8. No temporary file
    # ---------------------------------------------------------

    print("[8] No temporary file is required")

    before_files = set(os.listdir(tempfile.gettempdir()))

    prediction_mask_to_png(mask)

    after_files = set(os.listdir(tempfile.gettempdir()))

    new_files = after_files - before_files

    check(
        "no new files in the temp directory",
        len(new_files) == 0,
        f"new: {sorted(new_files)[:5]}"
    )

    print()
    print("=" * 66)

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed")

        for item in failures:
            print(f"  - {item}")

        print("=" * 66)
        raise SystemExit(1)

    print("PREDICTION MASK SERVICE TEST PASSED")
    print("=" * 66)


if __name__ == "__main__":
    main()
