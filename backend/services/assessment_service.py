from backend.services.preprocessing import (
    prepare_model_input
)

from backend.services.model_service import (
    damage_model
)


# ============================================================
# IMAGE METADATA HELPER
# ============================================================

def _image_metadata(info: dict) -> dict:
    """
    Strip the raw image bytes from validated
    image metadata for public responses.
    """

    return {
        "filename": info["filename"],
        "content_type": info["content_type"],
        "format": info["format"],
        "width": info["width"],
        "height": info["height"],
        "size_bytes": info["size_bytes"]
    }


# ============================================================
# ASSESSMENT PIPELINE
# ============================================================

def run_assessment(
    before_info: dict,
    after_info: dict
) -> dict:
    """
    Run the full assessment pipeline on validated
    before/after image metadata.

    Pipeline:

    1. Prepare the 9-channel model input.
    2. Run Change-Aware U-Net inference.
    3. Calculate damage statistics.
    4. Build the assessment result response.
    """

    # --------------------------------------------------------
    # PREPROCESS
    # [1, 9, 256, 256]
    # --------------------------------------------------------

    input_tensor = prepare_model_input(
        before_info["bytes"],
        after_info["bytes"]
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    prediction_mask = damage_model.predict(
        input_tensor
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    statistics = (
        damage_model.calculate_statistics(
            prediction_mask
        )
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "status": "success",
        "message": (
            "Disaster damage assessment "
            "completed successfully."
        ),
        "before_image": _image_metadata(
            before_info
        ),
        "after_image": _image_metadata(
            after_info
        ),
        "prediction": {
            "mask_shape": list(
                prediction_mask.shape
            ),
            "predicted_classes": sorted(
                set(
                    prediction_mask
                    .flatten()
                    .tolist()
                )
            )
        },
        "statistics": statistics,
        "model": {
            "name": "Change-Aware U-Net",
            "input_channels": 9,
            "output_classes": 5,
            "checkpoint": "best_model_change_aware.pth"
        }
    }