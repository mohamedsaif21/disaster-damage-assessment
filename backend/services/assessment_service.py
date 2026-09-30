import os

from backend.services.preprocessing import (
    prepare_model_input
)

from backend.services.model_service import (
    damage_model,
    CHECKPOINT_PATH,
    CLASS_NAMES
)

from backend.services.model_repository import (
    get_model_by_checkpoint
)

from backend.services.assessment_repository import (
    create_assessment,
    add_assessment_image,
    add_prediction,
    add_class_statistics
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = os.path.basename(
    CHECKPOINT_PATH
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
    4. Persist the assessment to Supabase.
    5. Build the assessment result response.
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
    # PERSISTENCE
    # --------------------------------------------------------

    # Look up the registered model by checkpoint

    model_record = get_model_by_checkpoint(
        CHECKPOINT
    )

    if not model_record:
        raise RuntimeError(
            "No registered assessment model was found "
            f"for checkpoint: {CHECKPOINT}"
        )

    # Create the main assessment record

    assessment = create_assessment(
        user_id=None,
        model_id=model_record["id"],
        status="completed",
        damage_level=statistics["damage_level"],
        damage_percentage=statistics["damage_percentage"],
        total_pixels=statistics["total_pixels"],
        damage_pixels=statistics["damage_pixels"]
    )

    assessment_id = assessment["id"]

    # Save before image metadata

    add_assessment_image(
        assessment_id=assessment_id,
        image_type="before",
        filename=before_info["filename"],
        storage_path=None,
        content_type=before_info["content_type"],
        image_format=before_info["format"],
        width=before_info["width"],
        height=before_info["height"],
        size_bytes=before_info["size_bytes"]
    )

    # Save after image metadata

    add_assessment_image(
        assessment_id=assessment_id,
        image_type="after",
        filename=after_info["filename"],
        storage_path=None,
        content_type=after_info["content_type"],
        image_format=after_info["format"],
        width=after_info["width"],
        height=after_info["height"],
        size_bytes=after_info["size_bytes"]
    )

    # Save prediction metadata

    mask_height, mask_width = (
        prediction_mask.shape
    )

    predicted_classes = sorted(
        set(
            prediction_mask
            .flatten()
            .tolist()
        )
    )

    add_prediction(
        assessment_id=assessment_id,
        mask_width=mask_width,
        mask_height=mask_height,
        predicted_classes=predicted_classes
    )

    # Save the five damage-class statistics

    add_class_statistics(
        assessment_id=assessment_id,
        statistics={
            class_id: {
                "class_name": class_name,
                "pixels": statistics[
                    "classes"
                ][class_name]["pixels"],
                "percentage": statistics[
                    "classes"
                ][class_name]["percentage"]
            }
            for class_id, class_name
            in CLASS_NAMES.items()
        }
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
            "predicted_classes": predicted_classes
        },
        "statistics": statistics,
        "model": {
            "name": "Change-Aware U-Net",
            "input_channels": 9,
            "output_classes": 5,
            "checkpoint": CHECKPOINT
        }
    }