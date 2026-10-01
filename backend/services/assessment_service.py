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

from backend.services.prediction_mask_service import (
    prediction_mask_to_png
)

from backend.services.storage_service import (
    upload_prediction_mask,
    upload_assessment_image,
    delete_assessment_image
)

from backend.services.auth_service import (
    ensure_user_profile
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
# STORAGE CLEANUP
# ============================================================

def _discard_uploaded_objects(
    storage_paths: list
) -> None:
    """
    Best-effort removal of objects uploaded by a pipeline
    run that then failed.

    Each path is removed independently so one failure cannot
    strand the rest, and no cleanup error is allowed to mask
    the original failure that triggered the rollback.
    """

    for storage_path in storage_paths:
        if not storage_path:
            continue

        try:
            delete_assessment_image(storage_path)
        except Exception:
            # Reported in the report as a known limitation:
            # a Storage outage during rollback can leave an
            # orphan. Surfacing this would replace the real
            # cause of the failure with a secondary error.
            pass


# ============================================================
# ASSESSMENT PIPELINE
# ============================================================

def run_assessment(
    before_info: dict,
    after_info: dict,
    user_id: str,
    user_email: str | None = None,
) -> dict:
    """
    Run the full assessment pipeline on validated
    before/after image metadata.

    Pipeline:

    1. Prepare the 9-channel model input.
    2. Run Change-Aware U-Net inference.
    3. Calculate damage statistics.
    4. Persist the assessment to Supabase.
    5. Upload the before image, after image and prediction
       mask to the private assessments bucket.
    6. Record every stored path in the database.
    7. Build the assessment result response.

    The three objects are uploaded before their paths are
    written, so the database never records a path for a file
    that failed to upload. If a later step fails, the objects
    already uploaded are removed again.
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

    # Create the main assessment record, owned by the
    # verified authenticated user.
    #
    # assessments.user_id references public.users.id, so the
    # profile row is ensured first. The id is derived from the
    # verified token and is never supplied by the client.
    #
    # The assessment id is required before anything can be
    # stored, because every Storage path is namespaced by it.

    ensure_user_profile(
        user_id=user_id,
        email=user_email,
    )

    assessment = create_assessment(
        user_id=user_id,
        model_id=model_record["id"],
        status="completed",
        damage_level=statistics["damage_level"],
        damage_percentage=statistics["damage_percentage"],
        total_pixels=statistics["total_pixels"],
        damage_pixels=statistics["damage_pixels"]
    )

    assessment_id = assessment["id"]

    # Every path uploaded by this run, so a later failure can
    # remove them again instead of leaving orphans behind.

    uploaded_paths: list = []

    try:
        # ----------------------------------------------------
        # STORAGE
        #
        # The bucket is private and every upload uses
        # upsert=False, so no existing object is overwritten
        # and no duplicate is created.
        # ----------------------------------------------------

        before_storage_path = upload_assessment_image(
            assessment_id=assessment_id,
            image_type="before",
            image_bytes=before_info["bytes"],
            content_type=before_info["content_type"]
        )

        uploaded_paths.append(before_storage_path)

        after_storage_path = upload_assessment_image(
            assessment_id=assessment_id,
            image_type="after",
            image_bytes=after_info["bytes"],
            content_type=after_info["content_type"]
        )

        uploaded_paths.append(after_storage_path)

        # Convert the class-ID mask into a lossless PNG
        # and store it in the private assessments bucket

        prediction_mask_png = prediction_mask_to_png(
            prediction_mask
        )

        mask_storage_path = upload_prediction_mask(
            assessment_id=assessment_id,
            mask_bytes=prediction_mask_png
        )

        uploaded_paths.append(mask_storage_path)

        # ----------------------------------------------------
        # DATABASE
        #
        # Paths are recorded only after every upload
        # succeeded.
        # ----------------------------------------------------

        # Save before image metadata

        add_assessment_image(
            assessment_id=assessment_id,
            image_type="before",
            filename=before_info["filename"],
            storage_path=before_storage_path,
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
            storage_path=after_storage_path,
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
            predicted_classes=predicted_classes,
            mask_storage_path=mask_storage_path
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

    except Exception:
        # Undo the Storage writes so a failed run does not
        # leave unreferenced objects in the bucket.

        _discard_uploaded_objects(uploaded_paths)

        raise

    # --------------------------------------------------------
    # RESPONSE
    #
    # Storage paths stay out of this response. The client
    # requests signed links separately through
    # GET /api/assessment/{assessment_id}/assets, so no
    # Week 4 response field changes.
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