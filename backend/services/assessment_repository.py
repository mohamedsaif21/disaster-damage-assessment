import uuid
from typing import Any, Dict, List, Optional

from backend.services.supabase_service import get_supabase


# ============================================================
# RETRIEVAL
# ============================================================

HISTORY_COLUMNS = (
    "id, "
    "user_id, "
    "model_id, "
    "status, "
    "damage_level, "
    "damage_percentage, "
    "total_pixels, "
    "damage_pixels, "
    "created_at, "
    "updated_at"
)

HISTORY_MODEL_EMBED = (
    "assessment_models("
    "id, "
    "name, "
    "architecture, "
    "checkpoint, "
    "input_channels, "
    "output_classes, "
    "epoch, "
    "validation_loss"
    ")"
)

DETAIL_COLUMNS = f"*, {HISTORY_MODEL_EMBED}"

DETAIL_EMBEDS = (
    ", "
    "assessment_images(*), "
    "assessment_predictions(*), "
    "assessment_class_statistics(*)"
)


def get_assessment_history(
    user_id: str | None,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Return the assessments belonging to a user,
    newest first, with the related model included.

    A null user_id selects assessments that were
    created without an owner. An empty result is
    returned as an empty list.
    """

    supabase = get_supabase()

    query = supabase.table("assessments").select(
        f"{HISTORY_COLUMNS}, {HISTORY_MODEL_EMBED}"
    )

    if user_id is None:
        query = query.is_("user_id", "null")
    else:
        query = query.eq("user_id", user_id)

    query = query.order("created_at", desc=True)

    if limit is not None:
        query = query.limit(limit)

    response = query.execute()

    return response.data or []


def get_assessment_by_id(
    assessment_id: str
) -> Optional[Dict[str, Any]]:
    """
    Return one complete assessment with its model,
    before/after image metadata, prediction metadata
    and all five class statistics.

    Returns None when the id is not a valid UUID or
    when no assessment matches.

    Embedded child rows are ordered here because
    PostgREST cannot sort an embedded resource.
    """

    try:
        uuid.UUID(str(assessment_id))
    except (ValueError, AttributeError, TypeError):
        return None

    supabase = get_supabase()

    response = (
        supabase
        .table("assessments")
        .select(f"{DETAIL_COLUMNS}{DETAIL_EMBEDS}")
        .eq("id", assessment_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    assessment = response.data[0]

    assessment["assessment_images"] = sorted(
        assessment.get("assessment_images") or [],
        key=lambda image: image["image_type"]
    )

    assessment["assessment_class_statistics"] = sorted(
        assessment.get("assessment_class_statistics") or [],
        key=lambda statistic: statistic["class_id"]
    )

    return assessment


# ============================================================
# PERSISTENCE
# ============================================================

def create_assessment(
    user_id: str | None,
    model_id: str,
    status: str,
    damage_level: str | None,
    damage_percentage: float | None,
    total_pixels: int | None,
    damage_pixels: int | None,
) -> Dict[str, Any]:
    """
    Create the main assessment record.
    """

    supabase = get_supabase()

    payload = {
        "user_id": user_id,
        "model_id": model_id,
        "status": status,
        "damage_level": damage_level,
        "damage_percentage": damage_percentage,
        "total_pixels": total_pixels,
        "damage_pixels": damage_pixels,
    }

    response = (
        supabase
        .table("assessments")
        .insert(payload)
        .execute()
    )

    if not response.data:
        raise RuntimeError("Failed to create assessment.")

    return response.data[0]


def add_assessment_image(
    assessment_id: str,
    image_type: str,
    filename: str,
    storage_path: str | None,
    content_type: str,
    image_format: str,
    width: int,
    height: int,
    size_bytes: int,
) -> Dict[str, Any]:
    """
    Store metadata for a before/after assessment image.
    """

    supabase = get_supabase()

    payload = {
        "assessment_id": assessment_id,
        "image_type": image_type,
        "filename": filename,
        "storage_path": storage_path,
        "content_type": content_type,
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": size_bytes,
    }

    response = (
        supabase
        .table("assessment_images")
        .insert(payload)
        .execute()
    )

    if not response.data:
        raise RuntimeError("Failed to create assessment image record.")

    return response.data[0]


def add_prediction(
    assessment_id: str,
    mask_width: int,
    mask_height: int,
    predicted_classes: List[int],
    mask_storage_path: str | None = None,
) -> Dict[str, Any]:
    """
    Store prediction metadata for an assessment.
    """

    supabase = get_supabase()

    payload = {
        "assessment_id": assessment_id,
        "mask_width": mask_width,
        "mask_height": mask_height,
        "predicted_classes": predicted_classes,
        "mask_storage_path": mask_storage_path,
    }

    response = (
        supabase
        .table("assessment_predictions")
        .insert(payload)
        .execute()
    )

    if not response.data:
        raise RuntimeError("Failed to create assessment prediction.")

    return response.data[0]


def add_class_statistics(
    assessment_id: str,
    statistics: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Store the five damage-class statistics.

    Expected structure:

    {
        0: {
            "class_name": "background",
            "pixels": 100,
            "percentage": 50.0
        },
        ...
    }
    """

    supabase = get_supabase()

    rows = []

    for class_id, class_data in statistics.items():
        rows.append(
            {
                "assessment_id": assessment_id,
                "class_id": class_id,
                "class_name": class_data["class_name"],
                "pixel_count": class_data["pixels"],
                "percentage": class_data["percentage"],
            }
        )

    response = (
        supabase
        .table("assessment_class_statistics")
        .insert(rows)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Failed to create assessment class statistics."
        )

    return response.data


def set_report_storage_path(
    assessment_id: str,
    report_storage_path: str
) -> Dict[str, Any]:
    """
    Store the private storage path of the generated
    PDF report on the assessment row.

    Returns the updated assessment record.
    """

    if not report_storage_path or not isinstance(
        report_storage_path, str
    ):
        raise ValueError(
            "report_storage_path must be a non-empty "
            "string."
        )

    supabase = get_supabase()

    response = (
        supabase
        .table("assessments")
        .update(
            {"report_storage_path": report_storage_path}
        )
        .eq("id", assessment_id)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Failed to store the report storage path."
        )

    return response.data[0]