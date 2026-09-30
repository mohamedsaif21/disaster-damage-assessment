from typing import Any, Dict, List

from backend.services.supabase_service import get_supabase


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