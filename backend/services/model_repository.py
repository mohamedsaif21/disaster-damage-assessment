from typing import Optional, Dict, Any

from backend.services.supabase_service import get_supabase


def get_model_by_checkpoint(checkpoint: str) -> Optional[Dict[str, Any]]:
    """
    Find an assessment model by its checkpoint name.
    """

    supabase = get_supabase()

    response = (
        supabase
        .table("assessment_models")
        .select("*")
        .eq("checkpoint", checkpoint)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]


def create_model(
    name: str,
    architecture: str,
    checkpoint: str,
    input_channels: int,
    output_classes: int,
    epoch: Optional[int] = None,
    validation_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create a new AI model record.
    """

    supabase = get_supabase()

    payload = {
        "name": name,
        "architecture": architecture,
        "checkpoint": checkpoint,
        "input_channels": input_channels,
        "output_classes": output_classes,
        "epoch": epoch,
        "validation_loss": validation_loss,
    }

    response = (
        supabase
        .table("assessment_models")
        .insert(payload)
        .execute()
    )

    if not response.data:
        raise RuntimeError("Failed to create assessment model.")

    return response.data[0]