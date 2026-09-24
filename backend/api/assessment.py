import io
from backend.models.assessment import AssessmentResponse
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

from backend.services.assessment_service import (
    run_assessment,
    _image_metadata
)


router = APIRouter(
    prefix="/api/assessment",
    tags=["Assessment"]
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg"
}

ALLOWED_FORMATS = {
    "PNG",
    "JPEG"
}

MIN_IMAGE_WIDTH = 256
MIN_IMAGE_HEIGHT = 256


# ============================================================
# IMAGE VALIDATION
# ============================================================

async def validate_image(
    uploaded_file: UploadFile,
    image_name: str
):
    """
    Validate an uploaded disaster-assessment image.

    Checks:
    - File exists
    - MIME type
    - File size
    - Actual image validity
    - Image format
    - Image dimensions
    """

    if uploaded_file is None:
        raise HTTPException(
            status_code=400,
            detail=f"{image_name} is required."
        )

    # --------------------------------------------------------
    # MIME TYPE
    # --------------------------------------------------------

    if uploaded_file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{image_name} must be PNG or JPEG. "
                f"Received: {uploaded_file.content_type}"
            )
        )

    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    file_bytes = await uploaded_file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"{image_name} is empty."
        )

    # --------------------------------------------------------
    # FILE SIZE
    # --------------------------------------------------------

    file_size = len(file_bytes)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{image_name} is too large. "
                f"Maximum size is 10 MB."
            )
        )

    # --------------------------------------------------------
    # VERIFY ACTUAL IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(file_bytes)
        )

        image.verify()

        # Reopen after verify()
        image = Image.open(
            io.BytesIO(file_bytes)
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"{image_name} is not a valid image."
            )
        )

    # --------------------------------------------------------
    # FORMAT
    # --------------------------------------------------------

    image_format = image.format

    if image_format not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{image_name} must be PNG or JPEG. "
                f"Detected format: {image_format}"
            )
        )

    # --------------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------------

    width, height = image.size

    if (
        width < MIN_IMAGE_WIDTH
        or height < MIN_IMAGE_HEIGHT
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{image_name} is too small. "
                f"Minimum dimensions are "
                f"{MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}. "
                f"Received: {width}x{height}."
            )
        )

    # --------------------------------------------------------
    # RETURN METADATA
    # --------------------------------------------------------

    return {
        "filename": uploaded_file.filename,
        "content_type": uploaded_file.content_type,
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": file_size,
        "bytes": file_bytes
    }


# ============================================================
# UPLOAD ENDPOINT
# ============================================================

@router.post("/upload")
async def upload_assessment_images(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...)
):

    # --------------------------------------------------------
    # VALIDATE BEFORE IMAGE
    # --------------------------------------------------------

    before_info = await validate_image(
        before_image,
        "Before image"
    )

    # --------------------------------------------------------
    # VALIDATE AFTER IMAGE
    # --------------------------------------------------------

    after_info = await validate_image(
        after_image,
        "After image"
    )

    # --------------------------------------------------------
    # MATCH DIMENSIONS
    # --------------------------------------------------------

    if (
        before_info["width"]
        != after_info["width"]
        or
        before_info["height"]
        != after_info["height"]
    ):

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Before and after images "
                    "must have matching dimensions."
                ),
                "before_dimensions": (
                    f"{before_info['width']}x"
                    f"{before_info['height']}"
                ),
                "after_dimensions": (
                    f"{after_info['width']}x"
                    f"{after_info['height']}"
                )
            }
        )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return {
        "status": "success",
        "message": (
            "Before and after images "
            "validated successfully."
        ),
        "before_image": _image_metadata(
            before_info
        ),
        "after_image": _image_metadata(
            after_info
        ),
        "dimensions": {
            "width": before_info["width"],
            "height": before_info["height"]
        }
    }


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@router.post(
    "/analyze",
    response_model=AssessmentResponse
)
async def analyze_assessment_images(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...)
):

    # --------------------------------------------------------
    # VALIDATE BEFORE IMAGE
    # --------------------------------------------------------

    before_info = await validate_image(
        before_image,
        "Before image"
    )

    # --------------------------------------------------------
    # VALIDATE AFTER IMAGE
    # --------------------------------------------------------

    after_info = await validate_image(
        after_image,
        "After image"
    )

    # --------------------------------------------------------
    # MATCH DIMENSIONS
    # --------------------------------------------------------

    if (
        before_info["width"]
        != after_info["width"]
        or
        before_info["height"]
        != after_info["height"]
    ):

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Before and after images "
                    "must have matching dimensions."
                ),
                "before_dimensions": (
                    f"{before_info['width']}x"
                    f"{before_info['height']}"
                ),
                "after_dimensions": (
                    f"{after_info['width']}x"
                    f"{after_info['height']}"
                )
            }
        )

    # --------------------------------------------------------
    # RUN ASSESSMENT SERVICE
    # --------------------------------------------------------

    return run_assessment(
        before_info,
        after_info
    )