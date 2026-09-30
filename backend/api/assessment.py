import io
from backend.models.assessment import (
    AssessmentResponse,
    AssessmentHistoryResponse,
    AssessmentDetail,
    AssessmentAssetsResponse
)
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from PIL import Image, UnidentifiedImageError

from backend.services.assessment_service import (
    run_assessment,
    _image_metadata
)

from backend.services.assessment_repository import (
    get_assessment_history,
    get_assessment_by_id
)

from backend.services.asset_service import (
    get_assessment_assets,
    resolve_auth_mode,
    AssetAssessmentNotFound,
    AssetAccessDenied,
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


# ============================================================
# RETRIEVAL ENDPOINTS
#
# GET /history is declared before GET /{assessment_id} so the
# literal "history" segment is matched first instead of being
# captured by the assessment_id path parameter.
# ============================================================

# ============================================================
# HISTORY ENDPOINT
# ============================================================

@router.get(
    "/history",
    response_model=AssessmentHistoryResponse
)
async def get_assessment_history_endpoint(
    limit: int | None = Query(
        default=None,
        ge=1,
        le=100,
        description=(
            "Maximum number of assessments to "
            "return, newest first."
        )
    )
):
    """
    Return assessment history, newest first.

    Authentication does not exist yet, and every current
    assessment row has user_id = NULL, so this always reads
    the unowned assessments. No user_id is accepted from
    the client, because at this stage that would not be a
    real access control mechanism.
    """

    assessments = get_assessment_history(
        user_id=None,
        limit=limit
    )

    return {
        "status": "success",
        "count": len(assessments),
        "limit": limit,
        "assessments": assessments
    }


# ============================================================
# SINGLE ASSESSMENT ENDPOINT
# ============================================================

@router.get(
    "/{assessment_id}",
    response_model=AssessmentDetail
)
async def get_assessment_detail_endpoint(
    assessment_id: str
):
    """
    Return one complete assessment with its model,
    before and after image metadata, prediction metadata
    and all five class statistics.

    The id is taken as a plain string on purpose: a
    malformed UUID is reported as 404 by the repository
    rather than rejected as 422 by path validation.
    """

    assessment = get_assessment_by_id(
        assessment_id
    )

    if assessment is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    "Assessment was not found."
                ),
                "assessment_id": (
                    assessment_id
                )
            }
        )

    return assessment


# ============================================================
# ASSET ENDPOINT
#
# This route has two path segments, so it can never be
# captured by GET /{assessment_id} above.
# ============================================================

@router.get(
    "/{assessment_id}/assets",
    response_model=AssessmentAssetsResponse
)
async def get_assessment_assets_endpoint(
    assessment_id: str
):
    """
    Return short-lived signed URLs for one assessment's
    private assets: before image, after image, prediction
    mask and PDF report.

    The `assessments` bucket stays private. FastAPI reads the
    storage paths from the database and signs them, so the
    client can never ask for an arbitrary object and never
    receives a permanent public URL. Supabase credentials
    are not exposed.

    An asset that was never stored, or whose object is gone,
    is returned as null.

    Authorization status: authentication is not connected to
    the backend yet, so this endpoint runs in development mode
    and may only read assessments that have no owner
    (user_id IS NULL), the same scope GET /history uses. Once
    Supabase Auth is connected, the verified user id is passed
    in instead and the endpoint returns 403 for an assessment
    owned by a different user. The asset code is unchanged by
    that switch.
    """

    # No authenticated caller is available yet. This is the
    # single seam where a verified Supabase user id replaces
    # the development-mode value.

    request_user_id = None

    try:
        return get_assessment_assets(
            assessment_id=assessment_id,
            request_user_id=request_user_id
        )

    except AssetAssessmentNotFound:
        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    "Assessment was not found."
                ),
                "assessment_id": (
                    assessment_id
                )
            }
        )

    except AssetAccessDenied:
        raise HTTPException(
            status_code=403,
            detail={
                "message": (
                    "You do not have access to "
                    "this assessment."
                ),
                "assessment_id": (
                    assessment_id
                ),
                "auth_mode": resolve_auth_mode(
                    request_user_id
                )
            }
        )