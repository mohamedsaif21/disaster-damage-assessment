from typing import Any, Dict

from storage3.types import FileOptions

from backend.services.supabase_service import get_supabase


# ============================================================
# CONFIGURATION
# ============================================================

STORAGE_BUCKET = "assessments"

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_IMAGE_TYPES = {
    "before",
    "after"
}

# Mirrors the content types already accepted by
# the assessment upload endpoint.

ALLOWED_CONTENT_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg"
}

PREDICTION_MASK_FILENAME = "prediction-mask.png"

PREDICTION_MASK_CONTENT_TYPE = "image/png"

REPORT_FILENAME = "report.pdf"

REPORT_CONTENT_TYPE = "application/pdf"

# The project has a small total storage budget, so a
# report that exceeds this limit is rejected instead of
# being stored.

MAX_REPORT_SIZE = 2 * 1024 * 1024  # 2 MB

PDF_MAGIC = b"%PDF-"


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _storage_bucket():
    """
    Return the private assessments bucket handle
    from the existing service-role client.
    """

    return get_supabase().storage.from_(
        STORAGE_BUCKET
    )


def _validate_image_type(image_type: str) -> None:
    if image_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(
            "image_type must be one of: "
            f"{sorted(ALLOWED_IMAGE_TYPES)}. "
            f"Received: {image_type!r}"
        )


def _validate_content_type(
    content_type: str
) -> str:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            "content_type must be one of: "
            f"{sorted(ALLOWED_CONTENT_TYPES)}. "
            f"Received: {content_type!r}"
        )

    return ALLOWED_CONTENT_TYPES[content_type]


def _validate_image_bytes(
    image_bytes: bytes
) -> None:
    if not image_bytes:
        raise ValueError(
            "image_bytes is empty."
        )

    size = len(image_bytes)

    if size > MAX_IMAGE_SIZE:
        raise ValueError(
            "Image is too large. "
            f"Maximum size is {MAX_IMAGE_SIZE} bytes. "
            f"Received: {size} bytes"
        )


def _bucket_relative_path(
    storage_path: str
) -> str:
    """
    Convert a stored path into a path that is
    relative to the assessments bucket.

    Accepts the bucket-qualified path returned by
    upload_assessment_image() as well as a path that
    is already relative to the bucket.
    """

    if not storage_path or not isinstance(
        storage_path, str
    ):
        raise ValueError(
            "storage_path must be a non-empty string."
        )

    prefix = f"{STORAGE_BUCKET}/"

    if storage_path.startswith(prefix):
        return storage_path[len(prefix):]

    return storage_path


# ============================================================
# PATH
# ============================================================

def get_assessment_image_path(
    assessment_id: str,
    image_type: str,
    content_type: str
) -> str:
    """
    Return the private storage path for one
    assessment image.

    The assessments bucket is private, so this
    returns a path only. No public or signed URL
    is generated here.
    """

    _validate_image_type(image_type)

    extension = _validate_content_type(
        content_type
    )

    if not assessment_id or not isinstance(
        assessment_id, str
    ):
        raise ValueError(
            "assessment_id must be a non-empty string."
        )

    return (
        f"{STORAGE_BUCKET}/"
        f"{assessment_id}/"
        f"{image_type}.{extension}"
    )


# ============================================================
# UPLOAD
# ============================================================

def _upload_bytes(
    storage_path: str,
    data: bytes,
    content_type: str
) -> str:
    """
    Upload raw bytes into the private assessments
    bucket at storage_path, without overwriting an
    existing object.
    """

    relative_path = _bucket_relative_path(
        storage_path
    )

    bucket = _storage_bucket()

    try:
        bucket.upload(
            relative_path,
            data,
            file_options=FileOptions(
                upsert="false",
                **{
                    "content-type": content_type
                }
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to upload assessment image to "
            f"{storage_path}: {error}"
        ) from error

    return storage_path


def upload_assessment_image(
    assessment_id: str,
    image_type: str,
    image_bytes: bytes,
    content_type: str
) -> str:
    """
    Upload one before/after assessment image to the
    private assessments bucket and return its
    storage path.

    The original extension is derived from the
    content type. The upload uses upsert=False, so
    an existing before/after file is never
    overwritten.
    """

    storage_path = get_assessment_image_path(
        assessment_id,
        image_type,
        content_type
    )

    _validate_image_bytes(image_bytes)

    return _upload_bytes(
        storage_path,
        image_bytes,
        content_type
    )


# ============================================================
# PREDICTION MASK
# ============================================================

def get_prediction_mask_path(
    assessment_id: str
) -> str:
    """
    Return the private storage path for an
    assessment prediction mask.
    """

    if not assessment_id or not isinstance(
        assessment_id, str
    ):
        raise ValueError(
            "assessment_id must be a non-empty string."
        )

    return (
        f"{STORAGE_BUCKET}/"
        f"{assessment_id}/"
        f"{PREDICTION_MASK_FILENAME}"
    )


def upload_prediction_mask(
    assessment_id: str,
    mask_bytes: bytes
) -> str:
    """
    Upload a lossless prediction-mask PNG to the
    private assessments bucket and return its
    storage path.
    """

    storage_path = get_prediction_mask_path(
        assessment_id
    )

    _validate_image_bytes(mask_bytes)

    return _upload_bytes(
        storage_path,
        mask_bytes,
        PREDICTION_MASK_CONTENT_TYPE
    )


# ============================================================
# PDF REPORT
# ============================================================

def _validate_report_bytes(pdf_bytes: bytes) -> None:
    """
    Reject payloads that are not compact PDFs.

    Guards the storage budget by refusing oversized
    reports before they are sent to Supabase.
    """

    if not pdf_bytes or not isinstance(
        pdf_bytes, bytes
    ):
        raise ValueError(
            "report_bytes must be non-empty bytes."
        )

    if not pdf_bytes.startswith(PDF_MAGIC):
        raise ValueError(
            "report_bytes must start with the "
            "%PDF- header."
        )

    if len(pdf_bytes) > MAX_REPORT_SIZE:
        size_mb = len(pdf_bytes) / (1024 * 1024)

        raise ValueError(
            f"Report is {size_mb:.2f} MB, which exceeds "
            f"the {MAX_REPORT_SIZE // (1024 * 1024)} MB "
            "storage limit."
        )


def get_assessment_report_path(
    assessment_id: str
) -> str:
    """
    Return the private storage path for an assessment
    PDF report.
    """

    if not assessment_id or not isinstance(
        assessment_id, str
    ):
        raise ValueError(
            "assessment_id must be a non-empty string."
        )

    return (
        f"{STORAGE_BUCKET}/"
        f"{assessment_id}/"
        f"{REPORT_FILENAME}"
    )


def upload_assessment_report(
    assessment_id: str,
    pdf_bytes: bytes
) -> str:
    """
    Upload an assessment PDF report to the private
    assessments bucket and return its storage path.

    The upload uses upsert=False, so an existing report
    is never overwritten and no duplicate PDF is created.
    """

    storage_path = get_assessment_report_path(
        assessment_id
    )

    _validate_report_bytes(pdf_bytes)

    return _upload_bytes(
        storage_path,
        pdf_bytes,
        REPORT_CONTENT_TYPE
    )


# ============================================================
# SIGNED URL ACCESS
# ============================================================

# The assessments bucket is private, so the browser can
# never load an object directly. FastAPI mints a short-lived
# signed URL on request instead, so a leaked link stops
# working within minutes.

DEFAULT_SIGNED_URL_TTL = 300  # 5 minutes

# A caller may ask for less time than the default but can
# never extend a link beyond this ceiling.

MAX_SIGNED_URL_TTL = 900  # 15 minutes


def _validate_signed_url_ttl(
    expires_in: int
) -> int:
    """
    Clamp a requested signed-URL lifetime to the
    maximum allowed short-lived window.
    """

    if isinstance(expires_in, bool) or not isinstance(
        expires_in, int
    ):
        raise ValueError(
            "expires_in must be an integer number of "
            "seconds."
        )

    if expires_in <= 0:
        raise ValueError(
            "expires_in must be positive."
        )

    return min(expires_in, MAX_SIGNED_URL_TTL)


def create_signed_asset_url(
    storage_path: str,
    expires_in: int = DEFAULT_SIGNED_URL_TTL
) -> str:
    """
    Return a short-lived signed URL for one object in the
    private assessments bucket.

    The caller must already have decided that the object
    exists; this function only signs the path it is given
    and never inspects or returns a permanent public URL.

    Raises ValueError for an invalid path or lifetime and
    RuntimeError if Supabase refuses to sign the object.
    """

    relative_path = _bucket_relative_path(
        storage_path
    )

    ttl = _validate_signed_url_ttl(expires_in)

    bucket = _storage_bucket()

    try:
        signed = bucket.create_signed_url(
            relative_path,
            expires_in=ttl
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to create a signed URL for "
            f"{relative_path}: {error}"
        ) from error

    # storage3 returns a mapping, but tolerate a plain
    # string or an object exposing the same attributes so
    # a client upgrade cannot silently break delivery.

    url = None

    if isinstance(signed, str):
        url = signed
    elif isinstance(signed, dict):
        url = (
            signed.get("signedURL")
            or signed.get("signed_url")
        )
    else:
        url = (
            getattr(signed, "signedURL", None)
            or getattr(signed, "signed_url", None)
        )

    if not url or not isinstance(url, str):
        raise RuntimeError(
            "Supabase returned no signed URL for "
            f"{relative_path}."
        )

    return url


def asset_object_exists(
    storage_path: str
) -> bool:
    """
    Report whether one private object is still present in
    the assessments bucket.

    A missing object returns False so a stale database path
    cannot produce a signed link that never resolves. A real
    Storage failure is raised instead of being reported as
    "missing", so an outage is never mistaken for an absent
    asset.
    """

    relative_path = _bucket_relative_path(
        storage_path
    )

    bucket = _storage_bucket()

    try:
        return bool(bucket.exists(relative_path))
    except Exception as error:
        raise RuntimeError(
            f"Failed to check assessment object "
            f"{relative_path}: {error}"
        ) from error


# ============================================================
# DELETE
# ============================================================

def delete_assessment_image(
    storage_path: str
) -> None:
    """
    Delete one assessment image from the private
    assessments bucket.

    Accepts either a bucket-qualified path or a
    path that is already relative to the bucket.
    Raises RuntimeError if the deletion fails.
    """

    relative_path = _bucket_relative_path(
        storage_path
    )

    bucket = _storage_bucket()

    try:
        results = bucket.remove([relative_path])
    except Exception as error:
        raise RuntimeError(
            f"Failed to delete assessment image "
            f"{relative_path}: {error}"
        ) from error

    if isinstance(results, list):
        if not results:
            raise RuntimeError(
                "Failed to delete assessment image "
                f"{relative_path}: no object was removed"
            )

        first = results[0]

        if isinstance(first, dict):
            error = first.get("error")

            if error:
                raise RuntimeError(
                    "Failed to delete assessment image "
                    f"{relative_path}: {error}"
                )


def get_assessment_image_info(
    storage_path: str
) -> Dict[str, Any]:
    """
    Return the stored object metadata for one
    assessment image, used to confirm an upload
    landed in the bucket.
    """

    relative_path = _bucket_relative_path(
        storage_path
    )

    bucket = _storage_bucket()

    try:
        return bucket.info(relative_path)
    except Exception as error:
        raise RuntimeError(
            f"Failed to read assessment image info "
            f"{relative_path}: {error}"
        ) from error
