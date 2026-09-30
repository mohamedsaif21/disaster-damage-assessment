"""
Private assessment asset delivery.

Every assessment asset lives in the private
`assessments` bucket, so the browser can never load one
directly. This service is the only place that turns a
stored Storage path into a temporary signed link:

    FastAPI
        -> authenticate / authorize
        -> read storage paths from the database
        -> sign each path that exists
        -> return short-lived URLs

Two rules shape this module:

1. The bucket stays private and no permanent public URL is
   ever produced. Links expire within minutes.
2. Supabase credentials never leave the backend. This
   service reads paths from the database, never from the
   client, so the frontend cannot choose which object it
   wants signed.

Assets that are absent are reported as null instead of
being invented, so the frontend can render "not
available" rather than a broken image.
"""

from typing import Any, Dict, Optional

from backend.services.assessment_repository import (
    get_assessment_asset_paths
)

from backend.services.storage_service import (
    create_signed_asset_url,
    asset_object_exists,
    DEFAULT_SIGNED_URL_TTL,
)


# ============================================================
# AUTHORIZATION
# ============================================================

# Authentication is not wired into the backend yet, so the
# API layer currently passes request_user_id=None to mean
# "development mode, no authenticated caller".
#
# In that mode only assessments that have no owner
# (user_id IS NULL) may be read, which is exactly the same
# scope the history endpoint already uses. When Supabase
# Auth is connected, the endpoint passes the verified user
# id instead and the same comparison enforces ownership,
# with no change to the asset code below.

AUTH_MODE_DEVELOPMENT = "development"
AUTH_MODE_OWNER = "owner"


class AssetAssessmentNotFound(Exception):
    """
    The requested assessment id does not exist.
    """


class AssetAccessDenied(Exception):
    """
    The assessment exists but does not belong to the
    requesting user.
    """


def resolve_auth_mode(
    request_user_id: Optional[str]
) -> str:
    """
    Report which authorization rule applies to this
    request.
    """

    return (
        AUTH_MODE_OWNER
        if request_user_id
        else AUTH_MODE_DEVELOPMENT
    )


def is_assessment_accessible(
    assessment_user_id: Optional[str],
    request_user_id: Optional[str],
) -> bool:
    """
    Decide whether one caller may read one assessment's
    assets.

    - With an authenticated caller, the assessment must be
      owned by exactly that user.
    - Without an authenticated caller (development mode),
      only unowned assessments are readable. A caller's id
      is never taken from the request, so this cannot be
      used to reach another user's data.
    """

    if request_user_id:
        return assessment_user_id == request_user_id

    return assessment_user_id is None


# ============================================================
# SERVICE
# ============================================================

def get_assessment_assets(
    assessment_id: str,
    request_user_id: Optional[str] = None,
    expires_in: int = DEFAULT_SIGNED_URL_TTL,
) -> Dict[str, Any]:
    """
    Return temporary signed URLs for one assessment's
    before image, after image, prediction mask and PDF
    report.

    Any asset that has no stored path, or whose object is
    no longer in the bucket, is returned as null. A report
    is never generated here: if report.pdf was never
    produced, report_url stays null.

    Raises AssetAssessmentNotFound when the id is unknown
    and AssetAccessDenied when the assessment belongs to a
    different user.
    """

    paths = get_assessment_asset_paths(assessment_id)

    if paths is None:
        raise AssetAssessmentNotFound(
            "Assessment was not found."
        )

    if not is_assessment_accessible(
        paths["user_id"],
        request_user_id
    ):
        raise AssetAccessDenied(
            "This assessment belongs to another user."
        )

    return {
        "assessment_id": paths["id"],
        "before_image_url": _sign_if_present(
            paths["before_storage_path"],
            expires_in
        ),
        "after_image_url": _sign_if_present(
            paths["after_storage_path"],
            expires_in
        ),
        "prediction_mask_url": _sign_if_present(
            paths["mask_storage_path"],
            expires_in
        ),
        "report_url": _sign_if_present(
            paths["report_storage_path"],
            expires_in
        ),
        "expires_in": min(
            expires_in,
            _max_expiry()
        ),
    }


def _max_expiry() -> int:
    """
    Expose the storage service's signed-URL ceiling so the
    response reports the lifetime actually applied.
    """

    from backend.services.storage_service import (
        MAX_SIGNED_URL_TTL
    )

    return MAX_SIGNED_URL_TTL


def _sign_if_present(
    storage_path: Optional[str],
    expires_in: int,
) -> Optional[str]:
    """
    Sign one stored path, or return None when the asset is
    unavailable.

    Only a genuinely absent object becomes null. A real
    Storage or signing failure is allowed to propagate so it
    surfaces as a 500 instead of being disguised as a
    missing asset.
    """

    if not storage_path:
        return None

    if not asset_object_exists(storage_path):
        return None

    return create_signed_asset_url(
        storage_path,
        expires_in=expires_in
    )
