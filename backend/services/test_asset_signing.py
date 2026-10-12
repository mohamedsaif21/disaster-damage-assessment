"""
Offline verification of the private asset delivery contract.

The live asset-delivery integration test needs a reachable
Supabase project. While that project is unavailable, this
file verifies the same service-level rules without any
network access by replacing the repository and Storage
helpers with deterministic fakes:

  1. a stored report with a present object is signed
  2. a stored report whose object is gone becomes null
  3. an absent report path becomes null
  4. an unknown assessment raises AssetAssessmentNotFound
  5. another owner's assessment raises AssetAccessDenied
  6. development mode only exposes unowned assessments
  7. the reported lifetime is capped at the storage ceiling
  8. a real signing failure propagates instead of being
     disguised as a missing asset
  9. before/after/mask objects are signed through the same
     path as the report

Nothing here touches Supabase, the network, or the bucket.
"""

import sys

from dotenv import load_dotenv


load_dotenv("backend/.env")

import backend.services.asset_service as asset_service
from backend.services.asset_service import (
    get_assessment_assets,
    resolve_auth_mode,
    is_assessment_accessible,
    AssetAssessmentNotFound,
    AssetAccessDenied,
    AUTH_MODE_DEVELOPMENT,
    AUTH_MODE_OWNER,
)
from backend.services.storage_service import (
    DEFAULT_SIGNED_URL_TTL,
    MAX_SIGNED_URL_TTL,
)


ASSESSMENT_ID = "11111111-1111-1111-1111-111111111111"

OWNER_ID = "owner-user-id"


class Patch:
    def __init__(self):
        self._saved = []

    def set(self, owner, name, value):
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def restore(self):
        for owner, name, original in reversed(self._saved):
            setattr(owner, name, original)
        self._saved = []


def check(label, condition, detail, failures):
    if condition:
        print(f"  [PASS] {label}")
        print(f"         {detail}")
        return True

    print(f"  [FAIL] {label}")
    print(f"         {detail}")
    failures.append(label)
    return False


def asset_paths(user_id, report="assessments/x/report.pdf"):
    return {
        "id": ASSESSMENT_ID,
        "user_id": user_id,
        "before_storage_path": "assessments/x/before.png",
        "after_storage_path": "assessments/x/after.png",
        "mask_storage_path": "assessments/x/prediction-mask.png",
        "report_storage_path": report,
    }


def run_cases(failures, patch):
    signed = {}

    def fake_sign(storage_path, expires_in=DEFAULT_SIGNED_URL_TTL):
        signed[storage_path] = expires_in
        return f"https://signed.example/{storage_path}?token=t"

    patch.set(asset_service, "create_signed_asset_url", fake_sign)

    # 1. report present and object in the bucket
    signed.clear()
    patch.set(
        asset_service,
        "get_assessment_asset_paths",
        lambda assessment_id: asset_paths(OWNER_ID),
    )
    patch.set(
        asset_service,
        "asset_object_exists",
        lambda storage_path: True,
    )

    assets = get_assessment_assets(
        ASSESSMENT_ID, OWNER_ID, expires_in=DEFAULT_SIGNED_URL_TTL
    )

    check(
        "present report is signed",
        bool(assets["report_url"])
        and assets["report_url"].startswith("https://signed.example/"),
        str(assets["report_url"]),
        failures,
    )

    check(
        "images and mask are signed too",
        all(
            assets[field]
            for field in (
                "before_image_url",
                "after_image_url",
                "prediction_mask_url",
            )
        ),
        "before/after/mask signed",
        failures,
    )

    check(
        "lifetime is capped at the storage ceiling",
        assets["expires_in"]
        == min(DEFAULT_SIGNED_URL_TTL, MAX_SIGNED_URL_TTL),
        str(assets["expires_in"]),
        failures,
    )

    # 2. report path stored but the object is gone
    patch.set(
        asset_service,
        "asset_object_exists",
        lambda storage_path: "report.pdf" not in storage_path,
    )

    assets = get_assessment_assets(ASSESSMENT_ID, OWNER_ID)

    check(
        "stored path with a missing object becomes null",
        assets["report_url"] is None,
        str(assets["report_url"]),
        failures,
    )

    check(
        "existing images stay signed alongside it",
        bool(assets["before_image_url"]),
        str(assets["before_image_url"]),
        failures,
    )

    # 3. no report path at all
    patch.set(
        asset_service,
        "asset_object_exists",
        lambda storage_path: True,
    )
    patch.set(
        asset_service,
        "get_assessment_asset_paths",
        lambda assessment_id: asset_paths(OWNER_ID, report=None),
    )

    assets = get_assessment_assets(ASSESSMENT_ID, OWNER_ID)

    check(
        "absent report path becomes null",
        assets["report_url"] is None,
        str(assets["report_url"]),
        failures,
    )

    # 4. unknown assessment
    patch.set(
        asset_service,
        "get_assessment_asset_paths",
        lambda assessment_id: None,
    )

    try:
        get_assessment_assets(ASSESSMENT_ID, OWNER_ID)
        raised = None
    except AssetAssessmentNotFound:
        raised = "AssetAssessmentNotFound"

    check(
        "unknown assessment raises not-found",
        raised == "AssetAssessmentNotFound",
        str(raised),
        failures,
    )

    # 5. another owner
    patch.set(
        asset_service,
        "get_assessment_asset_paths",
        lambda assessment_id: asset_paths(OWNER_ID),
    )

    try:
        get_assessment_assets(ASSESSMENT_ID, "someone-else")
        raised = None
    except AssetAccessDenied:
        raised = "AssetAccessDenied"

    check(
        "another owner is denied",
        raised == "AssetAccessDenied",
        str(raised),
        failures,
    )

    # 6. development mode scope
    check(
        "dev mode reads unowned assessments only",
        is_assessment_accessible(None, None)
        and not is_assessment_accessible(OWNER_ID, None),
        "unowned readable, owned not readable",
        failures,
    )

    check(
        "auth mode reflects the caller",
        resolve_auth_mode(None) == AUTH_MODE_DEVELOPMENT
        and resolve_auth_mode(OWNER_ID) == AUTH_MODE_OWNER,
        f"{resolve_auth_mode(None)}, {resolve_auth_mode(OWNER_ID)}",
        failures,
    )

    # 7. an oversized request is capped, not honored
    patch.set(
        asset_service,
        "get_assessment_asset_paths",
        lambda assessment_id: asset_paths(OWNER_ID),
    )
    patch.set(
        asset_service,
        "asset_object_exists",
        lambda storage_path: True,
    )

    assets = get_assessment_assets(
        ASSESSMENT_ID, OWNER_ID, expires_in=10 ** 7
    )

    check(
        "oversized lifetime is clamped to the ceiling",
        assets["expires_in"] == MAX_SIGNED_URL_TTL,
        f"{assets['expires_in']} vs {MAX_SIGNED_URL_TTL}",
        failures,
    )

    # 8. a real signing failure is not hidden
    def failing_sign(storage_path, expires_in=DEFAULT_SIGNED_URL_TTL):
        raise RuntimeError("signing service unavailable")

    patch.set(asset_service, "create_signed_asset_url", failing_sign)

    try:
        get_assessment_assets(ASSESSMENT_ID, OWNER_ID)
        raised = None
    except RuntimeError as error:
        raised = type(error).__name__

    check(
        "signing failure propagates, not nulled",
        raised == "RuntimeError",
        str(raised),
        failures,
    )


def main():
    failures = []
    patch = Patch()

    print("=" * 66)
    print("ASSET SIGNING VERIFICATION (OFFLINE)")
    print("=" * 66)
    print("\n[1-9] asset delivery contract")

    try:
        run_cases(failures, patch)
    finally:
        patch.restore()

    print("\n" + "=" * 66)

    if failures:
        print(
            f"ASSET SIGNING VERIFICATION FAILED: "
            f"{len(failures)} failure(s)"
        )
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("ASSET SIGNING VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())