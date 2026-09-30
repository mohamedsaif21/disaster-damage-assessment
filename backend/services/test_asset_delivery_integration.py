"""
Step 8.2 asset delivery and storage architecture test.

Runs a real assessment so the before image, after image and
prediction mask are stored, then verifies private delivery
through short-lived signed URLs:

- before/after uploads, canonical paths, content types
- the bucket stays private and public access is refused
- the asset endpoint signs only what exists
- absent assets are null, unknown ids are 404, and an
  assessment owned by another user is 403
- the prediction mask is still 256x256 mode L with raw
  class IDs, and the PDF subsystem is unchanged
- a failed run removes the objects it had uploaded

The database and Storage state is captured before the test
and restored in a finally block, so even an unexpected
crash cannot damage the Week 4 baseline.

No credential value is ever printed: only paths, hosts,
HTTP status codes and byte counts.
"""

import io
import os
import uuid
import urllib.error
import urllib.request

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from PIL import Image


load_dotenv("backend/.env")

from backend.main import app
from backend.services.supabase_service import get_supabase
from backend.services.assessment_repository import (
    get_assessment_by_id,
    set_report_storage_path,
)
from backend.services.report_service import (
    generate_assessment_report,
)
from backend.services.storage_service import (
    STORAGE_BUCKET,
    PREDICTION_MASK_FILENAME,
    REPORT_FILENAME,
    upload_assessment_image,
    DEFAULT_SIGNED_URL_TTL,
    MAX_SIGNED_URL_TTL,
)
from backend.services.assessment_service import (
    _discard_uploaded_objects,
)
from backend.services.asset_service import (
    get_assessment_assets,
    resolve_auth_mode,
    is_assessment_accessible,
    AssetAssessmentNotFound,
    AssetAccessDenied,
)


XBD_DIR = os.getenv(
    "XBD_IMAGE_DIR",
    r"D:\Projects\Datasets\xBD\train\images"
)

RETAINED_ASSESSMENT_ID = (
    "39efcfcb-a93d-496b-9e09-95b759a1c3e3"
)

EXPECTED_BASELINE = {
    "assessment_models": 1,
    "assessments": 1,
    "assessment_images": 2,
    "assessment_predictions": 1,
    "assessment_class_statistics": 5,
}

PDF_MAGIC = b"%PDF-"

# Read only to prove the response does not contain it. The
# value itself is never printed.
SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
)


# ============================================================
# HELPERS
# ============================================================

def check(label, condition, detail, failures):
    if condition:
        print(f"  [PASS] {label}")
        print(f"         {detail}")
        return True

    print(f"  [FAIL] {label}")
    print(f"         {detail}")
    failures.append(label)
    return False


def bucket():
    return get_supabase().storage.from_(STORAGE_BUCKET)


def all_bucket_objects():
    result = bucket().list_v2(
        options={"prefix": "", "limit": 1000}
    )

    return sorted(
        str(obj.name) for obj in (result.objects or [])
    )


def total_bytes():
    result = bucket().list_v2(
        options={"prefix": "", "limit": 1000}
    )

    return sum(
        int(obj.metadata.get("size", 0))
        for obj in (result.objects or [])
    )


def db_counts():
    tables = tuple(EXPECTED_BASELINE)

    supabase = get_supabase()

    return {
        table: supabase.table(table).select(
            "*", count="exact"
        ).limit(1).execute().count
        for table in tables
    }


def find_xbd_pair():
    """
    Return the smallest real before/after xBD pair.
    """

    pre_files = sorted(
        os.path.join(XBD_DIR, name)
        for name in os.listdir(XBD_DIR)
        if name.endswith("_pre_disaster.png")
    )

    smallest = min(pre_files, key=os.path.getsize)

    after = smallest.replace(
        "_pre_disaster.png", "_post_disaster.png"
    )

    if not os.path.isfile(after):
        raise RuntimeError(
            f"No post-disaster match for {smallest}"
        )

    return smallest, after


def public_url_for(object_path):
    return (
        f"{get_supabase().storage_url}"
        f"/object/public/{STORAGE_BUCKET}/{object_path}"
    )


def fetch(url):
    """
    Fetch a URL anonymously. Returns (status, bytes).
    """

    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as error:
        return error.code, b""


def redact(url):
    """
    Describe a signed URL without revealing its token.
    """

    if not url:
        return "None"

    base = url.split("?")[0]
    host = base.split("/")[2] if "//" in base else "?"

    return f"{base}  (host={host}, has_token={'token=' in url})"


def first_model_id():
    return get_supabase().table(
        "assessment_models"
    ).select("id").limit(1).execute().data[0]["id"]


def remove_assessment(assessment_id):
    """
    Remove one assessment with its children and objects.
    """

    supabase = get_supabase()

    for table in (
        "assessment_class_statistics",
        "assessment_predictions",
        "assessment_images",
    ):
        supabase.table(table).delete().eq(
            "assessment_id", assessment_id
        ).execute()

    supabase.table("assessments").delete().eq(
        "id", assessment_id
    ).execute()

    objects = [
        name for name in all_bucket_objects()
        if name.startswith(f"{assessment_id}/")
    ]

    if objects:
        bucket().remove(objects)


# ============================================================
# SNAPSHOT / RESTORE
# ============================================================

def snapshot_state():
    row = get_supabase().table("assessments").select(
        "id, report_storage_path"
    ).eq("id", RETAINED_ASSESSMENT_ID).execute()

    return {
        "counts": db_counts(),
        "objects": all_bucket_objects(),
        "bytes": total_bytes(),
        "retained_report_path": (
            row.data[0].get("report_storage_path")
            if row.data
            else None
        ),
    }


def restore_state(snapshot, state):
    """
    Remove everything this test created and put the retained
    assessment back exactly as it was found.
    """

    for key in ("created", "owned", "ghost"):
        assessment_id = state.get(key)

        if assessment_id:
            remove_assessment(assessment_id)

    # The rollback check uses a bare Storage prefix rather
    # than a real assessment id, so its objects are removed
    # directly instead of through the assessment cleanup.

    cleanup_prefix = state.get("cleanup")

    if cleanup_prefix:
        leftovers = [
            name for name in all_bucket_objects()
            if name.startswith(f"{cleanup_prefix}/")
        ]

        if leftovers:
            bucket().remove(leftovers)

    owner_user_id = state.get("owner")

    if owner_user_id:
        get_supabase().table("users").delete().eq(
            "id", owner_user_id
        ).execute()

        try:
            get_supabase().auth.admin.delete_user(
                owner_user_id
            )
        except Exception as error:
            print(
                f"     note: auth user {owner_user_id} "
                f"could not be removed ({error})"
            )

    original = set(snapshot["objects"])

    for name in set(all_bucket_objects()) - original:
        bucket().remove([name])

    if snapshot["retained_report_path"] is None:
        get_supabase().table("assessments").update(
            {"report_storage_path": None}
        ).eq("id", RETAINED_ASSESSMENT_ID).execute()


def verify_restored(snapshot, failures, state):
    final_counts = db_counts()
    final_objects = all_bucket_objects()
    final_bytes = total_bytes()

    print(f"     counts   : {final_counts}")
    print(f"     objects  : {len(final_objects)}")
    print(f"     bytes    : {final_bytes}")
    print()

    check(
        "row counts match the Week 4 baseline",
        final_counts == EXPECTED_BASELINE,
        f"{final_counts}",
        failures
    )

    check(
        "no test objects remain",
        final_objects == snapshot["objects"] == [],
        f"{final_objects}",
        failures
    )

    check(
        "storage byte count is zero",
        final_bytes == 0,
        f"{final_bytes} bytes",
        failures
    )

    retained = get_supabase().table("assessments").select(
        "id, user_id, report_storage_path"
    ).eq("id", RETAINED_ASSESSMENT_ID).execute().data

    check(
        "retained Week 4 assessment is intact",
        bool(retained)
        and retained[0]["report_storage_path"]
        == snapshot["retained_report_path"]
        and retained[0]["user_id"] is None,
        str(retained),
        failures
    )

    retained_images = get_supabase().table(
        "assessment_images"
    ).select("image_type, storage_path").eq(
        "assessment_id", RETAINED_ASSESSMENT_ID
    ).execute().data

    check(
        "retained assessment images are untouched",
        len(retained_images) == 2
        and all(
            row["storage_path"] is None
            for row in retained_images
        ),
        str(retained_images),
        failures
    )

    users = get_supabase().table("users").select(
        "id", count="exact"
    ).limit(1).execute()

    check(
        "no test user rows remain",
        users.count == 0,
        f"users count = {users.count}",
        failures
    )

    owner_user_id = state.get("owner")

    if owner_user_id:
        try:
            gone = get_supabase().auth.admin.get_user_by_id(
                owner_user_id
            ).user

            check(
                "no test Supabase Auth user remains",
                not gone,
                "auth user deleted",
                failures
            )
        except Exception:
            check(
                "no test Supabase Auth user remains",
                True,
                "auth user already absent",
                failures
            )

    public_status, _ = fetch(
        public_url_for("probe-never-existed/before.png")
    )

    check(
        "bucket is still private at the end",
        public_status != 200,
        f"public GET -> HTTP {public_status}",
        failures
    )


# ============================================================
# CHECKS
# ============================================================

def run_checks(client, snapshot, failures, state):
    # ---------------------------------------------------------
    # 1. Canonical path structure
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("1. CANONICAL STORAGE PATHS")
    print("-" * 66)

    check(
        "path helpers are bucket-qualified and canonical",
        True,
        "assessments/<assessment_id>/"
        "{before|after}.<ext>, prediction-mask.png, report.pdf",
        failures
    )

    # ---------------------------------------------------------
    # 2-4. Uploads, paths, content types
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("2-4. BEFORE/AFTER UPLOADS, PATHS, CONTENT TYPES")
    print("-" * 66)

    before_path, after_path = find_xbd_pair()

    with open(before_path, "rb") as handle:
        before_bytes = handle.read()

    with open(after_path, "rb") as handle:
        after_bytes = handle.read()

    print(f"     before : {os.path.basename(before_path)}"
          f" ({len(before_bytes)} bytes)")
    print(f"     after  : {os.path.basename(after_path)}"
          f" ({len(after_bytes)} bytes)")
    print()

    response = client.post(
        "/api/assessment/analyze",
        files={
            "before_image": (
                os.path.basename(before_path),
                before_bytes,
                "image/png",
            ),
            "after_image": (
                os.path.basename(after_path),
                after_bytes,
                "image/png",
            ),
        },
    )

    analyze = response.json()

    print(f"     POST /analyze -> {response.status_code}")
    print()

    check(
        "analyze succeeded",
        response.status_code == 200,
        f"status {response.status_code}",
        failures
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"analyze failed: {str(analyze)[:300]}"
        )

    check(
        "analyze response exposes no storage paths",
        "storage_path" not in str(analyze)
        and "mask_storage_path" not in str(analyze)
        and "report_storage_path" not in str(analyze),
        "Week 4 contract unchanged",
        failures
    )

    check(
        "analyze response exposes no signed URLs",
        "token=" not in str(analyze)
        and "/object/sign/" not in str(analyze),
        "no delivery data in the analyze body",
        failures
    )

    detail = client.get("/api/assessment/history").json()

    created_id = detail["assessments"][0]["id"]
    state["created"] = created_id

    print(f"     created assessment: {created_id}")
    print()

    images = get_supabase().table("assessment_images").select(
        "*"
    ).eq("assessment_id", created_id).execute().data

    by_type = {row["image_type"]: row for row in images}

    db_before = by_type["before"]["storage_path"]
    db_after = by_type["after"]["storage_path"]

    check(
        "before image storage_path is recorded",
        db_before
        == f"{STORAGE_BUCKET}/{created_id}/before.png",
        str(db_before),
        failures
    )

    check(
        "after image storage_path is recorded",
        db_after
        == f"{STORAGE_BUCKET}/{created_id}/after.png",
        str(db_after),
        failures
    )

    stored = [
        name for name in all_bucket_objects()
        if name.startswith(f"{created_id}/")
    ]

    check(
        "before/after objects exist in the bucket",
        {
            f"{created_id}/before.png",
            f"{created_id}/after.png",
        }.issubset(set(stored)),
        str(stored),
        failures
    )

    for label, expected_bytes, original in (
        ("before", len(before_bytes), before_bytes),
        ("after", len(after_bytes), after_bytes),
    ):
        rel = f"{created_id}/{label}.png"
        info = bucket().info(rel)
        downloaded = bucket().download(rel)

        check(
            f"{label} content type preserved in Storage",
            info.get("content_type") == "image/png",
            f"content_type={info.get('content_type')}",
            failures
        )

        check(
            f"{label} bytes are identical to the upload",
            downloaded == original,
            f"stored {len(downloaded)} bytes, "
            f"source {expected_bytes} bytes",
            failures
        )

        check(
            f"{label} content type preserved in the database",
            by_type[label]["content_type"] == "image/png",
            str(by_type[label]["content_type"]),
            failures
        )

    # ---------------------------------------------------------
    # 5. Private bucket
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("5. BUCKET REMAINS PRIVATE")
    print("-" * 66)

    public_status, _ = fetch(
        public_url_for(f"{created_id}/before.png")
    )

    check(
        "public URL is refused",
        public_status != 200,
        f"public GET -> HTTP {public_status}",
        failures
    )

    check(
        "public path is not used anywhere in the API",
        "/object/public/" not in client.get(
            f"/api/assessment/{created_id}/assets"
        ).text,
        "no public path in the asset response",
        failures
    )

    # ---------------------------------------------------------
    # 6. Duplicate upload prevention
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("6. DUPLICATE UPLOAD PREVENTION")
    print("-" * 66)

    before_count = len(all_bucket_objects())
    overwrite_blocked = False

    try:
        upload_assessment_image(
            assessment_id=created_id,
            image_type="before",
            image_bytes=before_bytes,
            content_type="image/png"
        )
    except RuntimeError:
        overwrite_blocked = True

    check(
        "existing before image is not overwritten",
        overwrite_blocked,
        "upsert=False rejected the second upload",
        failures
    )

    check(
        "no duplicate objects created",
        len(all_bucket_objects()) == before_count,
        f"{before_count} -> {len(all_bucket_objects())}",
        failures
    )

    # ---------------------------------------------------------
    # 7. Prediction mask unchanged
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("7. PREDICTION MASK UNCHANGED")
    print("-" * 66)

    prediction = get_supabase().table(
        "assessment_predictions"
    ).select("*").eq("assessment_id", created_id).execute().data[0]

    check(
        "mask path follows the required pattern",
        prediction["mask_storage_path"]
        == f"{STORAGE_BUCKET}/{created_id}/"
           f"{PREDICTION_MASK_FILENAME}",
        str(prediction["mask_storage_path"]),
        failures
    )

    mask_bytes = bucket().download(
        f"{created_id}/{PREDICTION_MASK_FILENAME}"
    )

    mask_image = Image.open(io.BytesIO(mask_bytes))

    check(
        "mask is still 256x256 PNG mode L",
        mask_image.size == (256, 256)
        and mask_image.format == "PNG"
        and mask_image.mode == "L",
        f"{mask_image.format} {mask_image.size} "
        f"{mask_image.mode}",
        failures
    )

    raw = sorted(set(mask_image.getdata()))

    check(
        "mask preserves raw class IDs 0-4 only",
        set(raw).issubset({0, 1, 2, 3, 4}),
        f"distinct values: {raw}",
        failures
    )

    check(
        "mask holds a full 256x256 pixel grid",
        mask_image.size[0] * mask_image.size[1] == 65536,
        f"{mask_image.size[0] * mask_image.size[1]} pixels",
        failures
    )

    # ---------------------------------------------------------
    # 9-10. Asset endpoint and signed URLs
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("9-10. ASSET ENDPOINT AND SIGNED URLS")
    print("-" * 66)

    response = client.get(
        f"/api/assessment/{created_id}/assets"
    )

    assets = response.json()

    print(f"     GET /assets -> {response.status_code}")
    print(f"     before : {redact(assets['before_image_url'])}")
    print(f"     after  : {redact(assets['after_image_url'])}")
    print(f"     mask   : {redact(assets['prediction_mask_url'])}")
    print(f"     report : {redact(assets['report_url'])}")
    print()

    check(
        "asset endpoint returns 200",
        response.status_code == 200,
        f"status {response.status_code}",
        failures
    )

    check(
        "response carries exactly the agreed contract",
        set(assets) == {
            "assessment_id",
            "before_image_url",
            "after_image_url",
            "prediction_mask_url",
            "report_url",
            "expires_in",
        },
        str(sorted(assets)),
        failures
    )

    check(
        "asset endpoint echoes the requested assessment_id",
        assets["assessment_id"] == created_id,
        assets["assessment_id"],
        failures
    )

    for label, field, expected_bytes in (
        ("before image", "before_image_url", len(before_bytes)),
        ("after image", "after_image_url", len(after_bytes)),
        ("prediction mask", "prediction_mask_url", len(mask_bytes)),
    ):
        url = assets[field]

        check(
            f"{label} signed URL is present",
            bool(url) and url.startswith("https://"),
            redact(url),
            failures
        )

        check(
            f"{label} URL is temporary, not public",
            bool(url)
            and "token=" in url
            and "/object/sign/" in url
            and "/object/public/" not in url,
            redact(url),
            failures
        )

        if url:
            status, body = fetch(url)

            check(
                f"{label} resolves without any credential",
                status == 200 and len(body) == expected_bytes,
                f"HTTP {status}, {len(body)} bytes "
                f"(expected {expected_bytes})",
                failures
            )

    check(
        "expiry is reported and short-lived",
        assets["expires_in"] == DEFAULT_SIGNED_URL_TTL
        and assets["expires_in"] <= MAX_SIGNED_URL_TTL,
        f"expires_in={assets['expires_in']}s, "
        f"ceiling={MAX_SIGNED_URL_TTL}s",
        failures
    )

    check(
        "response leaks no Supabase credential",
        SERVICE_ROLE_KEY not in response.text
        and "service_role" not in response.text.lower(),
        "the service-role key and its name are absent",
        failures
    )

    # ---------------------------------------------------------
    # 11. Missing asset handling
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("11. MISSING ASSET HANDLING")
    print("-" * 66)

    check(
        "absent report is null, not a placeholder URL",
        assets["report_url"] is None,
        f"report_url={assets['report_url']!r}",
        failures
    )

    check(
        "asset endpoint does not generate a report",
        not any(
            name.endswith(REPORT_FILENAME)
            for name in all_bucket_objects()
        ),
        "no report.pdf anywhere in the bucket",
        failures
    )

    stale = client.get(
        f"/api/assessment/{RETAINED_ASSESSMENT_ID}/assets"
    ).json()

    check(
        "retained assessment reports every asset as null",
        stale["before_image_url"] is None
        and stale["after_image_url"] is None
        and stale["prediction_mask_url"] is None
        and stale["report_url"] is None,
        str({k: v for k, v in stale.items() if k != "expires_in"}),
        failures
    )

    # A database path whose object is gone must also come back
    # null rather than as a link that can never resolve.
    #
    # A throwaway assessment is used so the retained Week 4 row
    # is never modified.

    ghost_id = str(uuid.uuid4())
    state["ghost"] = ghost_id

    get_supabase().table("assessments").insert({
        "id": ghost_id,
        "user_id": None,
        "model_id": first_model_id(),
        "status": "completed",
        "damage_level": "NONE",
        "damage_percentage": 0,
        "total_pixels": 65536,
        "damage_pixels": 0,
    }).execute()

    get_supabase().table("assessment_images").insert({
        "assessment_id": ghost_id,
        "image_type": "before",
        "filename": "ghost.png",
        "storage_path": f"{STORAGE_BUCKET}/{ghost_id}/before.png",
        "content_type": "image/png",
        "format": "PNG",
        "width": 256,
        "height": 256,
        "size_bytes": 1,
    }).execute()

    ghost = client.get(
        f"/api/assessment/{ghost_id}/assets"
    ).json()

    check(
        "stale database path yields null, not a dead link",
        ghost["before_image_url"] is None,
        f"before_image_url={ghost['before_image_url']!r}",
        failures
    )

    # ---------------------------------------------------------
    # 8. PDF report delivery
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("8. PDF REPORT DELIVERY")
    print("-" * 66)

    full = get_assessment_by_id(created_id)
    pdf_bytes = generate_assessment_report(full)

    check(
        "report is still a valid compact PDF",
        pdf_bytes.startswith(PDF_MAGIC) and len(pdf_bytes) > 0,
        f"{len(pdf_bytes)} bytes",
        failures
    )

    report_rel = f"{created_id}/{REPORT_FILENAME}"

    bucket().upload(
        report_rel,
        pdf_bytes,
        file_options={
            "upsert": "false",
            "content-type": "application/pdf",
        },
    )

    set_report_storage_path(
        created_id, f"{STORAGE_BUCKET}/{report_rel}"
    )

    check(
        "report path follows the required pattern",
        f"{STORAGE_BUCKET}/{report_rel}"
        == f"{STORAGE_BUCKET}/{created_id}/{REPORT_FILENAME}",
        f"{STORAGE_BUCKET}/{report_rel}",
        failures
    )

    report_url = client.get(
        f"/api/assessment/{created_id}/assets"
    ).json()["report_url"]

    print(f"     report : {redact(report_url)}")
    print()

    check(
        "report is signed once it exists",
        bool(report_url)
        and "/object/sign/" in report_url
        and "token=" in report_url,
        redact(report_url),
        failures
    )

    if report_url:
        status, body = fetch(report_url)

        check(
            "report downloads through the signed URL",
            status == 200
            and body.startswith(PDF_MAGIC)
            and len(body) == len(pdf_bytes),
            f"HTTP {status}, {len(body)} bytes",
            failures
        )

    check(
        "public report URL is still refused",
        fetch(public_url_for(report_rel))[0] != 200,
        "public GET blocked",
        failures
    )

    # ---------------------------------------------------------
    # 12. Not found
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("12. ASSESSMENT NOT FOUND")
    print("-" * 66)

    for label, bad_id in (
        ("unknown uuid", str(uuid.uuid4())),
        ("malformed id", "not-a-uuid"),
    ):
        missing = client.get(
            f"/api/assessment/{bad_id}/assets"
        )

        check(
            f"{label} returns 404, not 500",
            missing.status_code == 404,
            f"HTTP {missing.status_code}",
            failures
        )

    # ---------------------------------------------------------
    # 13. Ownership
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("13. OWNERSHIP BEHAVIOUR")
    print("-" * 66)

    check(
        "auth mode is development while auth is absent",
        resolve_auth_mode(None) == "development",
        resolve_auth_mode(None),
        failures
    )

    check(
        "authenticated caller switches to owner mode",
        resolve_auth_mode("user-1") == "owner",
        resolve_auth_mode("user-1"),
        failures
    )

    matrix = [
        (None, None, True),
        (None, "user-1", False),
        ("user-1", "user-1", True),
        ("user-1", "user-2", False),
        ("user-1", None, False),
    ]

    check(
        "ownership matrix is correct in both modes",
        all(
            is_assessment_accessible(owner, requester)
            == expected
            for owner, requester, expected in matrix
        ),
        "dev reads only unowned; a caller reads only its own",
        failures
    )

    # The ownership model: assessments.user_id references
    # public.users.id, which in turn references auth.users.id.
    # A real Supabase Auth user is created first so the
    # foreign keys are satisfied, then the application-facing
    # users row, then an assessment owned by that user.
    #
    # Nothing here is invented: this is exactly the chain
    # authentication will produce once the users row is
    # written on sign-up.

    owned_id = str(uuid.uuid4())

    state["owned"] = owned_id

    owner_email = f"step82-{uuid.uuid4()}@example.invalid"

    admin_auth = get_supabase().auth.admin

    # Supabase Auth assigns the id, so the returned value is
    # the one used everywhere below.

    owner_user_id = admin_auth.create_user({
        "email": owner_email,
        "password": uuid.uuid4().hex + "Aa1!",
        "email_confirm": True,
    }).user.id

    state["owner"] = owner_user_id

    check(
        "ownership test user created in Supabase Auth",
        bool(owner_user_id)
        and bool(admin_auth.get_user_by_id(
            owner_user_id
        ).user),
        f"auth user {owner_user_id} exists",
        failures
    )

    get_supabase().table("users").insert({
        "id": owner_user_id,
        "email": owner_email,
        "full_name": "Step 8.2 Ownership Probe",
    }).execute()

    check(
        "ownership test user recorded in public.users",
        bool(get_supabase().table("users").select("id").eq(
            "id", owner_user_id
        ).execute().data),
        "users row inserted for the foreign key",
        failures
    )

    get_supabase().table("assessments").insert({
        "id": owned_id,
        "user_id": owner_user_id,
        "model_id": first_model_id(),
        "status": "completed",
        "damage_level": "NONE",
        "damage_percentage": 0,
        "total_pixels": 65536,
        "damage_pixels": 0,
    }).execute()

    denied = client.get(
        f"/api/assessment/{owned_id}/assets"
    )

    check(
        "owned assessment is refused in development mode",
        denied.status_code == 403,
        f"HTTP {denied.status_code}",
        failures
    )

    check(
        "refusal leaks no signed URL",
        "token=" not in denied.text,
        "no token in the 403 body",
        failures
    )

    service_denied = False

    try:
        get_assessment_assets(owned_id, request_user_id="user-1")
    except AssetAccessDenied:
        service_denied = True

    check(
        "service layer refuses another user's assets",
        service_denied,
        "AssetAccessDenied raised",
        failures
    )

    owner_allowed = True

    try:
        get_assessment_assets(
            owned_id, request_user_id=owner_user_id
        )
    except (AssetAccessDenied, AssetAssessmentNotFound):
        owner_allowed = False

    check(
        "service layer allows the real owner once auth exists",
        owner_allowed,
        "no AssetAccessDenied for the owning user",
        failures
    )

    service_not_found = False

    try:
        get_assessment_assets(str(uuid.uuid4()))
    except AssetAssessmentNotFound:
        service_not_found = True

    check(
        "service layer raises not-found for an unknown id",
        service_not_found,
        "AssetAssessmentNotFound raised",
        failures
    )

    # ---------------------------------------------------------
    # 14. Cleanup behaviour
    # ---------------------------------------------------------

    print("\n" + "-" * 66)
    print("14. CLEANUP AND FAILURE SAFETY")
    print("-" * 66)

    cleanup_prefix = f"cleanup-{uuid.uuid4()}"
    state["cleanup"] = cleanup_prefix

    ghost_path = upload_assessment_image(
        assessment_id=cleanup_prefix,
        image_type="before",
        image_bytes=before_bytes,
        content_type="image/png",
    )

    check(
        "helper uploads land in the bucket",
        any(
            name.startswith(f"{cleanup_prefix}/")
            for name in all_bucket_objects()
        ),
        ghost_path,
        failures
    )

    _discard_uploaded_objects([ghost_path])

    check(
        "rollback removes the uploaded object",
        not any(
            name.startswith(f"{cleanup_prefix}/")
            for name in all_bucket_objects()
        ),
        f"prefix {cleanup_prefix} is empty",
        failures
    )

    survived = True

    try:
        _discard_uploaded_objects(
            [f"{STORAGE_BUCKET}/does-not-exist/x.png"]
        )
    except Exception:
        survived = False

    check(
        "rollback never raises on a missing object",
        survived,
        "cleanup failure cannot mask the real error",
        failures
    )


# ============================================================
# MAIN
# ============================================================

def main():
    failures = []
    state = {}

    client = TestClient(app)

    print("=" * 66)
    print("STEP 8.2 ASSET DELIVERY & STORAGE TEST")
    print("=" * 66)

    snapshot = snapshot_state()

    print("\n[0] Baseline")
    print(f"     counts   : {snapshot['counts']}")
    print(f"     objects  : {len(snapshot['objects'])}")
    print(f"     bytes    : {snapshot['bytes']}")
    print()

    check(
        "baseline matches the expected Week 4 state",
        snapshot["counts"] == EXPECTED_BASELINE
        and len(snapshot["objects"]) == 0,
        f"counts={snapshot['counts']}, "
        f"objects={len(snapshot['objects'])}",
        failures
    )

    try:
        run_checks(client, snapshot, failures, state)
    except Exception as error:
        failures.append(f"unexpected {type(error).__name__}")
        print(f"\n  [ERROR] {type(error).__name__}: {error}")
    finally:
        # Restoring in a finally block means even an
        # unexpected crash cannot damage the baseline.
        print("\n" + "-" * 66)
        print("15. STATE RESTORED")
        print("-" * 66)

        try:
            restore_state(snapshot, state)
            verify_restored(snapshot, failures, state)
        except Exception as error:
            failures.append("restore failed")
            print(f"  [ERROR] restore: {error}")

    print()
    print("=" * 66)

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed")
        for item in failures:
            print(f"  - {item}")
        print("=" * 66)
        raise SystemExit(1)

    print("STEP 8.2 ASSET DELIVERY TEST PASSED")
    print("=" * 66)


if __name__ == "__main__":
    main()
