"""
Step 7.13 - full backend integration verification.

Exercises the complete backend lifecycle as one system:

    upload -> validation -> analyze -> preprocessing ->
    U-Net -> prediction -> mask -> persistence -> storage ->
    history -> by-id -> error handling -> cleanup

Everything created by this test is removed afterwards and
the known baseline is restored exactly.
"""

import io
import os
import sys
import uuid

import numpy as np
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from PIL import Image


load_dotenv("backend/.env")

from backend.main import app
from backend.services.supabase_service import get_supabase


STORAGE_BUCKET = "assessments"
XBD_DIR = r"D:\Projects\Datasets\xBD\train\images"

RETAINED_ASSESSMENT_ID = (
    "39efcfcb-a93d-496b-9e09-95b759a1c3e3"
)

MASK_FILENAME = "prediction-mask.png"

BASELINE_COUNTS = {
    "assessment_models": 1,
    "assessments": 1,
    "assessment_images": 2,
    "assessment_predictions": 1,
    "assessment_class_statistics": 5,
}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

CLASS_NAMES = {
    0: "background",
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}

TABLES = [
    "assessment_models",
    "assessments",
    "assessment_images",
    "assessment_predictions",
    "assessment_class_statistics",
]


# ============================================================
# HELPERS
# ============================================================

def embedded(value):
    """
    Normalize a PostgREST embedded resource.

    A to-one embed can arrive as a single dictionary or as a
    one element list depending on the relationship, so
    normalize both before counting.
    """

    if isinstance(value, dict):
        return value

    if isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], dict):
            return value[0]

    return {}


def embedded_list(value) -> list:
    """
    Normalize a to-many embed into a real list.
    """

    if isinstance(value, dict):
        return [value]

    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, dict)]

    return []


def db_counts() -> dict:
    supabase = get_supabase()

    return {
        table: supabase.table(table)
        .select("id", count="exact")
        .execute()
        .count
        for table in TABLES
    }


def all_objects() -> list:
    result = get_supabase().storage.from_(
        STORAGE_BUCKET
    ).list_v2(options={"prefix": "", "limit": 1000})

    return sorted(
        str(obj.name)
        for obj in (result.objects or [])
    )


def total_bytes() -> int:
    result = get_supabase().storage.from_(
        STORAGE_BUCKET
    ).list_v2(options={"prefix": "", "limit": 1000})

    return sum(
        int(obj.metadata.get("size", 0))
        for obj in (result.objects or [])
    )


def find_xbd_pair() -> tuple:
    """
    Return the smallest real before/after xBD pair.
    """

    pre_files = sorted(
        os.path.join(XBD_DIR, name)
        for name in os.listdir(XBD_DIR)
        if name.endswith("_pre_disaster.png")
    )

    smallest = min(
        pre_files,
        key=os.path.getsize
    )

    after = smallest.replace(
        "_pre_disaster.png", "_post_disaster.png"
    )

    if not os.path.isfile(after):
        raise RuntimeError(
            f"No post-disaster match for {smallest}"
        )

    return smallest, after


def oversized_png() -> tuple:
    """
    Build a valid PNG larger than the 10 MB limit.
    """

    side = 2400
    noise = np.random.randint(
        0, 256, (side, side, 3), dtype=np.uint8
    )
    image = Image.fromarray(noise, mode="RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return (
        buffer.getvalue(),
        f"oversized_{side}.png",
        "image/png",
    )


def tiny_png() -> tuple:
    """
    Build a valid but undersized PNG.
    """

    buffer = io.BytesIO()
    Image.new("RGB", (100, 100)).save(
        buffer, format="PNG"
    )

    return buffer.getvalue(), "tiny.png", "image/png"


def check(label, condition, detail, failures):
    if condition:
        print(f"  [PASS] {label}")
        print(f"         {detail}")
        return True

    print(f"  [FAIL] {label}")
    print(f"         {detail}")
    failures.append(label)
    return False


def cleanup_assessment(assessment_id: str) -> None:
    """
    Remove one test assessment with its children and
    its storage objects.
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
        name for name in all_objects()
        if name.startswith(f"{assessment_id}/")
    ]

    if objects:
        supabase.storage.from_(
            STORAGE_BUCKET
        ).remove(objects)


# ============================================================
# TEST
# ============================================================

def main():
    failures = []
    created_id = None

    print("=" * 66)
    print("STEP 7.13 BACKEND INTEGRATION TEST")
    print("=" * 66)

    before_counts = db_counts()
    before_objects = all_objects()

    print(f"\n[0] Baseline")
    print(f"     db counts : {before_counts}")
    print(f"     storage   : {len(before_objects)} object(s)")

    check(
        "baseline matches the known counts",
        before_counts == BASELINE_COUNTS,
        str(before_counts),
        failures
    )

    # --------------------------------------------------------
    # STEP 1 - BACKEND STARTUP
    # --------------------------------------------------------

    print("\n" + "-" * 66)
    print("STEP 1 - BACKEND STARTUP")
    print("-" * 66)

    # This FastAPI build wraps included routers in an
    # _IncludedRouter object, so app.routes alone does not
    # expose the mounted sub-routes. Collect them from the
    # wrapper and cross-check against the served OpenAPI
    # document, which is the authoritative route contract.

    mounted: set = set()

    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            mounted.add(
                f"{sorted(route.methods)[0]} {route.path}"
            )

        for sub in getattr(route, "original_router", None) and (
            route.original_router.routes
        ) or []:
            methods = getattr(sub, "methods", None)

            if methods:
                mounted.add(
                    f"{sorted(methods)[0]} {sub.path}"
                )

    openapi_routes = {
        f"{method.upper()} {path}"
        for path, operations
        in app.openapi()["paths"].items()
        for method in operations
    }

    routes = mounted | openapi_routes

    print(f"     app title  : {app.title}")
    print(f"     app version: {app.version}")
    print(f"     discovered : {len(routes)} route(s)")

    expected_routes = {
        "GET /",
        "GET /health",
        "POST /api/assessment/upload",
        "POST /api/assessment/analyze",
        "GET /api/assessment/history",
        "GET /api/assessment/{assessment_id}",
    }

    for route in sorted(expected_routes):
        check(
            f"route registered: {route}",
            route in routes,
            route,
            failures
        )

    check(
        "all six documented routes are mounted",
        expected_routes.issubset(routes),
        f"missing = "
        f"{sorted(expected_routes - routes) or 'none'}",
        failures
    )

    client = TestClient(app)

    # --------------------------------------------------------
    # STEP 2 - ROOT AND HEALTH
    # --------------------------------------------------------

    print("\n" + "-" * 66)
    print("STEP 2 - ROOT AND HEALTH")
    print("-" * 66)

    response = client.get("/")

    check(
        "GET / returns 200",
        response.status_code == 200,
        f"{response.status_code} {response.json()}",
        failures
    )

    check(
        "GET / payload is correct",
        response.json() == {
            "message": "AI Disaster Damage Assessment API",
            "status": "running",
        },
        str(response.json()),
        failures
    )

    response = client.get("/health")

    check(
        "GET /health returns 200",
        response.status_code == 200,
        f"{response.status_code} {response.json()}",
        failures
    )

    check(
        "GET /health payload is correct",
        response.json() == {"status": "ok"},
        str(response.json()),
        failures
    )

    before_path, after_path = find_xbd_pair()

    with open(before_path, "rb") as handle:
        before_bytes = handle.read()

    with open(after_path, "rb") as handle:
        after_bytes = handle.read()

    before_name = os.path.basename(before_path)
    after_name = os.path.basename(after_path)

    print(f"\n     real xBD pair:")
    print(f"       before: {before_name} ({len(before_bytes)} bytes)")
    print(f"       after : {after_name} ({len(after_bytes)} bytes)")

    def files(before=before_bytes, after=after_bytes,
              bn=before_name, an=after_name):
        return {
            "before_image": (bn, before, "image/png"),
            "after_image": (an, after, "image/png"),
        }

    # --------------------------------------------------------
    # STEP 3 - UPLOAD ENDPOINT
    # --------------------------------------------------------

    print("\n" + "-" * 66)
    print("STEP 3 - UPLOAD ENDPOINT")
    print("-" * 66)

    with client:
        response = client.post(
            "/api/assessment/upload",
            files=files(),
        )

        check(
            "valid pair accepted",
            response.status_code == 200,
            f"{response.status_code}",
            failures
        )

        upload_payload = response.json()

        check(
            "upload contract unchanged",
            set(upload_payload.keys()) == {
                "status", "message", "before_image",
                "after_image", "dimensions",
            },
            str(sorted(upload_payload.keys())),
            failures
        )

        check(
            "upload reports dimensions",
            upload_payload["dimensions"]["width"] > 0
            and upload_payload["dimensions"]["height"] > 0,
            str(upload_payload["dimensions"]),
            failures
        )

        check(
            "upload returns metadata for both images",
            upload_payload["before_image"]["filename"]
            == before_name
            and upload_payload["after_image"]["filename"]
            == after_name,
            f"{upload_payload['before_image']['filename']}, "
            f"{upload_payload['after_image']['filename']}",
            failures
        )

        print("\n     invalid upload cases:")

        big_bytes, big_name, big_type = oversized_png()

        print(f"       oversized payload: {len(big_bytes)} bytes")

        tiny_bytes, tiny_name, tiny_type = tiny_png()

        cases = [
            (
                "invalid image data",
                files(before=b"this is not an image",
                      after=after_bytes,
                      bn=before_name, an=after_name),
                (400, 422),
            ),
            (
                "unsupported content type",
                files(before=b"x", after=after_bytes,
                      bn="a.gif", an=after_name),
                (400, 422),
            ),
            (
                "oversized image",
                files(before=big_bytes, after=after_bytes,
                      bn=big_name, an=after_name),
                (413,),
            ),
            (
                "image too small",
                files(before=tiny_bytes, after=after_bytes,
                      bn=tiny_name, an=after_name),
                (400,),
            ),
            (
                "mismatched dimensions",
                files(before=before_bytes,
                      after=tiny_bytes,
                      bn=before_name, an=tiny_name),
                (400,),
            ),
        ]

        for label, payload, expected in cases:
            response = client.post(
                "/api/assessment/upload",
                files=payload,
            )

            check(
                f"rejects {label}",
                response.status_code in expected,
                f"{response.status_code} "
                f"(expected {expected})",
                failures
            )

        # ----------------------------------------------------
        # STEP 4 - ANALYZE ENDPOINT
        # ----------------------------------------------------

        print("\n" + "-" * 66)
        print("STEP 4 - ANALYZE ENDPOINT")
        print("-" * 66)

        response = client.post(
            "/api/assessment/analyze",
            files=files(),
        )

        check(
            "analyze returns HTTP 200",
            response.status_code == 200,
            f"{response.status_code}",
            failures
        )

        if response.status_code != 200:
            print(f"         body: {response.text[:400]}")
            print(
                "\nBACKEND INTEGRATION TEST FAILED "
                "(analyze did not return 200)"
            )
            return 1

        analyze = response.json()

        check(
            "analyze contract unchanged",
            set(analyze.keys()) == {
                "status", "message", "before_image",
                "after_image", "prediction", "statistics",
                "model",
            },
            str(sorted(analyze.keys())),
            failures
        )

        check(
            "prediction block unchanged",
            set(analyze["prediction"].keys())
            == {"mask_shape", "predicted_classes"},
            str(sorted(analyze["prediction"].keys())),
            failures
        )

        check(
            "mask shape generated",
            analyze["prediction"]["mask_shape"] == [256, 256],
            str(analyze["prediction"]["mask_shape"]),
            failures
        )

        check(
            "predicted classes generated",
            len(analyze["prediction"]["predicted_classes"]) >= 1,
            str(analyze["prediction"]["predicted_classes"]),
            failures
        )

        stats = analyze["statistics"]

        check(
            "damage level generated",
            isinstance(stats["damage_level"], str)
            and stats["damage_level"] != "",
            stats["damage_level"],
            failures
        )

        check(
            "damage percentage generated",
            0.0 <= stats["damage_percentage"] <= 100.0,
            str(stats["damage_percentage"]),
            failures
        )

        check(
            "class statistics generated",
            set(stats["classes"].keys())
            == set(CLASS_NAMES.values()),
            str(sorted(stats["classes"].keys())),
            failures
        )

        check(
            "all five class rows present",
            len(stats["classes"]) == 5,
            f"{len(stats['classes'])} classes",
            failures
        )

        check(
            "total pixels is 256x256",
            stats["total_pixels"] == 65536,
            str(stats["total_pixels"]),
            failures
        )

        # ----------------------------------------------------
        # STEP 5 - DATABASE PERSISTENCE
        # ----------------------------------------------------

        print("\n" + "-" * 66)
        print("STEP 5 - DATABASE PERSISTENCE")
        print("-" * 66)

        after_counts = db_counts()

        created = (
            after_counts["assessments"] == before_counts["assessments"] + 1
        )

        check(
            "1 new assessment",
            created,
            f"{before_counts['assessments']} -> "
            f"{after_counts['assessments']}",
            failures
        )

        check(
            "2 new assessment_images",
            after_counts["assessment_images"]
            == before_counts["assessment_images"] + 2,
            f"{before_counts['assessment_images']} -> "
            f"{after_counts['assessment_images']}",
            failures
        )

        check(
            "1 new assessment_prediction",
            after_counts["assessment_predictions"]
            == before_counts["assessment_predictions"] + 1,
            f"{before_counts['assessment_predictions']} -> "
            f"{after_counts['assessment_predictions']}",
            failures
        )

        check(
            "5 new assessment_class_statistics",
            after_counts["assessment_class_statistics"]
            == before_counts["assessment_class_statistics"] + 5,
            f"{before_counts['assessment_class_statistics']} -> "
            f"{after_counts['assessment_class_statistics']}",
            failures
        )

        supabase = get_supabase()

        new_row = (
            supabase.table("assessments")
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data[0]
        )

        created_id = new_row["id"]

        print(f"     new assessment_id: {created_id}")

        check(
            "new assessment is not the retained one",
            created_id != RETAINED_ASSESSMENT_ID,
            created_id,
            failures
        )

        check(
            "status = completed",
            new_row["status"] == "completed",
            str(new_row["status"]),
            failures
        )

        registered_model = supabase.table(
            "assessment_models"
        ).select("*").eq(
            "checkpoint", "best_model_change_aware.pth"
        ).execute().data

        check(
            "registered model found",
            len(registered_model) == 1,
            f"{len(registered_model)} model(s)",
            failures
        )

        check(
            "model_id matches the registered model",
            bool(registered_model)
            and new_row["model_id"]
            == registered_model[0]["id"],
            f"db={new_row['model_id']} "
            f"model={registered_model[0]['id'] if registered_model else None}",
            failures
        )

        check(
            "API model name matches the registered model",
            bool(registered_model)
            and analyze["model"]["name"]
            == registered_model[0]["name"],
            f"api={analyze['model']['name']} "
            f"db={registered_model[0]['name'] if registered_model else None}",
            failures
        )

        check(
            "API checkpoint matches the registered model",
            bool(registered_model)
            and analyze["model"]["checkpoint"]
            == registered_model[0]["checkpoint"],
            f"api={analyze['model']['checkpoint']} "
            f"db={registered_model[0]['checkpoint'] if registered_model else None}",
            failures
        )

        check(
            "damage level matches API response",
            new_row["damage_level"] == stats["damage_level"],
            f"db={new_row['damage_level']} "
            f"api={stats['damage_level']}",
            failures
        )

        check(
            "damage percentage matches API response",
            abs(
                float(new_row["damage_percentage"])
                - float(stats["damage_percentage"])
            ) < 0.01,
            f"db={new_row['damage_percentage']} "
            f"api={stats['damage_percentage']}",
            failures
        )

        check(
            "total and damage pixels match API response",
            new_row["total_pixels"] == stats["total_pixels"]
            and new_row["damage_pixels"]
            == stats["damage_pixels"],
            f"db total={new_row['total_pixels']} "
            f"damage={new_row['damage_pixels']}",
            failures
        )

        db_stats = supabase.table(
            "assessment_class_statistics"
        ).select("*").eq(
            "assessment_id", created_id
        ).execute().data

        check(
            "5 class statistic rows for the new assessment",
            len(db_stats) == 5,
            f"{len(db_stats)} rows",
            failures
        )

        class_match = True
        for row in db_stats:
            name = row["class_name"]
            api_entry = stats["classes"].get(name)

            if api_entry is None:
                class_match = False
                break

            if (
                row["pixel_count"] != api_entry["pixels"]
                or abs(
                    float(row["percentage"])
                    - float(api_entry["percentage"])
                ) > 0.01
            ):
                class_match = False
                break

        check(
            "class statistics match API response",
            class_match,
            "pixel counts and percentages agree for "
            "all five classes",
            failures
        )

        db_images = supabase.table(
            "assessment_images"
        ).select("*").eq(
            "assessment_id", created_id
        ).execute().data

        check(
            "2 image rows for the new assessment",
            len(db_images) == 2,
            f"{len(db_images)} rows",
            failures
        )

        image_match = True
        for row in db_images:
            side = "before_image" if row["image_type"] == "before" else "after_image"

            if (
                row["filename"] != analyze[side]["filename"]
                or row["width"] != analyze[side]["width"]
                or row["height"] != analyze[side]["height"]
                or row["size_bytes"]
                != analyze[side]["size_bytes"]
            ):
                image_match = False

        check(
            "image metadata matches API response",
            image_match,
            "filename, width, height and size agree "
            "for both images",
            failures
        )

        # Since Step 8.2 the analyze flow stores the before
        # and after bytes in the private bucket and records the
        # paths on the assessment_images rows, so a historical
        # assessment stays reproducible and visually
        # inspectable. The Step 8.2 asset endpoint signs those
        # paths on demand.

        image_paths = {
            row["image_type"]: row["storage_path"]
            for row in db_images
        }

        print(
            f"     image storage_path values: "
            f"{image_paths}"
        )

        check(
            "before image path is stored and canonical",
            image_paths.get("before")
            == f"{STORAGE_BUCKET}/{created_id}/before.png",
            str(image_paths.get("before")),
            failures
        )

        check(
            "after image path is stored and canonical",
            image_paths.get("after")
            == f"{STORAGE_BUCKET}/{created_id}/after.png",
            str(image_paths.get("after")),
            failures
        )

        stored_for_assessment = [
            name for name in all_objects()
            if name.startswith(f"{created_id}/")
        ]

        check(
            "before/after objects exist for the assessment",
            {
                f"{created_id}/before.png",
                f"{created_id}/after.png",
            }.issubset(set(stored_for_assessment)),
            str(stored_for_assessment),
            failures
        )

        check(
            "exactly one before and one after object, "
            "no duplicates",
            sorted(stored_for_assessment) == [
                f"{created_id}/after.png",
                f"{created_id}/before.png",
                f"{created_id}/{MASK_FILENAME}",
            ],
            str(stored_for_assessment),
            failures
        )

        # ----------------------------------------------------
        # STEP 6 - PREDICTION MASK
        # ----------------------------------------------------

        print("\n" + "-" * 66)
        print("STEP 6 - PREDICTION MASK")
        print("-" * 66)

        db_prediction = supabase.table(
            "assessment_predictions"
        ).select("*").eq(
            "assessment_id", created_id
        ).execute().data

        check(
            "1 prediction row for the new assessment",
            len(db_prediction) == 1,
            f"{len(db_prediction)} rows",
            failures
        )

        prediction = db_prediction[0]
        mask_path = prediction["mask_storage_path"]

        check(
            "mask path follows the required pattern",
            mask_path
            == f"assessments/{created_id}/{MASK_FILENAME}",
            str(mask_path),
            failures
        )

        bucket = supabase.storage.from_(
            STORAGE_BUCKET
        )

        masks = [
            name for name in all_objects()
            if name.endswith(MASK_FILENAME)
        ]

        check(
            "exactly one mask, no duplicate",
            masks == [f"{created_id}/{MASK_FILENAME}"],
            str(masks),
            failures
        )

        check(
            "mask object exists",
            bucket.exists(f"{created_id}/{MASK_FILENAME}"),
            f"{created_id}/{MASK_FILENAME}",
            failures
        )

        mask_bytes = bucket.download(
            f"{created_id}/{MASK_FILENAME}"
        )

        check(
            "mask is a real PNG",
            mask_bytes.startswith(PNG_MAGIC),
            repr(mask_bytes[:8]),
            failures
        )

        mask_image = Image.open(io.BytesIO(mask_bytes))

        check(
            "mask dimensions are 256x256",
            mask_image.size == (256, 256),
            str(mask_image.size),
            failures
        )

        check(
            "mask mode is L",
            mask_image.mode == "L",
            mask_image.mode,
            failures
        )

        decoded = np.array(mask_image)
        unique = sorted(
            int(value) for value in np.unique(decoded)
        )

        check(
            "mask values are class IDs 0-4",
            all(0 <= value <= 4 for value in unique),
            f"unique = {unique}",
            failures
        )

        check(
            "mask is not rescaled to 0-255",
            max(unique) <= 4,
            f"max value = {max(unique)}",
            failures
        )

        check(
            "mask corresponds to the prediction",
            set(unique)
            == set(analyze["prediction"]["predicted_classes"]),
            f"mask={unique} "
            f"api={analyze['prediction']['predicted_classes']}",
            failures
        )

        check(
            "mask dimensions stored correctly",
            prediction["mask_width"] == 256
            and prediction["mask_height"] == 256,
            f"{prediction['mask_width']}x"
            f"{prediction['mask_height']}",
            failures
        )

        bucket_meta = [
            b for b in supabase.storage.list_buckets()
            if b.name == STORAGE_BUCKET
        ][0]

        check(
            "bucket remains private",
            bucket_meta.public is False,
            f"public = {bucket_meta.public}",
            failures
        )

        # ----------------------------------------------------
        # STEP 7 - GET HISTORY
        # ----------------------------------------------------

        print("\n" + "-" * 66)
        print("STEP 7 - GET HISTORY")
        print("-" * 66)

        response = client.get(
            "/api/assessment/history"
        )

        check(
            "history returns HTTP 200",
            response.status_code == 200,
            f"{response.status_code}",
            failures
        )

        history = response.json()

        check(
            "history contract unchanged",
            set(history.keys()) == {
                "status", "count", "limit", "assessments",
            },
            str(sorted(history.keys())),
            failures
        )

        check(
            "history count matches list length",
            history["count"]
            == len(history["assessments"]),
            f"count={history['count']} "
            f"len={len(history['assessments'])}",
            failures
        )

        history_ids = [
            item["id"] for item in history["assessments"]
        ]

        check(
            "new assessment appears in history",
            created_id in history_ids,
            f"{len(history_ids)} item(s)",
            failures
        )

        check(
            "newest-first ordering works",
            history_ids
            and history_ids[0] == created_id,
            f"first = {history_ids[0] if history_ids else None}",
            failures
        )

        first_item = history["assessments"][0]

        check(
            "history model information present",
            bool(
                embedded(
                    first_item.get("assessment_models")
                ).get("name")
            ),
            str(
                embedded(
                    first_item.get("assessment_models")
                ).get("name")
            ),
            failures
        )

        check(
            "history damage percentage present",
            first_item["damage_percentage"]
            == stats["damage_percentage"],
            str(first_item["damage_percentage"]),
            failures
        )

        check(
            "history leaks no report path",
            "report_storage_path" not in first_item,
            "report_storage_path absent from history item",
            failures
        )

        check(
            "history leaks no storage paths",
            "storage_path" not in first_item
            and "mask_storage_path" not in first_item,
            "no internal storage path exposed",
            failures
        )

        response = client.get(
            "/api/assessment/history?limit=1"
        )

        check(
            "history limit parameter works",
            response.status_code == 200
            and response.json()["count"] == 1,
            f"count={response.json().get('count')}",
            failures
        )

        # ----------------------------------------------------
        # STEP 8 - GET ASSESSMENT BY ID
        # ----------------------------------------------------

        print("\n" + "-" * 66)
        print("STEP 8 - GET ASSESSMENT BY ID")
        print("-" * 66)

        response = client.get(
            f"/api/assessment/{created_id}"
        )

        check(
            "by-id returns HTTP 200",
            response.status_code == 200,
            f"{response.status_code}",
            failures
        )

        detail = response.json()

        check(
            "detail contract unchanged",
            set(detail.keys()) == {
                "id", "user_id", "model_id", "status",
                "damage_level", "damage_percentage",
                "total_pixels", "damage_pixels",
                "created_at", "updated_at",
                "assessment_models", "assessment_images",
                "assessment_predictions",
                "assessment_class_statistics",
            },
            str(sorted(detail.keys())),
            failures
        )

        check(
            "detail returns the model",
            bool(
                embedded(
                    detail.get("assessment_models")
                ).get("name")
            ),
            str(
                embedded(
                    detail.get("assessment_models")
                ).get("name")
            ),
            failures
        )

        detail_images = embedded_list(
            detail.get("assessment_images")
        )

        check(
            "detail returns 2 images",
            len(detail_images) == 2,
            f"{len(detail_images)} images",
            failures
        )

        detail_prediction = embedded(
            detail.get("assessment_predictions")
        )

        check(
            "detail returns 1 prediction",
            bool(detail_prediction.get("id")),
            str(detail_prediction.get("id")),
            failures
        )

        detail_stats = embedded_list(
            detail.get("assessment_class_statistics")
        )

        check(
            "detail returns 5 class statistics",
            len(detail_stats) == 5,
            f"{len(detail_stats)} rows",
            failures
        )

        check(
            "detail values match POST /analyze",
            detail["damage_level"] == stats["damage_level"]
            and detail["total_pixels"] == stats["total_pixels"]
            and detail["damage_pixels"]
            == stats["damage_pixels"],
            f"level={detail['damage_level']} "
            f"total={detail['total_pixels']} "
            f"damage={detail['damage_pixels']}",
            failures
        )

        check(
            "detail predicted classes match POST",
            detail_prediction.get("predicted_classes")
            == analyze["prediction"]["predicted_classes"],
            str(
                detail_prediction.get("predicted_classes")
            ),
            failures
        )

        check(
            "detail leaks no report path",
            "report_storage_path" not in detail,
            "report_storage_path absent from detail",
            failures
        )

        missing_uuid = str(uuid.uuid4())

        response = client.get(
            f"/api/assessment/{missing_uuid}"
        )

        check(
            "valid but nonexistent UUID returns 404",
            response.status_code == 404,
            f"{response.status_code} for {missing_uuid}",
            failures
        )

        response = client.get(
            "/api/assessment/not-a-valid-uuid"
        )

        check(
            "malformed UUID returns 404",
            response.status_code == 404,
            f"{response.status_code}",
            failures
        )

        # ----------------------------------------------------
        # STEP 9 - ERROR HANDLING
        # ----------------------------------------------------

        print("\n" + "-" * 66)
        print("STEP 9 - ERROR HANDLING")
        print("-" * 66)

        error_matrix = []

        error_matrix.append((
            "malformed UUID",
            client.get("/api/assessment/abc-123"),
            (404, 422),
        ))

        error_matrix.append((
            "empty id segment",
            client.get("/api/assessment/"),
            (404, 405),
        ))

        error_matrix.append((
            "valid nonexistent UUID",
            client.get(
                f"/api/assessment/{uuid.uuid4()}"
            ),
            (404,),
        ))

        error_matrix.append((
            "invalid image upload",
            client.post(
                "/api/assessment/upload",
                files=files(before=b"not an image"),
            ),
            (400, 422),
        ))

        error_matrix.append((
            "unsupported image type",
            client.post(
                "/api/assessment/upload",
                files=files(before=b"x", bn="a.gif"),
            ),
            (400, 422),
        ))

        error_matrix.append((
            "oversized image",
            client.post(
                "/api/assessment/upload",
                files=files(
                    before=big_bytes, bn=big_name
                ),
            ),
            (413,),
        ))

        error_matrix.append((
            "analyze with invalid image",
            client.post(
                "/api/assessment/analyze",
                files=files(before=b"not an image"),
            ),
            (400, 422),
        ))

        for label, response, expected in error_matrix:
            check(
                f"{label} -> {expected}",
                response.status_code in expected,
                f"actual {response.status_code}",
                failures
            )

        check(
            "history rejects an invalid limit",
            client.get(
                "/api/assessment/history?limit=0"
            ).status_code == 422,
            str(
                client.get(
                    "/api/assessment/history?limit=0"
                ).status_code
            ),
            failures
        )

    # ------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------

    print("\n" + "-" * 66)
    print("CLEANUP AND BASELINE RESTORE")
    print("-" * 66)

    if created_id:
        cleanup_assessment(created_id)
        print(f"     removed assessment {created_id}")

    remaining = all_objects()

    if remaining != before_objects:
        supabase = get_supabase()

        extra = [
            name for name in remaining
            if name not in before_objects
        ]

        if extra:
            supabase.storage.from_(
                STORAGE_BUCKET
            ).remove(extra)

    final_counts = db_counts()
    final_objects = all_objects()

    check(
        "database restored to baseline",
        final_counts == BASELINE_COUNTS,
        str(final_counts),
        failures
    )

    check(
        "storage restored to baseline",
        final_objects == before_objects,
        f"{final_objects}",
        failures
    )

    retained = supabase.table(
        "assessments"
    ).select("*").eq(
        "id", RETAINED_ASSESSMENT_ID
    ).execute().data

    check(
        "verified assessment still present",
        bool(retained),
        RETAINED_ASSESSMENT_ID,
        failures
    )

    if retained:
        check(
            "verified assessment status = completed",
            retained[0]["status"] == "completed",
            str(retained[0]["status"]),
            failures
        )

        check(
            "verified report_storage_path = NULL",
            retained[0]["report_storage_path"] is None,
            str(retained[0]["report_storage_path"]),
            failures
        )

    retained_mask = supabase.table(
        "assessment_predictions"
    ).select("mask_storage_path").eq(
        "assessment_id", RETAINED_ASSESSMENT_ID
    ).execute().data

    check(
        "verified mask_storage_path = NULL",
        retained_mask
        and retained_mask[0]["mask_storage_path"] is None,
        str(
            retained_mask[0]["mask_storage_path"]
            if retained_mask
            else None
        ),
        failures
    )

    check(
        "registered model still present",
        final_counts["assessment_models"] == 1,
        f"{final_counts['assessment_models']} model(s)",
        failures
    )

    print("\n" + "=" * 66)

    if failures:
        print(
            f"BACKEND INTEGRATION TEST FAILED: "
            f"{len(failures)} failure(s)"
        )
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("BACKEND INTEGRATION TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
