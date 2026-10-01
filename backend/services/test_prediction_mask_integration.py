import glob
import io
import os
import uuid

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.supabase_service import get_supabase
from backend.services.storage_service import (
    STORAGE_BUCKET,
    upload_assessment_image,
    delete_assessment_image,
    get_assessment_image_path,
    _storage_bucket
)
from backend.services.auth_service import (
    AuthenticatedUser,
    get_current_user,
)


load_dotenv()


# ============================================================
# TEST CONFIGURATION
# ============================================================

XBD_IMAGE_DIR = os.getenv(
    "XBD_IMAGE_DIR",
    r"D:\Projects\Datasets\xBD\train\images"
)

TABLES = (
    "assessments",
    "assessment_images",
    "assessment_predictions",
    "assessment_class_statistics",
    "assessment_models",
)


# ============================================================
# HELPERS
# ============================================================

def find_xbd_pair() -> tuple:
    """
    Return the smallest real xBD before/after PNG
    pair, so the test transfers as little as possible.
    """

    if not os.path.isdir(XBD_IMAGE_DIR):
        raise RuntimeError(
            f"xBD image directory not found: "
            f"{XBD_IMAGE_DIR}"
        )

    candidates = glob.glob(
        os.path.join(
            XBD_IMAGE_DIR,
            "*_pre_disaster.png"
        )
    )

    if not candidates:
        raise RuntimeError(
            f"No xBD pre-disaster PNGs found in: "
            f"{XBD_IMAGE_DIR}"
        )

    before = min(candidates, key=os.path.getsize)

    after = before.replace(
        "_pre_disaster.png", "_post_disaster.png"
    )

    if not os.path.isfile(after):
        raise RuntimeError(
            f"No matching post-disaster image for: "
            f"{os.path.basename(before)}"
        )

    return before, after


def snapshot_state() -> dict:
    """
    Capture the current database and Storage state.
    """

    supabase = get_supabase()

    counts = {
        table: supabase.table(table)
        .select("id", count="exact")
        .execute()
        .count
        for table in TABLES
    }

    objects = supabase.storage.from_(
        STORAGE_BUCKET
    ).list()

    return {
        "counts": counts,
        "objects": all_bucket_objects()
    }


def bucket_objects(assessment_id: str) -> list:
    """
    Return the full bucket paths of every object
    stored under one assessment prefix.

    list(prefix) reports names relative to that
    prefix, while remove() needs full paths, so the
    prefix is restored here.
    """

    entries = get_supabase().storage.from_(
        STORAGE_BUCKET
    ).list(assessment_id)

    return [
        f"{assessment_id}/{entry.get('name', '')}"
        for entry in (entries or [])
    ]


def all_bucket_objects() -> list:
    """
    Return the full path of every object in the
    bucket, at any depth.
    """

    result = get_supabase().storage.from_(
        STORAGE_BUCKET
    ).list_v2(options={"prefix": "", "limit": 1000})

    return sorted(
        str(obj.name)
        for obj in (result.objects or [])
    )


def purge_assessment(assessment_id: str) -> None:
    """
    Remove one assessment and every Storage object
    created for it.
    """

    supabase = get_supabase()

    for table in (
        "assessment_class_statistics",
        "assessment_predictions",
        "assessment_images",
    ):
        (
            supabase.table(table)
            .delete()
            .eq("assessment_id", assessment_id)
            .execute()
        )

    (
        supabase.table("assessments")
        .delete()
        .eq("id", assessment_id)
        .execute()
    )

    remaining = bucket_objects(assessment_id)

    if remaining:
        supabase.storage.from_(
            STORAGE_BUCKET
        ).remove(remaining)

    still_there = bucket_objects(assessment_id)

    if still_there:
        raise RuntimeError(
            "Storage cleanup incomplete, "
            f"still present: {still_there}"
        )


# ============================================================
# TEST
# ============================================================

def main():
    failures = []

    def check(label, condition, detail=""):
        status = "PASS" if condition else "FAIL"

        print(f"  [{status}] {label}")

        if detail:
            print(f"         {detail}")

        if not condition:
            failures.append(label)

    print("=" * 66)
    print("STEP 7.11 INTEGRATION TEST")
    print("=" * 66)
    print()

    supabase = get_supabase()

    # ---------------------------------------------------------
    # 0. Baseline
    # ---------------------------------------------------------

    print("[0] Capture state before the test")

    before_state = snapshot_state()

    print(f"     db counts : {before_state['counts']}")
    print(f"     storage   : {before_state['objects']}")
    print()

    assessment_id = None
    caller_user_id = None

    try:
        # -----------------------------------------------------
        # 1. Real xBD pair
        # -----------------------------------------------------

        print("[1] Use a real xBD before/after pair")

        before_path, after_path = find_xbd_pair()

        with open(before_path, "rb") as handle:
            before_bytes = handle.read()

        with open(after_path, "rb") as handle:
            after_bytes = handle.read()

        print(f"     before : {os.path.basename(before_path)} ({len(before_bytes)} bytes)")
        print(f"     after  : {os.path.basename(after_path)} ({len(after_bytes)} bytes)")
        print()

        with Image.open(io.BytesIO(before_bytes)) as a, \
                Image.open(io.BytesIO(after_bytes)) as b:
            check(
                "pair has matching dimensions",
                a.size == b.size,
                f"{a.size} vs {b.size}"
            )

        # -----------------------------------------------------
        # 2. POST /analyze
        # -----------------------------------------------------

        print("[2] POST /api/assessment/analyze")

        # /analyze is authenticated now. A real Supabase Auth
        # user is created and installed as the verified caller
        # so the ownership path production uses is exercised
        # instead of bypassed.

        caller_email = f"step711-{uuid.uuid4()}@example.invalid"

        caller_user_id = supabase.auth.admin.create_user({
            "email": caller_email,
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
        }).user.id

        app.dependency_overrides[get_current_user] = (
            lambda: AuthenticatedUser(
                user_id=caller_user_id,
                email=caller_email,
            )
        )

        with TestClient(app) as client:
            response = client.post(
                "/api/assessment/analyze",
                files={
                    "before_image": (
                        os.path.basename(before_path),
                        before_bytes,
                        "image/png"
                    ),
                    "after_image": (
                        os.path.basename(after_path),
                        after_bytes,
                        "image/png"
                    ),
                }
            )

        print(f"     HTTP {response.status_code}")
        print()

        check(
            "HTTP 200",
            response.status_code == 200,
            response.text[:200]
        )

        if response.status_code != 200:
            raise RuntimeError(
                "analyze request failed; skipping "
                "dependent checks"
            )

        payload = response.json()

        api_classes = payload["prediction"][
            "predicted_classes"
        ]

        api_mask_shape = payload["prediction"][
            "mask_shape"
        ]

        print(f"     predicted_classes : {api_classes}")
        print(f"     mask_shape        : {api_mask_shape}")
        print()

        # Newest row is the one just created
        row = (
            supabase.table("assessments")
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data[0]
        )

        assessment_id = row["id"]

        print(f"     new assessment_id : {assessment_id}")
        print()

        # -----------------------------------------------------
        # 3. Response contract
        # -----------------------------------------------------

        print("[3] Response contract is unchanged")

        check(
            "response has no mask_storage_path leaked",
            "mask_storage_path" not in str(payload)
        )
        check(
            "prediction block has exactly 2 keys",
            set(payload["prediction"].keys())
            == {"mask_shape", "predicted_classes"},
            str(sorted(payload["prediction"].keys()))
        )

        # -----------------------------------------------------
        # 4. assessment_predictions
        # -----------------------------------------------------

        print("[4] assessment_predictions")

        predictions = (
            supabase.table("assessment_predictions")
            .select("*")
            .eq("assessment_id", assessment_id)
            .execute()
            .data
        )

        check(
            "exactly one row",
            len(predictions) == 1,
            f"got {len(predictions)}"
        )

        if len(predictions) != 1:
            raise RuntimeError(
                "expected exactly one prediction row"
            )

        prediction = predictions[0]

        check(
            "mask_width is 256",
            prediction["mask_width"] == 256,
            str(prediction["mask_width"])
        )
        check(
            "mask_height is 256",
            prediction["mask_height"] == 256,
            str(prediction["mask_height"])
        )
        check(
            "predicted_classes matches the API response",
            sorted(prediction["predicted_classes"])
            == sorted(api_classes),
            f"db={sorted(prediction['predicted_classes'])} "
            f"api={sorted(api_classes)}"
        )
        check(
            "mask_storage_path is NOT NULL",
            prediction["mask_storage_path"] is not None,
            str(prediction["mask_storage_path"])
        )
        print(
            f"         mask_storage_path = "
            f"{prediction['mask_storage_path']}"
        )
        print()

        # -----------------------------------------------------
        # 5. Storage object
        # -----------------------------------------------------

        print("[5] Storage object")

        stored_path = prediction["mask_storage_path"]

        check(
            "path is bucket-qualified as specified",
            stored_path
            == f"{STORAGE_BUCKET}/{assessment_id}/prediction-mask.png",
            stored_path
        )

        objects = bucket_objects(assessment_id)

        print(f"     objects under test prefix : {objects}")
        print()

        # Since Step 8.2 the analyze flow also stores the
        # before/after bytes under the same prefix, so the
        # no-duplicate guarantee is asserted against the mask
        # objects specifically rather than the whole prefix.

        mask_name = f"{assessment_id}/prediction-mask.png"

        mask_objects = [
            name for name in objects
            if name.endswith("prediction-mask.png")
        ]

        check(
            "exactly one mask object stored (no duplicates)",
            len(mask_objects) == 1,
            f"got {len(mask_objects)}"
        )
        check(
            "the single mask object is the expected mask",
            len(mask_objects) == 1
            and mask_objects[0] == mask_name,
            str(mask_objects)
        )
        check(
            "before/after objects share the same canonical prefix",
            sorted(objects) == [
                f"{assessment_id}/after.png",
                f"{assessment_id}/before.png",
                mask_name,
            ],
            str(sorted(objects))
        )

        # -----------------------------------------------------
        # 6. Downloaded object
        # -----------------------------------------------------

        print("[6] Downloaded object is a valid 256x256 PNG")

        downloaded = (
            supabase.storage.from_(STORAGE_BUCKET)
            .download(
                f"{assessment_id}/prediction-mask.png"
            )
        )

        print(f"     downloaded bytes : {len(downloaded)}")
        print()

        check(
            "PNG magic signature present",
            downloaded[:8] == b"\x89PNG\r\n\x1a\n"
        )
        check(
            "not a JPEG",
            downloaded[:3] != b"\xff\xd8\xff"
        )

        with Image.open(io.BytesIO(downloaded)) as image:
            mask_w, mask_h = image.size
            mask_mode = image.mode
            decoded = np.array(image)

        print(f"     size  : {mask_w}x{mask_h}")
        print(f"     mode  : {mask_mode}")
        print(f"     unique ids : {sorted(set(decoded.flatten().tolist()))}")
        print()

        check(
            "downloaded PNG is 256x256",
            mask_w == 256 and mask_h == 256
        )
        check(
            "mode is 8-bit grayscale",
            mask_mode == "L",
            str(mask_mode)
        )
        check(
            "decoded contains only class IDs 0-4",
            set(decoded.flatten().tolist())
            .issubset({0, 1, 2, 3, 4})
        )
        check(
            "decoded unique IDs match the response classes",
            sorted(set(decoded.flatten().tolist()))
            == sorted(api_classes),
            f"decoded={sorted(set(decoded.flatten().tolist()))} "
            f"api={sorted(api_classes)}"
        )

        # -----------------------------------------------------
        # 7. Before/after storage still works
        # -----------------------------------------------------

        print("[7] Existing before/after storage still works")

        probe_path = get_assessment_image_path(
            assessment_id, "before", "image/png"
        )

        # Since Step 8.2 the analyze flow already stored
        # before/after.png under this prefix, so uploading the
        # same path again must be refused rather than
        # silently overwriting the stored bytes.

        # storage3 download() takes a bucket-relative path,
        # while probe_path is bucket-qualified.

        probe_relative = probe_path.split("/", 1)[1]

        original_before_bytes = _storage_bucket().download(
            probe_relative
        )[1]

        refused = False

        try:
            upload_assessment_image(
                assessment_id=assessment_id,
                image_type="before",
                image_bytes=before_bytes,
                content_type="image/png"
            )
        except Exception:
            refused = True

        check(
            "re-uploading an existing before.png is refused",
            refused,
            "upsert=False rejected the duplicate"
        )

        check(
            "stored before.png bytes were not overwritten",
            _storage_bucket().download(probe_relative)[1]
            == original_before_bytes,
            "stored bytes unchanged"
        )

        after_upload = bucket_objects(assessment_id)

        check(
            "before.png present alongside the mask",
            len(after_upload) == 3
            and probe_relative in after_upload,
            f"got {after_upload}"
        )

        delete_assessment_image(probe_path)

        after_delete = bucket_objects(assessment_id)

        check(
            "before.png deleted, mask retained",
            len(after_delete) == 2
            and any(
                name.endswith("prediction-mask.png")
                for name in after_delete
            )
            and not any(
                name.endswith("before.png")
                for name in after_delete
            ),
            f"got {after_delete}"
        )

    finally:
        # -----------------------------------------------------
        # 8. Cleanup
        # -----------------------------------------------------

        print()
        print("[8] Cleanup")

        if assessment_id:
            purge_assessment(assessment_id)
            print(f"     purged assessment {assessment_id}")
        else:
            print("     no assessment was created")

        app.dependency_overrides.pop(get_current_user, None)

        if caller_user_id:
            try:
                supabase.table("users").delete().eq(
                    "id", caller_user_id
                ).execute()
                supabase.auth.admin.delete_user(
                    caller_user_id
                )
                print("     removed test caller user")
            except Exception as error:
                print(f"     note: caller cleanup: {error}")

        after_state = snapshot_state()

        print(f"     db counts : {after_state['counts']}")
        print(f"     storage   : {after_state['objects']}")
        print()

        check(
            "database state restored",
            after_state["counts"] == before_state["counts"],
            f"before={before_state['counts']} "
            f"after={after_state['counts']}"
        )
        check(
            "storage state restored",
            after_state["objects"] == before_state["objects"],
            f"before={before_state['objects']} "
            f"after={after_state['objects']}"
        )

    print()
    print("=" * 66)

    if failures:
        print(f"FAILED: {len(failures)} check(s) failed")

        for item in failures:
            print(f"  - {item}")

        print("=" * 66)
        raise SystemExit(1)

    print("INTEGRATION TEST PASSED")
    print("=" * 66)


if __name__ == "__main__":
    main()
