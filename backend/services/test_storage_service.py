import glob
import os
import uuid

from dotenv import load_dotenv

from backend.services.storage_service import (
    STORAGE_BUCKET,
    MAX_IMAGE_SIZE,
    upload_assessment_image,
    delete_assessment_image,
    get_assessment_image_path,
    get_assessment_image_info
)

from backend.services.supabase_service import get_supabase


load_dotenv()


# ============================================================
# TEST CONFIGURATION
# ============================================================

# The xBD dataset lives outside the repository, so the
# root is overridable through XBD_IMAGE_DIR.

XBD_IMAGE_DIR = os.getenv(
    "XBD_IMAGE_DIR",
    r"D:\Projects\Datasets\xBD\train\images"
)


# ============================================================
# HELPERS
# ============================================================

def find_smallest_xbd_png() -> str:
    """
    Return the path of the smallest pre-disaster PNG
    in the xBD training image directory, so the test
    moves as little data as possible.
    """

    if not os.path.isdir(XBD_IMAGE_DIR):
        raise RuntimeError(
            f"xBD image directory not found: "
            f"{XBD_IMAGE_DIR}"
        )

    pattern = os.path.join(
        XBD_IMAGE_DIR,
        "*_pre_disaster.png"
    )

    candidates = glob.glob(pattern)

    if not candidates:
        raise RuntimeError(
            f"No xBD pre-disaster PNGs found in: "
            f"{XBD_IMAGE_DIR}"
        )

    return min(
        candidates,
        key=os.path.getsize
    )


def bucket_object_count(assessment_id: str) -> int:
    """
    Count the objects currently stored under one
    test assessment prefix.
    """

    bucket = get_supabase().storage.from_(
        STORAGE_BUCKET
    )

    entries = bucket.list(
        f"{assessment_id}"
    )

    return len(entries or [])


# ============================================================
# TEST
# ============================================================

def main():
    failures = []

    def check(
        label: str,
        condition: bool,
        detail: str = ""
    ) -> None:
        status = "PASS" if condition else "FAIL"

        print(f"  [{status}] {label}")

        if detail:
            print(f"         {detail}")

        if not condition:
            failures.append(label)

    # ---------------------------------------------------------
    # 0. Baseline
    # ---------------------------------------------------------

    print("=" * 66)
    print("STEP 7.10 STORAGE SERVICE TEST")
    print("=" * 66)
    print()

    print("[0] Read one small valid PNG from the xBD dataset")

    source_path = find_smallest_xbd_png()

    with open(source_path, "rb") as handle:
        image_bytes = handle.read()

    content_type = "image/png"

    print(f"     source : {os.path.basename(source_path)}")
    print(f"     bytes  : {len(image_bytes)}")
    print()

    check(
        "source image is non-empty",
        len(image_bytes) > 0
    )
    check(
        "source image is within the 10 MB limit",
        len(image_bytes) <= MAX_IMAGE_SIZE
    )

    # A random UUID gives a unique path without ever
    # inserting an assessments row.

    test_id = f"test-{uuid.uuid4()}"

    print()
    print("[1] Temporary test id created (no database insert)")
    print(f"     test_id : {test_id}")
    print()

    check(
        "bucket prefix present in test id",
        test_id.startswith("test-")
    )

    # ---------------------------------------------------------
    # 2. Expected path, no public URL
    # ---------------------------------------------------------

    print("[2] Path helper returns a private path")

    expected_path = (
        f"{STORAGE_BUCKET}/"
        f"{test_id}/before.png"
    )

    path = get_assessment_image_path(
        test_id,
        "before",
        content_type
    )

    print(f"     path    : {path}")
    print()

    check(
        "path is correct",
        path == expected_path,
        f"expected {expected_path}"
    )
    check(
        "path is not a URL",
        not path.startswith("http")
    )
    check(
        "path carries no query string or signature",
        "?" not in path and "token" not in path
    )

    # ---------------------------------------------------------
    # 3. Upload
    # ---------------------------------------------------------

    print("[3] Upload to the private assessments bucket")

    uploaded_path = upload_assessment_image(
        assessment_id=test_id,
        image_type="before",
        image_bytes=image_bytes,
        content_type=content_type
    )

    print(f"     uploaded: {uploaded_path}")
    print()

    check(
        "upload returned the expected path",
        uploaded_path == expected_path
    )
    check(
        "object exists in the bucket",
        bucket_object_count(test_id) == 1
    )

    info = get_assessment_image_info(
        uploaded_path
    )

    stored_size = info.get("size") or info.get(
        "Size"
    )

    print(f"     stored  : {stored_size} bytes")
    print()

    check(
        "stored byte count matches the source",
        stored_size == len(image_bytes),
        f"stored {stored_size}, source {len(image_bytes)}"
    )

    # ---------------------------------------------------------
    # 4. No overwrite
    # ---------------------------------------------------------

    print("[4] Re-upload is rejected (upsert disabled)")

    overwrite_blocked = False

    try:
        upload_assessment_image(
            assessment_id=test_id,
            image_type="before",
            image_bytes=image_bytes,
            content_type=content_type
        )
    except RuntimeError:
        overwrite_blocked = True

    check(
        "existing file was not overwritten",
        overwrite_blocked
    )

    # ---------------------------------------------------------
    # 5. Validation
    # ---------------------------------------------------------

    print("[5] Input validation")

    for label, kwargs in (
        ("bad image_type", {
            "assessment_id": test_id,
            "image_type": "sideways",
            "image_bytes": image_bytes,
            "content_type": content_type,
        }),
        ("bad content_type", {
            "assessment_id": test_id,
            "image_type": "before",
            "image_bytes": image_bytes,
            "content_type": "image/gif",
        }),
        ("empty bytes", {
            "assessment_id": test_id,
            "image_type": "before",
            "image_bytes": b"",
            "content_type": content_type,
        }),
    ):
        raised = None

        try:
            upload_assessment_image(**kwargs)
        except ValueError as error:
            raised = type(error).__name__
        except RuntimeError:
            raised = "RuntimeError"

        check(
            f"rejects {label}",
            raised is not None,
            f"raised {raised}"
        )

    oversize_blocked = False

    try:
        upload_assessment_image(
            assessment_id=test_id,
            image_type="after",
            image_bytes=b"\x00" * (
                MAX_IMAGE_SIZE + 1
            ),
            content_type=content_type
        )
    except ValueError:
        oversize_blocked = True

    check(
        "rejects payload over 10 MB",
        oversize_blocked
    )

    # ---------------------------------------------------------
    # 6. Delete
    # ---------------------------------------------------------

    print("[6] Delete the uploaded test file")

    delete_assessment_image(uploaded_path)

    check(
        "object gone from the bucket",
        bucket_object_count(test_id) == 0
    )

    print()

    # ---------------------------------------------------------
    # 7. Cleanup
    # ---------------------------------------------------------

    print("[7] No test files left behind")

    remaining = bucket_object_count(test_id)

    check(
        "test prefix is empty",
        remaining == 0,
        f"{remaining} object(s) left"
    )

    stray = [
        entry
        for entry in (
            get_supabase()
            .storage
            .from_(STORAGE_BUCKET)
            .list()
        )
        if "test-" in str(entry.get("name", ""))
    ]

    check(
        "no test- prefixed objects anywhere in the bucket",
        len(stray) == 0,
        f"stray: {[e.get('name') for e in stray]}"
    )

    print()
    print("=" * 66)

    if failures:
        print(
            f"FAILED: {len(failures)} check(s) failed"
        )
        for item in failures:
            print(f"  - {item}")
        print("=" * 66)
        raise SystemExit(1)

    print("STORAGE SERVICE TEST PASSED")
    print("=" * 66)


if __name__ == "__main__":
    main()
