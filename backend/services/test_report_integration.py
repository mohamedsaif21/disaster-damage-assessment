"""
Step 7.12 integration test.

Generates a PDF report from the existing verified
assessment, uploads it to the private assessments
bucket, verifies it, then restores the database and
Storage state exactly as it was found.
"""

import io
import os
import sys

from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv("backend/.env")

from backend.services.assessment_repository import (
    get_assessment_by_id,
    set_report_storage_path
)
from backend.services.report_service import (
    generate_assessment_report,
    REPORT_TITLE,
    DAMAGE_CLASS_NAMES
)
from backend.services.storage_service import (
    upload_assessment_report,
    get_assessment_report_path,
    REPORT_FILENAME,
    MAX_REPORT_SIZE
)
from backend.services.supabase_service import get_supabase


STORAGE_BUCKET = "assessments"

VERIFICATION_ASSESSMENT_ID = (
    "39efcfcb-a93d-496b-9e09-95b759a1c3e3"
)

PDF_MAGIC = b"%PDF-"


# ============================================================
# HELPERS
# ============================================================

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


def report_objects() -> list:
    """
    Return every report.pdf currently in the bucket.
    """

    return [
        name for name in all_bucket_objects()
        if name.endswith(REPORT_FILENAME)
    ]


def snapshot_state() -> dict:
    """
    Capture database and Storage state before the test.
    """

    row = (
        get_supabase()
        .table("assessments")
        .select("id, report_storage_path")
        .eq("id", VERIFICATION_ASSESSMENT_ID)
        .execute()
    )

    original_path = (
        row.data[0].get("report_storage_path")
        if row.data
        else None
    )

    return {
        "original_path": original_path,
        "objects": all_bucket_objects(),
        "reports": report_objects(),
    }


def remove_object(full_path: str) -> None:
    """
    Delete one object by its full bucket path.
    """

    get_supabase().storage.from_(
        STORAGE_BUCKET
    ).remove([full_path])


def restore_state(snapshot: dict) -> None:
    """
    Remove test objects and restore the original
    report_storage_path value.
    """

    current = set(all_bucket_objects())
    original = set(snapshot["objects"])

    for name in current - original:
        remove_object(name)

    if (
        snapshot["original_path"]
        is not None
    ):
        set_report_storage_path(
            VERIFICATION_ASSESSMENT_ID,
            snapshot["original_path"]
        )
    else:
        get_supabase().table("assessments").update(
            {"report_storage_path": None}
        ).eq("id", VERIFICATION_ASSESSMENT_ID).execute()


def check(label, condition, detail, failures):
    if condition:
        print(f"  [PASS] {label}")
        print(f"         {detail}")
        return

    print(f"  [FAIL] {label}")
    print(f"         {detail}")
    failures.append(label)


def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


def embedded(value) -> dict:
    """
    Normalize an embedded PostgREST resource.

    Depending on the relationship the Supabase client
    returns either a single dictionary or a list with
    one dictionary, so accept both shapes.
    """

    if isinstance(value, dict):
        return value

    if isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], dict):
            return value[0]

    return {}


# ============================================================
# TEST
# ============================================================

def main():
    failures = []

    print("=" * 66)
    print("STEP 7.12 INTEGRATION TEST")
    print("=" * 66)

    print("\n[0] Capture state before the test")
    snapshot = snapshot_state()

    print(f"     original report_storage_path : "
          f"{snapshot['original_path']}")
    print(f"     storage objects              : "
          f"{len(snapshot['objects'])}")

    uploaded_path = None

    try:
        print("\n[1] Load the existing verified assessment")
        assessment = get_assessment_by_id(
            VERIFICATION_ASSESSMENT_ID
        )

        loaded = assessment is not None
        images = assessment.get("assessment_images") or []
        stats = assessment.get("assessment_class_statistics") or []
        predictions = embedded(
            assessment.get("assessment_predictions")
        )

        detail = (
            f"id {assessment['id']}, "
            f"{len(images)} images, "
            f"{1 if predictions else 0} prediction(s), "
            f"{len(stats)} class rows"
            if loaded
            else "get_assessment_by_id returned None"
        )

        check(
            "assessment loaded from database",
            loaded,
            detail,
            failures
        )

        if not loaded:
            return 1

        check(
            "assessment is the verification row",
            assessment["id"] == VERIFICATION_ASSESSMENT_ID,
            assessment["id"],
            failures
        )

        print("\n[2] Generate the PDF report")
        pdf_bytes = generate_assessment_report(assessment)

        check(
            "PDF generated",
            isinstance(pdf_bytes, bytes),
            f"{len(pdf_bytes)} bytes = "
            f"{len(pdf_bytes) / 1024:.1f} KB",
            failures
        )

        check(
            "PDF header is valid",
            pdf_bytes.startswith(PDF_MAGIC),
            repr(pdf_bytes[:8]),
            failures
        )

        check(
            "PDF is below the 2 MB limit",
            len(pdf_bytes) < MAX_REPORT_SIZE,
            f"{len(pdf_bytes)} bytes < "
            f"{MAX_REPORT_SIZE} bytes",
            failures
        )

        print("\n[3] Upload to the private assessments bucket")
        expected_path = get_assessment_report_path(
            VERIFICATION_ASSESSMENT_ID
        )

        check(
            "path is bucket-qualified as specified",
            expected_path
            == f"{STORAGE_BUCKET}/"
            f"{VERIFICATION_ASSESSMENT_ID}/"
            f"{REPORT_FILENAME}",
            expected_path,
            failures
        )

        uploaded_path = upload_assessment_report(
            VERIFICATION_ASSESSMENT_ID,
            pdf_bytes
        )

        check(
            "upload returned the expected path",
            uploaded_path == expected_path,
            uploaded_path,
            failures
        )

        print("\n[4] Verify the object exists")
        bucket = get_supabase().storage.from_(
            STORAGE_BUCKET
        )
        relative_path = f"{VERIFICATION_ASSESSMENT_ID}/{REPORT_FILENAME}"

        check(
            "object exists in the bucket",
            bucket.exists(relative_path),
            relative_path,
            failures
        )

        stored = report_objects()

        check(
            "exactly one report object",
            len(stored) == 1,
            f"{stored}",
            failures
        )

        print("\n[5] Verify the stored object is below 2 MB")
        info = bucket.info(relative_path)

        check(
            "stored size below 2 MB",
            int(info["size"]) < MAX_REPORT_SIZE,
            f"{info['size']} bytes = "
            f"{int(info['size']) / 1024:.1f} KB",
            failures
        )

        check(
            "content type is application/pdf",
            info.get("content_type") == "application/pdf",
            str(info.get("content_type")),
            failures
        )

        check(
            "stored bytes match generated bytes",
            int(info["size"]) == len(pdf_bytes),
            f"{info['size']} == {len(pdf_bytes)}",
            failures
        )

        print("\n[6] Download and validate the PDF")
        downloaded = bucket.download(relative_path)

        check(
            "download returned bytes",
            isinstance(downloaded, bytes),
            f"type = {type(downloaded).__name__}",
            failures
        )

        check(
            "downloaded data is a PDF",
            downloaded.startswith(PDF_MAGIC),
            repr(downloaded[:8]),
            failures
        )

        check(
            "downloaded size is below 2 MB",
            len(downloaded) < MAX_REPORT_SIZE,
            f"{len(downloaded)} bytes = "
            f"{len(downloaded) / 1024:.1f} KB",
            failures
        )

        try:
            text = extract_text(downloaded)
            pages = len(PdfReader(io.BytesIO(downloaded)).pages)
            valid = pages >= 1
            detail = f"{pages} page(s), {len(text)} chars"
        except Exception as error:
            valid = False
            text = ""
            detail = f"raised {type(error).__name__}: {error}"

        check(
            "downloaded PDF opens in pypdf",
            valid,
            detail,
            failures
        )

        print("\n[7] Verify the report content")
        check(
            "report contains the title",
            REPORT_TITLE in text,
            REPORT_TITLE,
            failures
        )

        check(
            "report contains the assessment ID",
            VERIFICATION_ASSESSMENT_ID in text,
            VERIFICATION_ASSESSMENT_ID,
            failures
        )

        model = embedded(
            assessment.get("assessment_models")
        )

        check(
            "report contains the model name",
            str(model.get("name", "")) in text,
            str(model.get("name", "")),
            failures
        )

        for class_name in DAMAGE_CLASS_NAMES.values():
            check(
                f"report contains class '{class_name}'",
                class_name in text,
                class_name,
                failures
            )

        print("\n[8] Persist the report storage path")
        stored_row = set_report_storage_path(
            VERIFICATION_ASSESSMENT_ID,
            uploaded_path
        )

        check(
            "path written to the database",
            stored_row.get("report_storage_path")
            == uploaded_path,
            str(stored_row.get("report_storage_path")),
            failures
        )

        reread = (
            get_supabase()
            .table("assessments")
            .select("report_storage_path")
            .eq("id", VERIFICATION_ASSESSMENT_ID)
            .execute()
        )

        check(
            "path readable back from the database",
            reread.data
            and reread.data[0]["report_storage_path"]
            == uploaded_path,
            str(
                reread.data[0]["report_storage_path"]
                if reread.data
                else None
            ),
            failures
        )

        print("\n[9] Storage budget")
        total = sum(
            int(obj.metadata.get("size", 0))
            for obj in get_supabase()
            .storage.from_(STORAGE_BUCKET)
            .list_v2(options={"prefix": "", "limit": 1000})
            .objects
        )

        print(f"     total bucket size : {total} bytes")

        check(
            "bucket stays far below 50 MB",
            total < 50 * 1024 * 1024,
            f"{total / 1024:.1f} KB of 50 MB budget",
            failures
        )

    finally:
        print("\n[10] Cleanup")
        restore_state(snapshot)

        final = all_bucket_objects()

        print(f"     storage objects : {final}")

        check(
            "storage state restored",
            final == snapshot["objects"],
            f"before={snapshot['objects']} after={final}",
            failures
        )

        final_row = (
            get_supabase()
            .table("assessments")
            .select("report_storage_path")
            .eq("id", VERIFICATION_ASSESSMENT_ID)
            .execute()
        )

        final_path = (
            final_row.data[0]["report_storage_path"]
            if final_row.data
            else None
        )

        check(
            "database state restored",
            final_path == snapshot["original_path"],
            f"before={snapshot['original_path']} "
            f"after={final_path}",
            failures
        )

        check(
            "no leftover report object",
            report_objects() == snapshot["reports"],
            f"{report_objects()}",
            failures
        )

    print("\n" + "=" * 66)

    if failures:
        print(
            f"INTEGRATION TEST FAILED: "
            f"{len(failures)} failure(s)"
        )
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("INTEGRATION TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
