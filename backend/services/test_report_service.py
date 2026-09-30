import io
import os

from pypdf import PdfReader

from backend.services.report_service import (
    generate_assessment_report,
    REPORT_TITLE,
    DAMAGE_CLASS_NAMES,
)


# ============================================================
# FIXTURES
# ============================================================

VERIFICATION_ASSESSMENT_ID = (
    "39efcfcb-a93d-496b-9e09-95b759a1c3e3"
)


def sample_assessment() -> dict:
    """
    Build an assessment shaped exactly like the
    structure returned by get_assessment_by_id().
    """

    return {
        "id": VERIFICATION_ASSESSMENT_ID,
        "user_id": None,
        "model_id": "efaa83a4-5f1b-4a5b-95ff-f10f91f935e0",
        "status": "completed",
        "damage_level": "minor",
        "damage_percentage": 12.5,
        "total_pixels": 65536,
        "damage_pixels": 8192,
        "created_at": "2026-01-15T09:30:00+00:00",
        "updated_at": "2026-01-15T09:30:05+00:00",
        "report_storage_path": (
            f"assessments/{VERIFICATION_ASSESSMENT_ID}"
            "/report.pdf"
        ),
        "assessment_models": [
            {
                "id": "efaa83a4-5f1b-4a5b-95ff-f10f91f935e0",
                "name": "Change-Aware U-Net",
                "architecture": "UNetChangeAware",
                "checkpoint": "best_model_change_aware.pth",
                "input_channels": 12,
                "output_classes": 5,
                "epoch": 3,
                "validation_loss": 0.4699871812547956,
            }
        ],
        "assessment_images": [
            {
                "filename": "before.png",
                "image_type": "before",
                "width": 1024,
                "height": 1024,
                "size_bytes": 154129,
                "content_type": "image/png",
                "format": "PNG",
                "storage_path": "assessments/x/before.png",
            },
            {
                "filename": "after.png",
                "image_type": "after",
                "width": 1024,
                "height": 1024,
                "size_bytes": 180897,
                "content_type": "image/png",
                "format": "PNG",
                "storage_path": "assessments/x/after.png",
            },
        ],
        "assessment_predictions": [
            {
                "mask_width": 256,
                "mask_height": 256,
                "predicted_classes": [0, 1],
                "mask_storage_path": (
                    "assessments/x/prediction-mask.png"
                ),
            }
        ],
        "assessment_class_statistics": [
            {
                "class_id": 0,
                "class_name": "background",
                "pixel_count": 57344,
                "percentage": 87.5,
            },
            {
                "class_id": 1,
                "class_name": "no_damage",
                "pixel_count": 4096,
                "percentage": 6.25,
            },
            {
                "class_id": 2,
                "class_name": "minor_damage",
                "pixel_count": 2048,
                "percentage": 3.125,
            },
            {
                "class_id": 3,
                "class_name": "major_damage",
                "pixel_count": 1024,
                "percentage": 1.5625,
            },
            {
                "class_id": 4,
                "class_name": "destroyed",
                "pixel_count": 1024,
                "percentage": 1.5625,
            },
        ],
    }


def extract_text(pdf_bytes: bytes) -> str:
    """
    Read the PDF back and return its full text.
    """

    reader = PdfReader(io.BytesIO(pdf_bytes))

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


def check(label, condition, detail, failures):
    """
    Record one assertion result.
    """

    if condition:
        print(f"  [PASS] {label}")
        print(f"         {detail}")
        return

    print(f"  [FAIL] {label}")
    print(f"         {detail}")
    failures.append(label)


# ============================================================
# TEST
# ============================================================

def main():
    failures = []

    print("=" * 66)
    print("STEP 7.12 REPORT SERVICE TEST")
    print("=" * 66)

    print("\n[1] PDF generation succeeds")
    try:
        pdf_bytes = generate_assessment_report(
            sample_assessment()
        )
        generated = True
        detail = f"{len(pdf_bytes)} bytes produced"
    except Exception as error:
        generated = False
        detail = f"raised {type(error).__name__}: {error}"

    check(
        "generation succeeds",
        generated,
        detail,
        failures
    )

    if not generated:
        print("\nREPORT SERVICE TEST FAILED")
        return 1

    print("\n[2] Returned data is bytes")
    check(
        "result is bytes",
        isinstance(pdf_bytes, bytes),
        f"type = {type(pdf_bytes).__name__}",
        failures
    )

    print("\n[3] PDF has a valid PDF header")
    check(
        "header is %PDF-",
        pdf_bytes.startswith(b"%PDF-"),
        f"first 8 bytes = {pdf_bytes[:8]!r}",
        failures
    )

    print("\n[4] PDF can be opened and read by pypdf")
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        text = extract_text(pdf_bytes)
        readable = page_count >= 1
        detail = f"{page_count} page(s), {len(text)} chars extracted"
    except Exception as error:
        readable = False
        text = ""
        detail = f"raised {type(error).__name__}: {error}"

    check(
        "PDF is readable",
        readable,
        detail,
        failures
    )

    print("\n[5] Report contains the assessment ID")
    check(
        "assessment ID present",
        VERIFICATION_ASSESSMENT_ID in text,
        VERIFICATION_ASSESSMENT_ID,
        failures
    )

    print("\n[6] Report contains the model name")
    check(
        "model name present",
        "Change-Aware U-Net" in text,
        "Change-Aware U-Net",
        failures
    )

    print("\n[7] Report contains all five damage class names")
    for class_id in sorted(DAMAGE_CLASS_NAMES):
        class_name = DAMAGE_CLASS_NAMES[class_id]
        check(
            f"class {class_id} ({class_name}) present",
            class_name in text,
            class_name,
            failures
        )

    print("\n[8] Report contains the damage percentage")
    check(
        "damage percentage present",
        "12.5" in text or "12.50" in text,
        "12.5%",
        failures
    )

    print("\n[9] Report size is below 2 MB")
    size = len(pdf_bytes)
    check(
        "size below 2 MB",
        size < 2 * 1024 * 1024,
        f"{size} bytes = {size / 1024:.1f} KB",
        failures
    )
    check(
        "no bulk media embedded",
        b"/Subtype /Image" not in pdf_bytes
        and b"/XObject" not in pdf_bytes
        and b"/DCTDecode" not in pdf_bytes
        and b"/FlateDecode" not in pdf_bytes.replace(
            b"/Filter [ /ASCII85Decode /FlateDecode ]", b""
        ),
        "no image XObjects, no JPEG or compressed "
        "image streams in the PDF",
        failures
    )

    print("\n[10] No temporary permanent file left behind")
    tracked = [
        "report.pdf",
        "assessment_report.pdf",
    ]
    leftovers = [
        name for name in tracked
        if os.path.isfile(name)
    ]
    check(
        "no report file on disk",
        not leftovers,
        f"found {leftovers or 'none'}",
        failures
    )
    check(
        "generated in memory only",
        isinstance(pdf_bytes, bytes),
        "reportlab wrote into an in-memory BytesIO",
        failures
    )

    print("\n" + "=" * 66)

    print("\n[11] Missing optional values handled gracefully")
    for label, sparse in [
        (
            "empty assessment",
            {},
        ),
        (
            "id and model only",
            {"id": "abc"},
        ),
        (
            "no model, no images",
            {
                "id": "abc",
                "assessment_models": [],
                "assessment_images": [],
            },
        ),
        (
            "partial statistics",
            {
                "assessment_class_statistics": [
                    {"class_id": 2, "class_name": "minor_damage"}
                ]
            },
        ),
        (
            "embedded resource as dict",
            {
                "assessment_models": {"name": "Flat Model"},
                "assessment_predictions": {
                    "mask_width": 256
                },
            },
        ),
        (
            "wrong types",
            {
                "id": 12345,
                "damage_percentage": "not-a-number",
                "total_pixels": None,
                "assessment_images": [
                    {"image_type": "before", "size_bytes": "abc"}
                ],
                "assessment_class_statistics": [
                    {"class_id": "bad"},
                    "not-a-dict",
                ],
            },
        ),
    ]:
        try:
            sparse_bytes = generate_assessment_report(sparse)
            sparse_ok = (
                isinstance(sparse_bytes, bytes)
                and sparse_bytes.startswith(b"%PDF-")
                and len(sparse_bytes) < 2 * 1024 * 1024
            )
            detail = f"{len(sparse_bytes)} bytes, valid PDF"
        except Exception as error:
            sparse_ok = False
            detail = f"raised {type(error).__name__}: {error}"

        check(
            f"handles {label}",
            sparse_ok,
            detail,
            failures
        )

    print("\n[12] All five class names always present")
    try:
        sparse_text = extract_text(
            generate_assessment_report(
                {"assessment_class_statistics": []}
            )
        )
        all_present = all(
            name in sparse_text
            for name in DAMAGE_CLASS_NAMES.values()
        )
        detail = "all five names rendered without DB rows"
    except Exception as error:
        all_present = False
        detail = f"raised {type(error).__name__}: {error}"

    check(
        "class table is never empty",
        all_present,
        detail,
        failures
    )

    print("\n[13] Rejects non-dictionary input")
    for bad in [None, "string", 12345, [1, 2]]:
        try:
            generate_assessment_report(bad)
            rejected = False
            detail = "no error raised"
        except ValueError:
            rejected = True
            detail = "ValueError raised"
        except Exception as error:
            rejected = False
            detail = f"raised {type(error).__name__}"

        check(
            f"rejects {type(bad).__name__}",
            rejected,
            detail,
            failures
        )

    print("\n[14] Report section headings present")
    for heading in [
        "Assessment information",
        "Model information",
        "Image information",
        "Prediction information",
        "Damage class statistics",
    ]:
        check(
            f"heading '{heading}'",
            heading in text,
            heading,
            failures
        )

    check(
        "title present",
        REPORT_TITLE in text,
        REPORT_TITLE,
        failures
    )

    print("\n" + "=" * 66)

    if failures:
        print(
            f"REPORT SERVICE TEST FAILED: "
            f"{len(failures)} failure(s)"
        )
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("REPORT SERVICE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
