"""
Focused test for wiring PDF report generation into the live
assessment workflow.

Two layers are covered without touching Supabase or Storage:

  _store_assessment_report()
    1. reads the persisted record, then generates the PDF,
       then uploads it, then persists the path - never the
       reverse order
    2. a missing record raises and stores nothing
    3. an upload failure stores no path and leaves no object
    4. a path-persist failure removes the just-uploaded
       object and re-raises

  run_assessment() report step
    5. a report failure is non-fatal: the assessment still
       completes, and neither an upload nor a path write
       happens for the report
    6. a successful report step records the stored path

The heavyweight pipeline (preprocessing, U-Net, Supabase) is
replaced with deterministic fakes so the failure semantics can
be observed directly and quickly. Nothing in this file contacts
the network or the database.
"""

import sys
import uuid
from types import SimpleNamespace

from dotenv import load_dotenv


load_dotenv("backend/.env")

import numpy as np

import backend.services.assessment_service as service


ASSESSMENT_ID = str(uuid.uuid4())

REPORT_PATH = (
    f"assessments/{ASSESSMENT_ID}/report.pdf"
)

SAMPLE_PDF = b"%PDF-1.4\n% fake report for workflow tests\n"


class Patch:
    """
    Replace module attributes and restore them afterwards so a
    partially failed test cannot leak a fake into the module.
    """

    def __init__(self):
        self._saved = []
        self.calls = {}

    def set(self, owner, name, value):
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def reset_calls(self):
        self.calls = {}

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


def fake_calculations():
    total = 65536
    share = total // len(service.CLASS_NAMES)

    return {
        "damage_level": "minor",
        "damage_percentage": 8.0,
        "total_pixels": total,
        "damage_pixels": share,
        "classes": {
            name: {"pixels": share, "percentage": 20.0}
            for name in service.CLASS_NAMES.values()
        },
    }


def image_info(filename):
    return {
        "filename": filename,
        "content_type": "image/png",
        "format": "PNG",
        "width": 256,
        "height": 256,
        "size_bytes": 1024,
        "bytes": b"\x89PNG\r\n\x1a\n",
    }


# ============================================================
# PIPELINE FAKES
# ============================================================

def patch_pipeline(patch):
    """
    Replace every heavyweight pipeline dependency of
    run_assessment() with a deterministic fake.
    """

    mask = np.zeros((256, 256), dtype=np.int32)
    mask[0, 0] = 1
    mask[1, 1] = 2
    mask[2, 2] = 3
    mask[3, 3] = 4

    stats = fake_calculations()

    patch.set(
        service,
        "prepare_model_input",
        lambda before, after: np.zeros((9, 256, 256), dtype="float32"),
    )
    patch.set(
        service,
        "damage_model",
        SimpleNamespace(
            predict=lambda tensor: mask,
            calculate_statistics=lambda value: stats,
        ),
    )
    patch.set(
        service,
        "get_model_by_checkpoint",
        lambda checkpoint: {"id": "model-id"},
    )
    patch.set(
        service,
        "ensure_user_profile",
        lambda user_id, email=None: None,
    )
    patch.set(
        service,
        "create_assessment",
        lambda **kwargs: {"id": ASSESSMENT_ID},
    )
    patch.set(
        service,
        "upload_assessment_image",
        lambda **kwargs: (
            f"assessments/{kwargs['assessment_id']}/x.png"
        ),
    )
    patch.set(
        service,
        "prediction_mask_to_png",
        lambda value: b"\x89PNG\r\n\x1a\n",
    )
    patch.set(
        service,
        "upload_prediction_mask",
        lambda **kwargs: (
            f"assessments/{kwargs['assessment_id']}/"
            "prediction-mask.png"
        ),
    )
    patch.set(
        service,
        "add_assessment_image",
        lambda **kwargs: {},
    )
    patch.set(
        service,
        "add_prediction",
        lambda **kwargs: {},
    )
    patch.set(
        service,
        "add_class_statistics",
        lambda **kwargs: [],
    )


# ============================================================
# _store_assessment_report
# ============================================================

def test_store_success(failures, state):
    print("\n[1] _store_assessment_report success order")
    state["patch"].reset_calls()
    state["patch"].set(
        service,
        "get_assessment_by_id",
        lambda assessment_id: {"id": assessment_id},
    )
    state["patch"].set(
        service,
        "generate_assessment_report",
        lambda assessment: SAMPLE_PDF,
    )
    state["patch"].set(
        service,
        "upload_assessment_report",
        lambda assessment_id, pdf_bytes: REPORT_PATH,
    )
    state["patch"].set(
        service,
        "set_report_storage_path",
        lambda assessment_id, report_storage_path: state[
            "patch"
        ].calls.setdefault(
            "persist",
            report_storage_path,
        ),
    )

    returned = service._store_assessment_report(ASSESSMENT_ID)

    check(
        "returns the stored path",
        returned == REPORT_PATH,
        returned,
        failures,
    )

    persisted = state["patch"].calls.get("persist")

    check(
        "path written after upload",
        persisted == REPORT_PATH,
        str(persisted),
        failures,
    )


def test_store_missing_record(failures, state):
    print("\n[2] _store_assessment_report with a missing record")
    state["patch"].reset_calls()
    state["patch"].set(
        service,
        "get_assessment_by_id",
        lambda assessment_id: None,
    )
    state["patch"].set(
        service,
        "upload_assessment_report",
        lambda assessment_id, pdf_bytes: state["patch"].calls.update(
            {"upload": "called"}
        ) or REPORT_PATH,
    )
    state["patch"].set(
        service,
        "set_report_storage_path",
        lambda assessment_id, path: state["patch"].calls.update(
            {"persist": "called"}
        ),
    )

    try:
        service._store_assessment_report(ASSESSMENT_ID)
        raised = None
    except RuntimeError as error:
        raised = type(error).__name__

    check(
        "raises RuntimeError",
        raised == "RuntimeError",
        str(raised),
        failures,
    )

    check(
        "nothing uploaded or persisted",
        "upload" not in state["patch"].calls
        and "persist" not in state["patch"].calls,
        str(state["patch"].calls),
        failures,
    )


def test_store_upload_failure(failures, state):
    print("\n[3] _store_assessment_report upload failure")
    state["patch"].reset_calls()
    state["patch"].set(
        service,
        "get_assessment_by_id",
        lambda assessment_id: {"id": assessment_id},
    )
    state["patch"].set(
        service,
        "generate_assessment_report",
        lambda assessment: SAMPLE_PDF,
    )
    state["patch"].set(
        service,
        "upload_assessment_report",
        lambda assessment_id, pdf_bytes: _raise(
            "upload failed"
        ),
    )
    state["patch"].set(
        service,
        "set_report_storage_path",
        lambda assessment_id, path: state["patch"].calls.update(
            {"persist": "called"}
        ),
    )
    state["patch"].set(
        service,
        "_discard_uploaded_objects",
        lambda paths: state["patch"].calls.update(
            {"discard": list(paths)}
        ),
    )

    try:
        service._store_assessment_report(ASSESSMENT_ID)
        raised = None
    except RuntimeError as error:
        raised = type(error).__name__

    check(
        "raises RuntimeError",
        raised == "RuntimeError",
        str(raised),
        failures,
    )

    check(
        "no path persisted",
        "persist" not in state["patch"].calls,
        str(state["patch"].calls),
        failures,
    )

    check(
        "no orphan object removed",
        "discard" not in state["patch"].calls,
        str(state["patch"].calls),
        failures,
    )


def test_store_persist_failure(failures, state):
    print("\n[4] _store_assessment_report path-persist failure")
    state["patch"].reset_calls()
    state["patch"].set(
        service,
        "get_assessment_by_id",
        lambda assessment_id: {"id": assessment_id},
    )
    state["patch"].set(
        service,
        "generate_assessment_report",
        lambda assessment: SAMPLE_PDF,
    )
    state["patch"].set(
        service,
        "upload_assessment_report",
        lambda assessment_id, pdf_bytes: REPORT_PATH,
    )
    state["patch"].set(
        service,
        "set_report_storage_path",
        lambda assessment_id, path: _raise(
            "database write failed"
        ),
    )
    state["patch"].set(
        service,
        "_discard_uploaded_objects",
        lambda paths: state["patch"].calls.update(
            {"discard": list(paths)}
        ),
    )

    try:
        service._store_assessment_report(ASSESSMENT_ID)
        raised = None
    except RuntimeError as error:
        raised = type(error).__name__

    check(
        "raises RuntimeError",
        raised == "RuntimeError",
        str(raised),
        failures,
    )

    check(
        "uploaded object removed again",
        state["patch"].calls.get("discard") == [REPORT_PATH],
        str(state["patch"].calls),
        failures,
    )


def _raise(message):
    raise RuntimeError(message)


# ============================================================
# run_assessment report step
# ============================================================

def test_run_report_failure_is_non_fatal(failures, state):
    print("\n[5] run_assessment survives a report failure")
    patch_pipeline(state["patch"])
    state["patch"].set(
        service,
        "_store_assessment_report",
        lambda assessment_id: _raise("report boom"),
    )
    state["patch"].set(
        service,
        "upload_assessment_report",
        lambda assessment_id, pdf_bytes: state["patch"].calls.update(
            {"upload": "called"}
        ) or REPORT_PATH,
    )
    state["patch"].set(
        service,
        "set_report_storage_path",
        lambda assessment_id, path: state["patch"].calls.update(
            {"persist": "called"}
        ),
    )

    result = service.run_assessment(
        image_info("before.png"),
        image_info("after.png"),
        user_id="user-1",
        user_email="user-1@example.invalid",
    )

    check(
        "assessment still succeeds",
        result["status"] == "success"
        and result["id"] == ASSESSMENT_ID,
        f"status={result['status']} id={result['id']}",
        failures,
    )

    check(
        "no report path written",
        "persist" not in state["patch"].calls,
        str(state["patch"].calls),
        failures,
    )

    check(
        "no report uploaded",
        "upload" not in state["patch"].calls,
        str(state["patch"].calls),
        failures,
    )


def test_run_report_success(failures, state):
    print("\n[6] run_assessment records a successful report")
    patch_pipeline(state["patch"])
    state["patch"].set(
        service,
        "get_assessment_by_id",
        lambda assessment_id: {"id": assessment_id},
    )
    state["patch"].set(
        service,
        "generate_assessment_report",
        lambda assessment: SAMPLE_PDF,
    )
    state["patch"].set(
        service,
        "upload_assessment_report",
        lambda assessment_id, pdf_bytes: REPORT_PATH,
    )
    state["patch"].set(
        service,
        "set_report_storage_path",
        lambda assessment_id, path: state["patch"].calls.update(
            {"persist": path}
        ),
    )

    result = service.run_assessment(
        image_info("before.png"),
        image_info("after.png"),
        user_id="user-1",
        user_email="user-1@example.invalid",
    )

    check(
        "assessment succeeds",
        result["status"] == "success"
        and result["id"] == ASSESSMENT_ID,
        f"status={result['status']} id={result['id']}",
        failures,
    )

    check(
        "report path persisted for the assessment",
        state["patch"].calls.get("persist") == REPORT_PATH,
        str(state["patch"].calls.get("persist")),
        failures,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    failures = []
    state = {"patch": Patch()}

    print("=" * 66)
    print("REPORT LIVE WORKFLOW TEST")
    print("=" * 66)

    try:
        test_store_success(failures, state)
        test_store_missing_record(failures, state)
        test_store_upload_failure(failures, state)
        test_store_persist_failure(failures, state)
        test_run_report_failure_is_non_fatal(failures, state)
        test_run_report_success(failures, state)
    finally:
        state["patch"].restore()

    print("\n" + "=" * 66)

    if failures:
        print(
            f"REPORT LIVE WORKFLOW TEST FAILED: "
            f"{len(failures)} failure(s)"
        )
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("REPORT LIVE WORKFLOW TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())