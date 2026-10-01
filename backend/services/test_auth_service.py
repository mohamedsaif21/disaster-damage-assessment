"""
Step 8.4 - authentication and ownership verification.

Deterministic checks that need no live Supabase auth call, no
database write and no model inference:

    token verification -> dependency rejection ->
    route protection -> derived ownership -> no token leak ->
    frontend holds no service-role key

Live database and signed-URL behaviour stays in the integration
tests; this file only proves the authentication contract.
"""

import io
import logging
from types import SimpleNamespace

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from PIL import Image


load_dotenv("backend/.env")

from backend.main import app
from backend.api import assessment as assessment_api
from backend.services import auth_service
from backend.services.auth_service import (
    AuthenticatedUser,
    InvalidAccessTokenError,
    UserProfileError,
    ensure_user_profile,
    get_current_user,
    verify_access_token,
)
from backend.services.asset_service import AssetAccessDenied
from supabase_auth.errors import AuthError


TEST_USER_ID = "00000000-0000-4000-8000-0000000000a4"
TEST_EMAIL = "auth-test@example.com"

FRONTEND_DIRS = [
    r"frontend\src",
]


# ============================================================
# HELPERS
# ============================================================

def check(label, condition, detail, failures):
    if condition:
        print(f"  [PASS] {label}")
        return True

    print(f"  [FAIL] {label}")
    print(f"         {detail}")
    failures.append(label)
    return False


def png_pair(side=256):
    """
    Return two valid, identically sized PNG upload tuples.
    """

    def one(name):
        buffer = io.BytesIO()
        Image.new("RGB", (side, side)).save(
            buffer, format="PNG"
        )
        return (name, buffer.getvalue(), "image/png")

    return one("before.png"), one("after.png")


class FakeAuth:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.received = None

    def get_claims(self, token):
        self.received = token

        if self.error is not None:
            raise self.error

        return self.result


class FakeSupabase:
    def __init__(self, auth):
        self.auth = auth


def install_fake_auth(result=None, error=None):
    fake = FakeSupabase(FakeAuth(result=result, error=error))
    original = auth_service.get_supabase
    auth_service.get_supabase = lambda: fake
    return fake, original


def restore(original):
    auth_service.get_supabase = original


def claims_response(**claims):
    return SimpleNamespace(claims=claims)


# ============================================================
# PROFILE ROW FAKES
# ============================================================

class FakeQuery:
    def __init__(self, data):
        self._data = data

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=self._data)


class FakeUsersTable:
    def __init__(self, existing, calls):
        self._existing = existing
        self._calls = calls

    def select(self, *args, **kwargs):
        return FakeQuery(self._existing)

    def upsert(self, payload, **kwargs):
        self._calls.append((payload, kwargs))
        return FakeQuery(None)


class FakeProfileSupabase:
    def __init__(self, existing):
        self._existing = existing
        self.calls = []

    def table(self, name):
        assert name == "users", name
        return FakeUsersTable(self._existing, self.calls)


# ============================================================
# TESTS
# ============================================================

def test_verify_access_token(failures):
    print("\n" + "-" * 66)
    print("A - TOKEN VERIFICATION")
    print("-" * 66)

    fake, original = install_fake_auth(
        result=claims_response(
            sub=TEST_USER_ID,
            role="authenticated",
            email=TEST_EMAIL,
        )
    )

    try:
        user = verify_access_token("  good.token  ")

        check(
            "valid token derives the user id",
            user == AuthenticatedUser(
                user_id=TEST_USER_ID,
                email=TEST_EMAIL,
            ),
            str(user),
            failures,
        )

        check(
            "token is trimmed before verification",
            fake.auth.received == "good.token",
            repr(fake.auth.received),
            failures,
        )
    finally:
        restore(original)

    cases = [
        (
            "empty token",
            claims_response(
                sub=TEST_USER_ID, role="authenticated"
            ),
            "",
        ),
        (
            "missing subject",
            claims_response(role="authenticated"),
            "x.token",
        ),
        (
            "non authenticated role",
            claims_response(
                sub=TEST_USER_ID, role="anon"
            ),
            "x.token",
        ),
        (
            "no claims response",
            None,
            "x.token",
        ),
    ]

    for label, result, token in cases:
        _, original = install_fake_auth(result=result)

        try:
            raised = False
            try:
                verify_access_token(token)
            except InvalidAccessTokenError:
                raised = True

            check(
                f"{label} is rejected",
                raised,
                "InvalidAccessTokenError expected",
                failures,
            )
        finally:
            restore(original)

    fake, original = install_fake_auth(
        error=AuthError("bad jwt", None)
    )

    try:
        raised = False
        try:
            verify_access_token("expired.token")
        except InvalidAccessTokenError:
            raised = True

        check(
            "auth failure becomes InvalidAccessTokenError",
            raised,
            "InvalidAccessTokenError expected",
            failures,
        )

        # The token must never leak through the error text.
        try:
            verify_access_token("secret-token-value")
        except InvalidAccessTokenError as error:
            check(
                "verification error never echoes the token",
                "secret-token-value"
                not in str(error)
                and "secret-token-value"
                not in repr(error),
                str(error),
                failures,
            )
    finally:
        restore(original)

    # A non-auth outage must not be disguised as a 401 source.
    def boom(token):
        raise RuntimeError("supabase is down")

    fake = FakeSupabase(FakeAuth())
    fake.auth.get_claims = boom
    original = auth_service.get_supabase
    auth_service.get_supabase = lambda: fake

    try:
        propagated = False
        try:
            verify_access_token("x.token")
        except RuntimeError:
            propagated = True
        except InvalidAccessTokenError:
            propagated = False

        check(
            "unexpected outages propagate untouched",
            propagated,
            "RuntimeError should not become InvalidAccessTokenError",
            failures,
        )
    finally:
        restore(original)


def test_get_current_user(failures):
    print("\n" + "-" * 66)
    print("B - DEPENDENCY REJECTION")
    print("-" * 66)

    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials

    def status_of(credentials):
        try:
            get_current_user(credentials)
        except HTTPException as error:
            return error.status_code, error.headers
        return None, None

    code, headers = status_of(None)

    check(
        "missing credentials return 401 with Bearer challenge",
        code == 401
        and headers == {"WWW-Authenticate": "Bearer"},
        f"{code} {headers}",
        failures,
    )

    code, _ = status_of(
        HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=""
        )
    )

    check(
        "empty credentials return 401",
        code == 401,
        str(code),
        failures,
    )

    code, _ = status_of(
        HTTPAuthorizationCredentials(
            scheme="Basic", credentials="abc"
        )
    )

    check(
        "non-Bearer scheme returns 401",
        code == 401,
        str(code),
        failures,
    )

    original = auth_service.verify_access_token
    auth_service.verify_access_token = lambda token: AuthenticatedUser(
        user_id=TEST_USER_ID, email=TEST_EMAIL
    )

    try:
        user = get_current_user(
            HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="valid.token"
            )
        )

        check(
            "valid credentials return the verified user",
            user.user_id == TEST_USER_ID,
            str(user),
            failures,
        )
    finally:
        auth_service.verify_access_token = original

    auth_service.verify_access_token = lambda token: (
        _raise(InvalidAccessTokenError("bad"))
    )

    try:
        code, _ = status_of(
            HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="bad.token"
            )
        )

        check(
            "invalid token returns 401",
            code == 401,
            str(code),
            failures,
        )
    finally:
        auth_service.verify_access_token = original


def _raise(error):
    raise error


def test_ensure_user_profile(failures):
    print("\n" + "-" * 66)
    print("C - PROFILE ROW")
    print("-" * 66)

    existing = FakeProfileSupabase([{"id": TEST_USER_ID}])
    original = auth_service.get_supabase
    auth_service.get_supabase = lambda: existing

    try:
        ensure_user_profile(TEST_USER_ID, TEST_EMAIL)

        check(
            "existing profile is left untouched",
            existing.calls == [],
            str(existing.calls),
            failures,
        )
    finally:
        restore(original)

    missing = FakeProfileSupabase([])
    auth_service.get_supabase = lambda: missing

    try:
        ensure_user_profile(TEST_USER_ID, TEST_EMAIL)

        check(
            "missing profile is inserted once with verified values",
            len(missing.calls) == 1
            and missing.calls[0][0] == {
                "id": TEST_USER_ID,
                "email": TEST_EMAIL,
            }
            and missing.calls[0][1].get(
                "ignore_duplicates"
            )
            is True,
            str(missing.calls),
            failures,
        )
    finally:
        restore(original)

    email_less = FakeProfileSupabase([])
    auth_service.get_supabase = lambda: email_less

    try:
        raised = False
        try:
            ensure_user_profile(TEST_USER_ID, None)
        except UserProfileError:
            raised = True

        check(
            "profile without an email claim is refused",
            raised and email_less.calls == [],
            str(email_less.calls),
            failures,
        )
    finally:
        restore(original)


def test_route_protection(failures):
    print("\n" + "-" * 66)
    print("D - ROUTE PROTECTION")
    print("-" * 66)

    client = TestClient(app)

    protected = [
        ("GET", "/api/assessment/history"),
        ("GET", f"/api/assessment/{TEST_USER_ID}"),
        ("GET", f"/api/assessment/{TEST_USER_ID}/assets"),
    ]

    for method, path in protected:
        response = client.request(method, path)

        check(
            f"{method} {path} without a token returns 401",
            response.status_code == 401,
            f"{response.status_code}",
            failures,
        )

        check(
            f"{method} {path} sends a Bearer challenge",
            response.headers.get("www-authenticate")
            == "Bearer",
            str(response.headers.get("www-authenticate")),
            failures,
        )

    response = client.get("/health")

    check(
        "GET /health stays public",
        response.status_code == 200
        and response.json() == {"status": "ok"},
        f"{response.status_code} {response.json()}",
        failures,
    )

    response = client.get("/")

    check(
        "GET / stays public",
        response.status_code == 200,
        str(response.status_code),
        failures,
    )

    before, after = png_pair()

    response = client.post(
        "/api/assessment/upload",
        files={
            "before_image": before,
            "after_image": after,
        },
    )

    check(
        "upload without a token returns 401",
        response.status_code == 401,
        str(response.status_code),
        failures,
    )

    response = client.post(
        "/api/assessment/analyze",
        files={
            "before_image": before,
            "after_image": after,
        },
    )

    check(
        "analyze without a token returns 401",
        response.status_code == 401,
        str(response.status_code),
        failures,
    )

    # A well-formed but invalid token must be rejected as 401.
    original = auth_service.verify_access_token
    auth_service.verify_access_token = lambda token: _raise(
        InvalidAccessTokenError("bad")
    )

    try:
        response = client.get(
            "/api/assessment/history",
            headers={"Authorization": "Bearer not.a.real.token"},
        )

        check(
            "invalid token returns 401",
            response.status_code == 401,
            str(response.status_code),
            failures,
        )
    finally:
        auth_service.verify_access_token = original


def test_derived_ownership(failures):
    print("\n" + "-" * 66)
    print("E - DERIVED OWNERSHIP")
    print("-" * 66)

    client = TestClient(app)
    app.dependency_overrides[get_current_user] = (
        lambda: AuthenticatedUser(
            user_id=TEST_USER_ID, email=TEST_EMAIL
        )
    )

    captured = {}

    original_run = assessment_api.run_assessment
    original_history = assessment_api.get_assessment_history
    original_by_id = assessment_api.get_assessment_by_id
    original_assets = assessment_api.get_assessment_assets

    image_info = {
        "filename": "before.png",
        "content_type": "image/png",
        "format": "PNG",
        "width": 256,
        "height": 256,
        "size_bytes": 64,
    }

    class_stats = {"pixels": 0, "percentage": 0.0}

    def fake_run(before_info, after_info, user_id, user_email=None):
        captured["run"] = (user_id, user_email)
        return {
            "status": "success",
            "message": "ok",
            "before_image": dict(image_info),
            "after_image": dict(image_info),
            "prediction": {
                "mask_shape": [256, 256],
                "predicted_classes": [0],
            },
            "statistics": {
                "total_pixels": 65536,
                "damage_pixels": 0,
                "damage_percentage": 0.0,
                "damage_level": "none",
                "classes": {
                    name: dict(class_stats)
                    for name in (
                        "background",
                        "no_damage",
                        "minor_damage",
                        "major_damage",
                        "destroyed",
                    )
                },
            },
            "model": {
                "name": "test-model",
                "input_channels": 6,
                "output_classes": 5,
                "checkpoint": "test.pt",
            },
        }

    detail_row = {
        "id": "own-id",
        "user_id": TEST_USER_ID,
        "model_id": None,
        "status": "completed",
        "damage_level": "none",
        "damage_percentage": 0.0,
        "total_pixels": 1,
        "damage_pixels": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "assessment_models": None,
        "assessment_images": [],
        "assessment_predictions": None,
        "assessment_class_statistics": [],
    }

    def fake_history(user_id, limit):
        captured["history"] = (user_id, limit)
        return []

    def fake_by_id(assessment_id):
        return captured.get("row")

    def fake_assets(assessment_id, request_user_id):
        captured["assets"] = request_user_id
        raise AssetAccessDenied("denied")

    assessment_api.run_assessment = fake_run
    assessment_api.get_assessment_history = fake_history
    assessment_api.get_assessment_by_id = fake_by_id
    assessment_api.get_assessment_assets = fake_assets

    try:
        before, after = png_pair()
        response = client.post(
            "/api/assessment/analyze",
            files={
                "before_image": before,
                "after_image": after,
            },
        )

        check(
            "analyze passes the verified user id, not a client value",
            response.status_code == 200
            and captured.get("run")
            == (TEST_USER_ID, TEST_EMAIL),
            f"{response.status_code} {captured.get('run')}",
            failures,
        )

        response = client.get("/api/assessment/history")

        check(
            "history is filtered by the verified user id",
            response.status_code == 200
            and captured.get("history") == (TEST_USER_ID, None),
            f"{response.status_code} {captured.get('history')}",
            failures,
        )

        response = client.get(
            "/api/assessment/other-id"
        )

        check(
            "unknown assessment returns 404",
            response.status_code == 404,
            str(response.status_code),
            failures,
        )

        captured["row"] = {
            "id": "other-id",
            "user_id": "different-user",
        }

        response = client.get(
            "/api/assessment/other-id"
        )

        check(
            "assessment owned by another user returns 403",
            response.status_code == 403,
            str(response.status_code),
            failures,
        )

        captured["row"] = detail_row

        response = client.get("/api/assessment/own-id")

        check(
            "the owner can read their own assessment",
            response.status_code == 200
            and response.json()["id"] == "own-id",
            str(response.status_code),
            failures,
        )

        response = client.get(
            "/api/assessment/own-id/assets"
        )

        check(
            "assets endpoint passes the verified requester id",
            response.status_code == 403
            and captured.get("assets") == TEST_USER_ID,
            f"{response.status_code} {captured.get('assets')}",
            failures,
        )

        check(
            "403 body never leaks the access token",
            "Bearer" not in response.text
            and "not.a.real.token" not in response.text,
            response.text,
            failures,
        )
    finally:
        assessment_api.run_assessment = original_run
        assessment_api.get_assessment_history = original_history
        assessment_api.get_assessment_by_id = original_by_id
        assessment_api.get_assessment_assets = original_assets
        app.dependency_overrides.pop(get_current_user, None)


def test_no_token_logging(failures):
    print("\n" + "-" * 66)
    print("F - NO TOKEN LEAK")
    print("-" * 66)

    client = TestClient(app)

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = Capture()
    logger = logging.getLogger()
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)

    secret = "eyJ.secret.access.token"

    original = auth_service.verify_access_token
    auth_service.verify_access_token = lambda token: _raise(
        InvalidAccessTokenError("bad")
    )

    try:
        response = client.get(
            "/api/assessment/history",
            headers={"Authorization": f"Bearer {secret}"},
        )

        check(
            "rejected token never appears in the response",
            secret not in response.text,
            response.text,
            failures,
        )

        check(
            "rejected token never appears in the logs",
            all(secret not in line for line in records),
            "\n".join(records) or "(no log records)",
            failures,
        )
    finally:
        auth_service.verify_access_token = original
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_frontend_has_no_service_role(failures):
    print("\n" + "-" * 66)
    print("G - FRONTEND SECRET HYGIENE")
    print("-" * 66)

    import os

    forbidden = (
        "SUPABASE_SERVICE_ROLE_KEY",
        "service_role",
    )

    offenders = []

    for root_dir in FRONTEND_DIRS:
        for base, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if not filename.endswith(
                    (".ts", ".tsx", ".js", ".jsx")
                ):
                    continue

                path = os.path.join(base, filename)

                with open(
                    path, "r", encoding="utf-8", errors="ignore"
                ) as handle:
                    text = handle.read()

                if any(term in text for term in forbidden):
                    offenders.append(path)

    check(
        "frontend source contains no service-role reference",
        not offenders,
        str(offenders),
        failures,
    )

    # The public client must only ever read the public vars.
    client_path = os.path.join(
        "frontend", "src", "lib", "supabase", "client.ts"
    )

    with open(
        client_path, "r", encoding="utf-8"
    ) as handle:
        client_source = handle.read()

    check(
        "supabase client uses only NEXT_PUBLIC_ variables",
        "NEXT_PUBLIC_SUPABASE_URL" in client_source
        and "NEXT_PUBLIC_SUPABASE_ANON_KEY" in client_source,
        "public vars present",
        failures,
    )


def main():
    failures = []

    print("=" * 66)
    print("STEP 8.4 AUTHENTICATION TEST")
    print("=" * 66)

    test_verify_access_token(failures)
    test_get_current_user(failures)
    test_ensure_user_profile(failures)
    test_route_protection(failures)
    test_derived_ownership(failures)
    test_no_token_logging(failures)
    test_frontend_has_no_service_role(failures)

    print("\n" + "=" * 66)

    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("RESULT: ALL AUTHENTICATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
