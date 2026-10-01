"""
Supabase authentication for the FastAPI backend.

The client never proves identity by sending a user_id. It sends
the Supabase access token it already holds:

    Authorization: Bearer <supabase access token>

This module verifies that token against the project's Supabase
configuration, derives the authenticated user id from the
verified claims, and exposes it to the API as a FastAPI
dependency. The verified id is the only value ever used as
request_user_id, so ownership can never be chosen by a client.

Verification uses supabase-py's own `auth.get_claims`, the
Supabase-supported server-side path:

- a symmetric (HS256) or unidentified token is validated by the
  Auth server through `get_user`, and
- an asymmetric token is validated locally against the project's
  JWKS signing key, after which its expiry is checked.

The payload is never trusted without signature validation, and
access tokens are never logged.
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from supabase_auth.errors import AuthError

from backend.services.supabase_service import get_supabase


# ============================================================
# TYPES
# ============================================================

@dataclass(frozen=True)
class AuthenticatedUser:
    """
    The identity proven by a verified Supabase access token.
    """

    user_id: str
    email: Optional[str]


class InvalidAccessTokenError(Exception):
    """
    The supplied access token is missing, malformed, expired,
    not a valid signature, or does not describe an
    authenticated Supabase user.
    """


class UserProfileError(Exception):
    """
    A public.users row could not be created for a verified
    user.
    """


# ============================================================
# TOKEN VERIFICATION
# ============================================================

def verify_access_token(access_token: str) -> AuthenticatedUser:
    """
    Verify a Supabase access token and return its identity.

    Raises InvalidAccessTokenError for any token that is not a
    currently valid, signature-verified authenticated-user
    token. Unexpected failures (for example a Supabase outage)
    are intentionally allowed to propagate so they are not
    disguised as an authentication failure.
    """

    token = (access_token or "").strip()

    if not token:
        raise InvalidAccessTokenError(
            "Access token is empty."
        )

    try:
        claims_response = get_supabase().auth.get_claims(
            token
        )
    except AuthError as error:
        # Axios-style auth failures are expected for bad
        # tokens. The original error is chained for debugging
        # but never includes the token in the response.
        raise InvalidAccessTokenError(
            "Invalid or expired access token."
        ) from error

    if claims_response is None:
        raise InvalidAccessTokenError(
            "Invalid or expired access token."
        )

    claims = claims_response.claims

    user_id = claims.get("sub")
    role = claims.get("role")

    if not isinstance(user_id, str) or not user_id:
        raise InvalidAccessTokenError(
            "Token does not identify a user."
        )

    if role != "authenticated":
        raise InvalidAccessTokenError(
            "Token is not an authenticated user token."
        )

    email = claims.get("email")

    return AuthenticatedUser(
        user_id=user_id,
        email=email if isinstance(email, str) else None,
    )


# ============================================================
# PROFILE ROW
#
# assessments.user_id has a foreign key to public.users.id,
# which in turn references auth.users.id. Supabase Auth creates
# auth.users on sign-up, so the application-facing public.users
# row is the only missing link before an assessment can be
# written for that user.
#
# The row is created from the verified token only. A client can
# never choose the id or the email stored here.
# ============================================================

def ensure_user_profile(
    user_id: str,
    email: Optional[str],
) -> None:
    """
    Ensure a public.users row exists for a verified user.

    Existing profiles are left untouched. A missing profile is
    inserted once using the id and email derived from the
    verified token.
    """

    supabase = get_supabase()

    existing = (
        supabase
        .table("users")
        .select("id")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )

    if existing.data:
        return

    if not email:
        raise UserProfileError(
            "Cannot create a user profile without an email "
            "claim."
        )

    (
        supabase
        .table("users")
        .upsert(
            {"id": user_id, "email": email},
            on_conflict="id",
            ignore_duplicates=True,
        )
        .execute()
    )


# ============================================================
# FASTAPI DEPENDENCY
# ============================================================

_bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        _bearer_scheme
    ),
) -> AuthenticatedUser:
    """
    Resolve the authenticated Supabase user for a request.

    Rejects a missing, malformed, invalid or expired token with
    401. The returned id is derived from the verified token and
    is the only value the assessment layer accepts as the
    request's owner.
    """

    if credentials is None or not credentials.credentials:
        raise _unauthorized("Authentication is required.")

    if credentials.scheme.lower() != "bearer":
        raise _unauthorized(
            "Authorization scheme must be Bearer."
        )

    try:
        return verify_access_token(credentials.credentials)
    except InvalidAccessTokenError as error:
        raise _unauthorized(
            "Invalid or expired access token."
        ) from error
