"""Signed-in member checks for LayerPainter.

The browser session is an HMAC cookie set by the Next.js app after Supabase
Auth succeeds. This module checks that cookie and the layerpainter.user_roles
row. It does not store passwords.

AUTH_DISABLED=1 (the default) leaves the local single-user workflow open.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass

from fastapi import HTTPException, Request

import project_store

logger = logging.getLogger(__name__)

COOKIE_NAME = "lp_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
_PROJECT_PATH = re.compile(r"^/api/(?:projects|sessions)/([^/]+)")


class MembershipUnavailable(Exception):
    """The role table could not be read. Callers should fail closed."""


@dataclass(frozen=True)
class SessionUser:
    id: str
    email: str
    role: str
    auth_disabled: bool = False


def auth_disabled() -> bool:
    return os.getenv("AUTH_DISABLED", "1").strip().lower() in {"1", "true", "yes", "on"}


def disabled_user() -> SessionUser:
    return SessionUser(id="", email="", role="user", auth_disabled=True)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _payload_json(email: str, exp: int, uid: str) -> str:
    return json.dumps({"email": email, "exp": exp, "uid": uid}, separators=(",", ":"))


def sign_session(uid: str, email: str, secret: str, exp: int | None = None) -> str:
    expires = int(time.time()) + SESSION_TTL_SECONDS if exp is None else int(exp)
    body = _b64encode(_payload_json(email, expires, uid).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64encode(signature)}"


def read_session(token: str, secret: str) -> dict | None:
    if not token or not secret or "." not in token:
        return None
    body, signature = token.split(".", 1)
    if not body or not signature:
        return None
    expected = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    try:
        given = _b64decode(signature)
    except Exception:
        return None
    if not hmac.compare_digest(expected, given):
        return None
    try:
        payload = json.loads(_b64decode(body))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    uid = payload.get("uid")
    email = payload.get("email")
    exp = payload.get("exp")
    if not isinstance(uid, str) or not isinstance(email, str) or not isinstance(exp, int):
        return None
    if exp < int(time.time()):
        return None
    try:
        uuid.UUID(uid)
    except ValueError:
        return None
    return {"uid": uid, "email": email, "exp": exp}


def lookup_role(user_id: str) -> dict | None:
    """Return {role, display_name} or None when the person is not a member.

    Raises MembershipUnavailable when the directory cannot be queried.
    """
    override = getattr(lookup_role, "override", None)
    if override is not None:
        return override(user_id)

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise MembershipUnavailable("DATABASE_URL is not set")
    try:
        uuid.UUID(user_id)
    except ValueError:
        return None
    try:
        import psycopg
    except ImportError as exc:
        raise MembershipUnavailable("psycopg is not installed") from exc
    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, display_name FROM layerpainter.user_roles WHERE user_id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
    except Exception as exc:
        logger.exception("Could not read layerpainter.user_roles")
        raise MembershipUnavailable("membership lookup failed") from exc
    if row is None:
        return None
    return {"role": row[0], "display_name": row[1]}


def current_user(request: Request) -> SessionUser:
    user = getattr(request.state, "user", None)
    if isinstance(user, SessionUser):
        return user
    if auth_disabled():
        return disabled_user()
    raise HTTPException(status_code=401, detail="Sign in required")


def resolve_user(request: Request) -> SessionUser | None:
    if auth_disabled():
        return disabled_user()
    secret = os.getenv("AUTH_SECRET", "").strip()
    if len(secret) < 16:
        logger.error("AUTH_SECRET is missing or shorter than 16 characters")
        return None
    payload = read_session(request.cookies.get(COOKIE_NAME, ""), secret)
    if payload is None:
        return None
    role = lookup_role(payload["uid"])
    if role is None:
        return None
    if role.get("role") not in {"admin", "user"}:
        return None
    return SessionUser(id=payload["uid"], email=payload["email"], role=role["role"])


def project_id_from_path(path: str) -> str | None:
    match = _PROJECT_PATH.match(path)
    if not match:
        return None
    project_id = match.group(1).strip()
    return project_id or None


def require_project_access(user: SessionUser, project_id: str) -> None:
    if user.auth_disabled or not project_id:
        return
    if user.role == "admin":
        return
    if not project_store.project_exists(project_id):
        return
    manifest = project_store.load_manifest(project_id) or {}
    if project_store.user_can_access_manifest(manifest, user.id):
        return
    raise HTTPException(status_code=403, detail="You do not have access to this project")


def require_project_member(user: SessionUser, project_id: str) -> None:
    """Owner or collaborator. An admin can read every project, but not change one they are not on."""
    if user.auth_disabled or not project_id:
        return
    if not project_store.project_exists(project_id):
        return
    manifest = project_store.load_manifest(project_id) or {}
    if project_store.user_can_access_manifest(manifest, user.id):
        return
    raise HTTPException(status_code=403, detail="You do not have access to this project")


def require_project_owner(user: SessionUser, project_id: str) -> None:
    if user.auth_disabled or not project_id:
        return
    if not project_store.project_exists(project_id):
        return
    manifest = project_store.load_manifest(project_id) or {}
    if manifest.get("ownerId") != user.id:
        raise HTTPException(status_code=403, detail="Only the project owner can do that")


def stamp_project_owner(user: SessionUser, project_id: str) -> None:
    """Give a new or unowned project to this member. Refuse someone else's project."""
    if user.auth_disabled or not project_id:
        return
    manifest = project_store.load_manifest(project_id) or {
        "projectId": project_id,
        "createdAt": int(time.time() * 1000),
        "name": "Untitled",
    }
    owner = manifest.get("ownerId")
    if owner and owner != user.id:
        if user.id in project_store.collaborator_ids(manifest):
            return
        raise HTTPException(status_code=403, detail="You do not have access to this project")
    if owner == user.id:
        return
    manifest["ownerId"] = user.id
    project_store.save_manifest(project_id, manifest)
