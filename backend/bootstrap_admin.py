#!/usr/bin/env python3
"""Grant LayerPainter admin on the shared Supabase login.

Creates the auth user only when the email does not exist yet. An existing
login keeps its password and app metadata. This does not send email and does
not print secrets.

Run it only when you intend to change the shared login directory:

    ADMIN_EMAIL=john@streamtime.com.au \\
    NEXT_PUBLIC_SUPABASE_URL=... \\
    SUPABASE_SERVICE_ROLE_KEY=... \\
    DATABASE_URL=... \\
    python backend/bootstrap_admin.py
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def supabase_url() -> str:
    return (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip().rstrip("/")


def request_json(method: str, url: str, service_key: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("apikey", service_key)
    request.add_header("Authorization", f"Bearer {service_key}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise SystemExit(f"Supabase admin request failed ({exc.code})") from exc
    return json.loads(raw) if raw else {}


def find_user(base: str, service_key: str, email: str) -> dict | None:
    for page in range(1, 21):
        query = urllib.parse.urlencode({"page": page, "per_page": 200})
        body = request_json("GET", f"{base}/auth/v1/admin/users?{query}", service_key)
        users = body.get("users") if isinstance(body, dict) else None
        if not isinstance(users, list):
            raise SystemExit("Unexpected user list from Supabase")
        for user in users:
            if isinstance(user, dict) and str(user.get("email") or "").lower() == email:
                return user
        if len(users) < 200:
            return None
    return None


def main() -> None:
    email = (os.getenv("ADMIN_EMAIL") or "john@streamtime.com.au").strip().lower()
    base = supabase_url()
    if not base:
        raise SystemExit("NEXT_PUBLIC_SUPABASE_URL is not set")
    service_key = require_env("SUPABASE_SERVICE_ROLE_KEY")
    database_url = require_env("DATABASE_URL")

    created = False
    user = find_user(base, service_key, email)
    if user is None:
        created = True
        body = request_json(
            "POST",
            f"{base}/auth/v1/admin/users",
            service_key,
            {
                "email": email,
                "password": secrets.token_urlsafe(32),
                "email_confirm": True,
                "user_metadata": {"app": "layerpainter"},
            },
        )
        user = body.get("user") if isinstance(body.get("user"), dict) else body
    user_id = str((user or {}).get("id") or "")
    if not user_id:
        raise SystemExit("Supabase did not return a user id")

    import psycopg

    with psycopg.connect(database_url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO layerpainter.user_roles (user_id, role, display_name)
                VALUES (%s, 'admin', 'John')
                ON CONFLICT (user_id) DO UPDATE SET role = 'admin'
                """,
                (user_id,),
            )
        conn.commit()

    if created:
        print(f"Created a LayerPainter login for {email} and granted admin.")
        print("The password was not saved here. Use Forgot password to choose one.")
    else:
        print(f"Granted LayerPainter admin to the existing login {email}.")
        print("The password and other app memberships were left unchanged.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Could not grant admin: {exc.__class__.__name__}", file=sys.stderr)
        raise SystemExit(1) from exc
