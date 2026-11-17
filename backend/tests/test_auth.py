"""Session cookie and project ownership."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import auth
import project_store
from fastapi import HTTPException


OWNER_A = "11111111-1111-4111-8111-111111111111"
OWNER_B = "22222222-2222-4222-8222-222222222222"
SECRET = "test-secret-value-16"


def _user(user_id: str) -> auth.SessionUser:
    return auth.SessionUser(id=user_id, email=f"{user_id[:8]}@example.com", role="user")


class TestSessionCookie(unittest.TestCase):
    def test_round_trip(self):
        token = auth.sign_session(OWNER_A, "john@streamtime.com.au", SECRET, exp=2_000_000_000)
        payload = auth.read_session(token, SECRET)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["uid"], OWNER_A)
        self.assertEqual(payload["email"], "john@streamtime.com.au")

    def test_tampered_signature_is_rejected(self):
        token = auth.sign_session(OWNER_A, "john@streamtime.com.au", SECRET, exp=2_000_000_000)
        body, signature = token.split(".", 1)
        self.assertIsNone(auth.read_session(f"{body}.{signature[:-1]}A", SECRET))

    def test_expired_cookie_is_rejected(self):
        token = auth.sign_session(OWNER_A, "john@streamtime.com.au", SECRET, exp=1)
        self.assertIsNone(auth.read_session(token, SECRET))

    def test_auth_disabled_by_default(self):
        previous = os.environ.pop("AUTH_DISABLED", None)
        try:
            self.assertTrue(auth.auth_disabled())
        finally:
            if previous is not None:
                os.environ["AUTH_DISABLED"] = previous


class TestProjectOwnership(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original = project_store.PROJECTS_DIR
        project_store.PROJECTS_DIR = Path(self.tmp.name)

    def tearDown(self):
        project_store.PROJECTS_DIR = self.original
        self.tmp.cleanup()

    def _write(self, project_id: str, owner_id: str | None) -> None:
        manifest = {"name": project_id, "createdAt": 1}
        if owner_id:
            manifest["ownerId"] = owner_id
        project_store.save_manifest(project_id, manifest)

    def test_list_is_limited_to_the_owner(self):
        self._write("project-a", OWNER_A)
        self._write("project-b", OWNER_B)
        listed = [item["sessionId"] for item in project_store.list_projects(OWNER_A)]
        self.assertEqual(listed, ["project-a"])

    def test_client_cannot_replace_owner(self):
        self._write("project-a", OWNER_A)
        project_store.upsert_manifest_from_client_payload(
            "project-a",
            {"name": "Renamed", "ownerId": OWNER_B},
        )
        manifest = project_store.load_manifest("project-a")
        assert manifest is not None
        self.assertEqual(manifest["ownerId"], OWNER_A)
        self.assertEqual(manifest["name"], "Renamed")

    def test_other_member_is_forbidden(self):
        self._write("project-a", OWNER_A)
        with self.assertRaises(HTTPException) as raised:
            auth.require_project_access(_user(OWNER_B), "project-a")
        self.assertEqual(raised.exception.status_code, 403)

    def test_stamp_assigns_a_new_project_and_refuses_a_taken_one(self):
        auth.stamp_project_owner(_user(OWNER_A), "project-new")
        manifest = project_store.load_manifest("project-new")
        assert manifest is not None
        self.assertEqual(manifest["ownerId"], OWNER_A)
        with self.assertRaises(HTTPException):
            auth.stamp_project_owner(_user(OWNER_B), "project-new")

    def test_unowned_projects_go_to_the_bootstrap_admin(self):
        self._write("legacy", None)
        self._write("kept", OWNER_B)
        assigned = project_store.assign_unowned_projects(OWNER_A)
        self.assertEqual(assigned, 1)
        legacy = project_store.load_manifest("legacy")
        kept = project_store.load_manifest("kept")
        assert legacy is not None and kept is not None
        self.assertEqual(legacy["ownerId"], OWNER_A)
        self.assertEqual(kept["ownerId"], OWNER_B)

    def test_collaborator_sees_the_project_and_cannot_take_it_over(self):
        self._write("project-a", OWNER_A)
        project_store.add_collaborator("project-a", OWNER_B)
        listed = [item["sessionId"] for item in project_store.list_projects(OWNER_B)]
        self.assertEqual(listed, ["project-a"])
        auth.require_project_access(_user(OWNER_B), "project-a")
        project_store.upsert_manifest_from_client_payload(
            "project-a",
            {"name": "Shared edit", "collaboratorIds": [], "ownerId": OWNER_B},
        )
        manifest = project_store.load_manifest("project-a")
        assert manifest is not None
        self.assertEqual(manifest["ownerId"], OWNER_A)
        self.assertEqual(manifest["collaboratorIds"], [OWNER_B])
        self.assertEqual(manifest["name"], "Shared edit")
        with self.assertRaises(HTTPException) as raised:
            auth.require_project_owner(_user(OWNER_B), "project-a")
        self.assertEqual(raised.exception.status_code, 403)
        project_store.remove_collaborator("project-a", OWNER_B)
        self.assertEqual(project_store.list_projects(OWNER_B), [])

    def test_disabled_auth_does_not_stamp_an_owner(self):
        auth.stamp_project_owner(auth.disabled_user(), "open")
        self.assertFalse(project_store.project_exists("open"))


class TestCookieAgreesWithJavascript(unittest.TestCase):
    def test_python_accepts_the_node_cookie(self):
        node = Path(__file__).resolve().parents[2] / "frontend" / "lib" / "auth" / "session-cookie.mjs"
        if not node.is_file():
            self.skipTest("session cookie script is added with the frontend")
        script = (
            "import { signSession } from './frontend/lib/auth/session-cookie.mjs';"
            "const token = await signSession("
            "{ uid: '%s', email: 'john@streamtime.com.au', exp: 2000000000 }, '%s');"
            "process.stdout.write(token);"
        ) % (OWNER_A, SECRET)
        completed = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            cwd=node.parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(completed.stderr)
        payload = auth.read_session(completed.stdout.strip(), SECRET)
        self.assertEqual(payload["uid"], OWNER_A)
        token = auth.sign_session(OWNER_A, "john@streamtime.com.au", SECRET, exp=2_000_000_000)
        self.assertEqual(token, completed.stdout.strip())


class TestResolveUser(unittest.TestCase):
    def tearDown(self):
        auth.lookup_role.override = None
        os.environ.pop("AUTH_DISABLED", None)
        os.environ.pop("AUTH_SECRET", None)

    def test_missing_cookie_is_signed_out_when_auth_is_on(self):
        os.environ["AUTH_DISABLED"] = "0"
        os.environ["AUTH_SECRET"] = SECRET

        class Request:
            cookies = {}

        self.assertIsNone(auth.resolve_user(Request()))

    def test_cookie_requires_a_layerpainter_role(self):
        os.environ["AUTH_DISABLED"] = "0"
        os.environ["AUTH_SECRET"] = SECRET
        token = auth.sign_session(OWNER_A, "john@streamtime.com.au", SECRET, exp=2_000_000_000)

        class Request:
            cookies = {auth.COOKIE_NAME: token}

        auth.lookup_role.override = lambda user_id: {"role": "admin", "display_name": "John"} if user_id == OWNER_A else None
        user = auth.resolve_user(Request())
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.id, OWNER_A)
        self.assertEqual(user.role, "admin")

        auth.lookup_role.override = lambda user_id: None
        self.assertIsNone(auth.resolve_user(Request()))


if __name__ == "__main__":
    unittest.main()
