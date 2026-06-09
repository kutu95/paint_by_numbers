# LayerPainter — User accounts (implementation plan)

Last updated: 2026-05-30  
Status: Planning / not implemented  
Blocks: founding member outreach (`Docs/founding-member-brief.md`)

Goal: **minimal multi-user accounts** so each muralist has a private project workspace on hosted infra. Not full commercialisation — no Stripe, tiers, or marketplace yet.

---

## Principles

1. **One account = one tenant** for projects (paint libraries stay shared in v1; per-user paint forks later).
2. **Self-host still works** — `AUTH_DISABLED=1` preserves today's open local workflow.
3. **Small diff** — SQLite user store, JWT in httpOnly cookie, `ownerId` on project manifests.
4. **Fail closed** — when auth is enabled, unauthenticated requests get 401; cross-tenant access gets 403.

---

## Data model

### Users (`data/users.db` — SQLite)

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID |
| `email` | TEXT UNIQUE | Lowercased |
| `password_hash` | TEXT | bcrypt |
| `display_name` | TEXT | Optional |
| `created_at` | INTEGER | ms epoch |

### Projects (existing `manifest.json`)

Add field:

```json
{
  "ownerId": "user-uuid-here"
}
```

Legacy projects without `ownerId`: when auth enabled, assign to a bootstrap admin user on first startup, or treat as inaccessible until migrated.

### Paths (v1 — projects only)

```
data/
  users.db
  projects/{project_id}/     # unchanged layout; ownerId in manifest
  paint/                     # shared read-only libraries (v1)
```

Per-user paint isolation (`data/users/{uid}/paint/`) is a later slice.

---

## API surface (new)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/api/auth/register` | No | Create account, set session cookie |
| POST | `/api/auth/login` | No | Set session cookie |
| POST | `/api/auth/logout` | Yes | Clear cookie |
| GET | `/api/auth/me` | Yes | Current user profile |

### Protected routes (enforce `ownerId` when auth enabled)

All existing `/api/projects/*` routes, plus generate/upload/delete.

Paint routes (`/api/paint/*`) remain **shared read** in v1; calibration write paths get user scoping in a follow-up.

---

## Backend slices

### Slice 1 — Auth module (~1 day)

- `backend/auth.py`: SQLite helpers, bcrypt hash/verify, JWT create/verify
- `backend/requirements.txt`: add `PyJWT`, `bcrypt`
- Env: `AUTH_SECRET`, `AUTH_DISABLED` (default `1` for local dev)
- FastAPI dependency: `get_current_user_optional` / `get_current_user_required`
- CORS: allow credentials from frontend origin

### Slice 2 — Project ownership (~1 day)

- `project_store.list_projects(owner_id)` filter
- `upsert_manifest_from_client_payload` sets `ownerId` on create
- `assert_project_owner(project_id, user_id)` helper
- Wire dependency into all project endpoints in `main.py`
- Migration script: tag existing manifests with bootstrap `ownerId`

### Slice 3 — Frontend auth UI (~1 day)

- `frontend/app/login/page.tsx`, `frontend/app/register/page.tsx`
- `frontend/lib/auth.ts`: `fetchMe`, `login`, `register`, `logout` with `credentials: 'include'`
- `frontend/components/AuthGate.tsx` or layout wrapper: redirect to `/login` when auth required
- Header: signed-in email + logout
- All `fetch(`${API_BASE_URL}/api/...`)` calls use `credentials: 'include'`

### Slice 4 — Hosted config (~0.5 day)

- Production: `AUTH_DISABLED=0`, strong `AUTH_SECRET`
- Same-origin or explicit `NEXT_PUBLIC_API_BASE_URL` + CORS credentials
- Document in `deployment/DEPLOYMENT.md`

---

## Frontend env

| Variable | Purpose |
|----------|---------|
| `NEXT_PUBLIC_AUTH_REQUIRED` | `1` on hosted beta; `0` local self-host |

When `AUTH_REQUIRED=0`, skip login redirect (matches backend `AUTH_DISABLED`).

---

## Security notes (v1)

- httpOnly, Secure (prod), SameSite=Lax cookie
- Password min length 8; no email verification in v1 (add before public launch)
- Rate-limit login/register (simple in-memory or reverse proxy)
- No password reset in v1 — manual support for founding cohort is OK

---

## Explicitly deferred

- OAuth (Google)
- Email verification / magic links
- Stripe + plan entitlements
- Per-user paint library forks
- Admin panel / user management UI
- Password reset flow

---

## Definition of done (founding-beta ready)

- [ ] Two test accounts see different project lists on the same server
- [ ] User A cannot open User B's project URL (403)
- [ ] Register → upload → generate → projection works signed in
- [ ] Logout clears session; back button does not leak data after logout
- [ ] `AUTH_DISABLED=1` local dev unchanged for solo self-host

---

## Suggested build order

1. Slice 1 (auth endpoints + cookie)
2. Slice 2 (project scoping) — **highest risk; do immediately after auth**
3. Slice 3 (login UI + gate)
4. Slice 4 (deploy config)
5. Then: founding invites, then Stripe

---

## Related docs

- `Docs/founding-member-brief.md`
- `Docs/commercialisation.md` — Phase 1 commercial product table
