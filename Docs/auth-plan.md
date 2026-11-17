# LayerPainter — User accounts

Status: implemented in this repository. The shared database and the live server are unchanged until that step is approved.

Each member has a private project workspace. Paint libraries stay shared among signed-in members. There is no public signup.

## Login

Sign-in uses the existing Supabase Auth users. One email has one password across the apps that use this login, including Farm Cashbook. A password chosen in LayerPainter replaces that shared password.

LayerPainter membership is `layerpainter.user_roles` (`user_id`, `role`, `display_name`, `created_at`). It does not reference `auth.users`, because application migrations cannot point at another schema. The migration is [supabase/migrations/20261117100000_layerpainter_user_roles.sql](../supabase/migrations/20261117100000_layerpainter_user_roles.sql) and passes `sql_guard.py`. It is not applied here.

New LayerPainter-only people are created with `user_metadata.app = "layerpainter"`, so Cashbook does not give them a Cashbook role. An email that already has a login keeps its password and its `app` metadata. Removing someone deletes only the LayerPainter role. Their Supabase user and their project files stay.

`john@streamtime.com.au` is the admin. Grant that role with `backend/bootstrap_admin.py` when the database change is approved. The script does not print a password. If the email already has a login, sign in with it. If the script had to create the login, use Forgot password to choose one.

## Forgotten password

This follows Farm Cashbook:

1. `/forgot-password` accepts an email.
2. The server sends a recovery link only when that email has a LayerPainter role. Anyone else sees the same confirmation and no email is sent.
3. The link is created with `auth.admin.generateLink({ type: "recovery" })` and mailed through Resend from `LayerPainter <john@margies.app>`.
4. Opening `/reset-password` does not use the link. Continue calls `verifyOtp`, then the person sets a password with `updateUser`.
5. The optional email from the Users screen is the same link, with welcome wording.

Production links use `https://layerpainter.margies.app`. Localhost uses the request host.

## Projects

Project manifests store `ownerId`. The project list and every project or session route are limited to that id. Another member receives 403. The admin does not get a view of other people's projects. A client cannot change `ownerId`.

When auth is on and `BOOTSTRAP_ADMIN_USER_ID` is set, startup assigns that id to projects that have no owner. Paint routes require a signed-in member and still use the shared `data/paint/` files.

`AUTH_DISABLED=1` (the default) leaves local use as today's open workspace. The hosted app sets `AUTH_DISABLED=0` and `NEXT_PUBLIC_AUTH_REQUIRED=1` together, after the schema and admin user exist.

## Session

Hosted pages call same-origin `/api/...`. Next.js keeps `/api/auth/*` and `/api/users/*`, and rewrites every other `/api` request to FastAPI. After Supabase sign-in, the Next app sets an httpOnly `lp_session` cookie. FastAPI checks that cookie and reads the role again, so removing a member takes effect on the next request. Logout clears the cookie and the browser project list.

## Hosted switch-on

Do not point the live app at this until you mean to require login.

1. Create the `layerpainter_app` login role. Do not store its password in git.
2. Apply the migration.
3. Put `DATABASE_URL`, `AUTH_SECRET`, the Supabase URL, anon key, and service role key, and `RESEND_API_KEY` in the server env files. The service role key stays on the Next.js server.
4. Run `backend/bootstrap_admin.py` for `john@streamtime.com.au`.
5. Set `BOOTSTRAP_ADMIN_USER_ID` to that user's id and `AUTH_DISABLED=0` on the backend.
6. Build the frontend with `NEXT_PUBLIC_AUTH_REQUIRED=1` and an empty `NEXT_PUBLIC_API_BASE_URL`.
