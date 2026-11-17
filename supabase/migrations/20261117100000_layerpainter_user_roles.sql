-- LayerPainter membership. Passwords stay in Supabase Auth (auth.users).
-- This table only records who may use LayerPainter, and whether they are an admin.
-- No foreign key to auth.users: application migrations cannot reference another schema.
-- Create the login role layerpainter_app before applying this file. Do not apply it
-- to the shared database until that step is approved.

CREATE SCHEMA IF NOT EXISTS layerpainter;

REVOKE ALL ON SCHEMA layerpainter FROM PUBLIC;

CREATE TABLE layerpainter.user_roles (
    user_id uuid PRIMARY KEY,
    role text NOT NULL DEFAULT 'user',
    display_name text,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT user_roles_role_check CHECK (role IN ('admin', 'user'))
);

GRANT USAGE ON SCHEMA layerpainter TO layerpainter_app;
GRANT ALL ON TABLE layerpainter.user_roles TO layerpainter_app;
