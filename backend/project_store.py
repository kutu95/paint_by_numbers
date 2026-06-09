"""Project directory bundles — single source of truth for mural project data."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paint_manager import atomic_write, slugify

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROJECTS_DIR = Path(__file__).parent.parent / "data" / "projects"
LEGACY_SESSIONS_DIR = Path(__file__).parent.parent / "data" / "sessions"

SOURCE_ORIENTED = "oriented.jpg"
SOURCE_PRIORITY_REGION = "priority_region.png"
SESSION_SNAPSHOT = "session.json"


def project_root(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def source_dir(project_id: str) -> Path:
    return project_root(project_id) / "source"


def artifacts_dir(project_id: str) -> Path:
    return project_root(project_id) / "artifacts"


def manifest_path(project_id: str) -> Path:
    return project_root(project_id) / "manifest.json"


def state_path(project_id: str) -> Path:
    return project_root(project_id) / "state.json"


def ensure_project_dirs(project_id: str) -> Path:
    root = project_root(project_id)
    root.mkdir(parents=True, exist_ok=True)
    source_dir(project_id).mkdir(parents=True, exist_ok=True)
    artifacts_dir(project_id).mkdir(parents=True, exist_ok=True)
    return root


def project_exists(project_id: str) -> bool:
    return manifest_path(project_id).is_file()


def load_manifest(project_id: str) -> Optional[dict[str, Any]]:
    path = manifest_path(project_id)
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("Failed to load manifest for %s: %s", project_id, e)
        return None


def save_manifest(project_id: str, manifest: dict[str, Any]) -> None:
    ensure_project_dirs(project_id)
    manifest = dict(manifest)
    manifest["schemaVersion"] = SCHEMA_VERSION
    manifest["projectId"] = project_id
    manifest["sessionId"] = project_id  # legacy field for clients
    manifest["updatedAt"] = int(datetime.now().timestamp() * 1000)
    atomic_write(manifest_path(project_id), manifest)


def load_state(project_id: str) -> dict[str, Any]:
    path = state_path(project_id)
    if not path.is_file():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(project_id: str, state: dict[str, Any]) -> None:
    ensure_project_dirs(project_id)
    atomic_write(state_path(project_id), state)


def find_source_input(project_id: str) -> Optional[Path]:
    """Return the current project source upload (newest file if several exist)."""
    sdir = source_dir(project_id)
    if not sdir.is_dir():
        return None
    candidates: list[Path] = []
    for name in [
        "original.jpg",
        "original.jpeg",
        "original.png",
        "original.webp",
        "original.gif",
        "original.bmp",
        "input.jpg",
        "input.jpeg",
        "input.png",
        "input.webp",
    ]:
        p = sdir / name
        if p.is_file():
            candidates.append(p)
    for f in sdir.iterdir():
        if f.is_file() and f.name.startswith("original.") and f not in candidates:
            candidates.append(f)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def oriented_source_path(project_id: str) -> Path:
    return source_dir(project_id) / SOURCE_ORIENTED


def has_oriented_source(project_id: str) -> bool:
    return oriented_source_path(project_id).is_file()


def priority_region_path(project_id: str) -> Path:
    return source_dir(project_id) / SOURCE_PRIORITY_REGION


def has_priority_region(project_id: str) -> bool:
    return priority_region_path(project_id).is_file()


def save_priority_region(project_id: str, content: bytes) -> Path:
    ensure_project_dirs(project_id)
    dest = priority_region_path(project_id)
    with open(dest, "wb") as f:
        f.write(content)
    return dest


def delete_priority_region(project_id: str) -> None:
    path = priority_region_path(project_id)
    if path.is_file():
        path.unlink()


def save_uploaded_source(project_id: str, content: bytes, filename: str) -> Path:
    ensure_project_dirs(project_id)
    sdir = source_dir(project_id)
    for f in sdir.iterdir():
        if not f.is_file():
            continue
        if f.name.startswith("original.") or f.name.startswith("input."):
            try:
                f.unlink()
            except OSError:
                pass
    oriented = oriented_source_path(project_id)
    if oriented.is_file():
        try:
            oriented.unlink()
        except OSError:
            pass
    ext = Path(filename or "image.jpg").suffix.lower()
    if not ext or ext not in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        ext = ".jpg"
    dest = sdir / f"original{ext}"
    with open(dest, "wb") as f:
        f.write(content)
    return dest


def clear_artifacts(project_id: str) -> None:
    adir = artifacts_dir(project_id)
    if adir.is_dir():
        shutil.rmtree(adir, ignore_errors=True)
    adir.mkdir(parents=True, exist_ok=True)


def artifact_url(project_id: str, filename: str) -> str:
    return f"/api/projects/{project_id}/artifacts/{filename}"


def source_url(project_id: str, filename: str) -> str:
    return f"/api/projects/{project_id}/source/{filename}"


def rewrite_session_urls(session: dict[str, Any], project_id: str) -> dict[str, Any]:
    """Ensure layer and preview URLs point at project bundle paths."""
    out = dict(session)
    out["session_id"] = project_id
    if has_oriented_source(project_id):
        out["original_url"] = source_url(project_id, SOURCE_ORIENTED)
    if out.get("quantized_preview_url"):
        out["quantized_preview_url"] = artifact_url(project_id, "preview.jpg")
    layers = out.get("layers")
    if isinstance(layers, list):
        new_layers = []
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            L = dict(layer)
            idx = L.get("layer_index", 0)
            if L.get("is_finished"):
                prev = artifact_url(project_id, "preview.jpg")
                L["finished_url"] = prev
                L["mask_url"] = prev
                L["outline_thin_url"] = prev
                L["outline_thick_url"] = prev
                L["outline_glow_url"] = prev
            else:
                for key, suffix in [
                    ("mask_url", "_mask.png"),
                    ("mask_pure_url", "_pure_mask.png"),
                    ("outline_thin_url", "_outline_thin.png"),
                    ("outline_thick_url", "_outline_thick.png"),
                    ("outline_glow_url", "_outline_glow.png"),
                ]:
                    if key in L or suffix == "_mask.png":
                        L[key] = artifact_url(project_id, f"layer_{idx}{suffix}")
            new_layers.append(L)
        out["layers"] = new_layers
    return out


def save_session_snapshot(project_id: str, session: dict[str, Any]) -> None:
    ensure_project_dirs(project_id)
    snap = rewrite_session_urls(session, project_id)
    atomic_write(artifacts_dir(project_id) / SESSION_SNAPSHOT, snap)


def load_session_snapshot(project_id: str) -> Optional[dict[str, Any]]:
    path = artifacts_dir(project_id) / SESSION_SNAPSHOT
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return rewrite_session_urls(data, project_id)
    except Exception as e:
        logger.warning("Failed to load session snapshot for %s: %s", project_id, e)
    return None


def build_session_response(project_id: str) -> Optional[dict[str, Any]]:
    snap = load_session_snapshot(project_id)
    if not snap:
        return None
    manifest = load_manifest(project_id)
    if manifest:
        cw = manifest.get("canvasWidthCm") or manifest.get("canvas", {}).get("widthCm")
        ch = manifest.get("canvasHeightCm") or manifest.get("canvas", {}).get("heightCm")
        if cw:
            snap["canvas_width_cm"] = float(cw)
        if ch:
            snap["canvas_height_cm"] = float(ch)
        updated = manifest.get("updatedAt") or manifest.get("createdAt")
        if updated:
            snap["artifacts_version"] = int(updated)
    return snap


def manifest_to_list_item(manifest: dict[str, Any]) -> dict[str, Any]:
    """Shape returned by GET /api/projects for frontend Project type."""
    pid = str(manifest.get("projectId") or manifest.get("sessionId") or "")
    processing = manifest.get("processing") if isinstance(manifest.get("processing"), dict) else {}
    return {
        "sessionId": pid,
        "name": manifest.get("name") or "Untitled",
        "imageFileName": manifest.get("imageFileName") or manifest.get("image", {}).get("fileName", "image"),
        "libraryGroup": manifest.get("libraryGroup") or "default",
        "canvasWidthCm": float(manifest.get("canvasWidthCm") or manifest.get("canvas", {}).get("widthCm") or 0),
        "canvasHeightCm": float(manifest.get("canvasHeightCm") or manifest.get("canvas", {}).get("heightCm") or 0),
        "saturationBoost": float(manifest.get("saturationBoost") or processing.get("saturationBoost") or 1.0),
        "detailLevel": float(manifest.get("detailLevel") or processing.get("detailLevel") or 0.5),
        "easyPainting": bool(
            manifest.get("easyPainting")
            if manifest.get("easyPainting") is not None
            else processing.get("easyPainting", False)
        ),
        "easySimplify": float(
            manifest.get("easySimplify")
            if manifest.get("easySimplify") is not None
            else processing.get("easySimplify", 0.65)
        ),
        "easyFaceDetail": bool(
            manifest.get("easyFaceDetail")
            if manifest.get("easyFaceDetail") is not None
            else processing.get("easyFaceDetail", False)
        ),
        "favorSkinTones": bool(
            manifest.get("favorSkinTones")
            if manifest.get("favorSkinTones") is not None
            else (
                processing.get("favorSkinTones")
                if processing.get("favorSkinTones") is not None
                else True
            )
        ),
        "skinToneStrength": float(
            manifest.get("skinToneStrength")
            if manifest.get("skinToneStrength") is not None
            else processing.get("skinToneStrength", 0.65)
        ),
        "priorityRegionStrength": float(
            manifest.get("priorityRegionStrength")
            if manifest.get("priorityRegionStrength") is not None
            else processing.get("priorityRegionStrength", 0.7)
        ),
        "hasPriorityRegion": has_priority_region(pid) if pid else False,
        "mustIncludeColors": (
            manifest.get("mustIncludeColors")
            if isinstance(manifest.get("mustIncludeColors"), list)
            else (
                processing.get("mustIncludeColors")
                if isinstance(processing.get("mustIncludeColors"), list)
                else []
            )
        ),
        "stylePreset": str(
            manifest.get("stylePreset")
            or processing.get("stylePreset")
            or "natural"
        ),
        "detailEyes": bool(
            manifest.get("detailEyes")
            if manifest.get("detailEyes") is not None
            else processing.get("detailEyes", True)
        ),
        "detailFace": bool(
            manifest.get("detailFace")
            if manifest.get("detailFace") is not None
            else processing.get("detailFace", True)
        ),
        "detailBodyOutline": bool(
            manifest.get("detailBodyOutline")
            if manifest.get("detailBodyOutline") is not None
            else processing.get("detailBodyOutline", True)
        ),
        "createdAt": int(manifest.get("createdAt") or 0),
        "updatedAt": int(manifest.get("updatedAt") or manifest.get("createdAt") or 0),
        "nColors": processing.get("nColors") or manifest.get("nColors"),
        "overpaintMm": processing.get("overpaintMm") or manifest.get("overpaintMm"),
        "orderMode": processing.get("orderMode") or manifest.get("orderMode"),
        "maxSide": processing.get("maxSide") or manifest.get("maxSide"),
        "hasArtifacts": (artifacts_dir(pid) / SESSION_SNAPSHOT).is_file() if pid else False,
        "hasSource": has_oriented_source(pid) or find_source_input(pid) is not None,
        "thumbUrl": source_url(pid, SOURCE_ORIENTED) if has_oriented_source(pid) else None,
    }


def list_projects() -> list[dict[str, Any]]:
    if not PROJECTS_DIR.exists():
        return []
    items: list[dict[str, Any]] = []
    for entry in PROJECTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        manifest = load_manifest(entry.name)
        if manifest:
            items.append(manifest_to_list_item(manifest))
    items.sort(key=lambda x: int(x.get("createdAt", 0) or 0), reverse=True)
    return items


def upsert_manifest_from_client_payload(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    existing = load_manifest(project_id) or {}
    processing_existing = (
        existing.get("processing") if isinstance(existing.get("processing"), dict) else {}
    )
    created = int(existing.get("createdAt") or payload.get("createdAt") or datetime.now().timestamp() * 1000)
    manifest: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "projectId": project_id,
        "sessionId": project_id,
        "name": payload.get("name") or existing.get("name") or "Untitled",
        "createdAt": created,
        "imageFileName": payload.get("imageFileName") or existing.get("imageFileName") or "image",
        "libraryGroup": payload.get("libraryGroup") or existing.get("libraryGroup") or "default",
        "canvasWidthCm": float(payload.get("canvasWidthCm") or existing.get("canvasWidthCm") or 0),
        "canvasHeightCm": float(payload.get("canvasHeightCm") or existing.get("canvasHeightCm") or 0),
        "saturationBoost": float(payload.get("saturationBoost") if payload.get("saturationBoost") is not None else existing.get("saturationBoost", 1.0)),
        "detailLevel": float(payload.get("detailLevel") if payload.get("detailLevel") is not None else existing.get("detailLevel", 0.5)),
    }
    processing: dict[str, Any] = {
        "nColors": payload.get("nColors") if payload.get("nColors") is not None else processing_existing.get("nColors"),
        "overpaintMm": payload.get("overpaintMm") if payload.get("overpaintMm") is not None else processing_existing.get("overpaintMm"),
        "orderMode": payload.get("orderMode") or processing_existing.get("orderMode") or "largest",
        "maxSide": payload.get("maxSide") if payload.get("maxSide") is not None else processing_existing.get("maxSide"),
        "saturationBoost": float(
            payload.get("saturationBoost")
            if payload.get("saturationBoost") is not None
            else processing_existing.get("saturationBoost", existing.get("saturationBoost", 1.0))
        ),
        "detailLevel": float(
            payload.get("detailLevel")
            if payload.get("detailLevel") is not None
            else processing_existing.get("detailLevel", existing.get("detailLevel", 0.5))
        ),
        "maskDilationPx": processing_existing.get("maskDilationPx", 0),
    }
    # Image-tab settings — use `key in payload` so explicit false is preserved.
    for key in (
        "favorSkinTones",
        "skinToneStrength",
        "stylePreset",
        "easyPainting",
        "easySimplify",
        "easyFaceDetail",
        "detailEyes",
        "detailFace",
        "detailBodyOutline",
        "priorityRegionStrength",
        "mustIncludeColors",
    ):
        if key in payload:
            manifest[key] = payload[key]
            processing[key] = payload[key]
        elif key in existing:
            manifest[key] = existing[key]
            processing[key] = existing.get(key, processing_existing.get(key))
    manifest["processing"] = processing
    # Flat fields for backward compatibility in manifest file
    for k in ("nColors", "overpaintMm", "orderMode", "maxSide"):
        v = manifest["processing"].get(k if k != "nColors" else "nColors")
        if v is not None:
            manifest[k] = v
    save_manifest(project_id, manifest)
    return manifest


def apply_processing_to_manifest(project_id: str, processing: dict[str, Any]) -> None:
    manifest = load_manifest(project_id) or {"projectId": project_id, "createdAt": int(datetime.now().timestamp() * 1000)}
    manifest["processing"] = {**(manifest.get("processing") or {}), **processing}
    for k in (
        "nColors",
        "overpaintMm",
        "orderMode",
        "maxSide",
        "saturationBoost",
        "detailLevel",
        "easyPainting",
        "easySimplify",
        "easyFaceDetail",
        "stylePreset",
        "detailEyes",
        "detailFace",
        "detailBodyOutline",
        "favorSkinTones",
        "skinToneStrength",
        "priorityRegionStrength",
        "mustIncludeColors",
    ):
        if k in processing:
            manifest[k] = processing[k]
    manifest["artifacts"] = {
        "generatedAt": datetime.now().isoformat(),
    }
    save_manifest(project_id, manifest)


def delete_project(project_id: str) -> bool:
    root = project_root(project_id)
    if not root.exists():
        return False
    shutil.rmtree(root, ignore_errors=True)
    return True


def resolve_artifact_file(project_id: str, filename: str) -> Optional[Path]:
    """Resolve artifact or legacy flat filename to path under project bundle."""
    adir = artifacts_dir(project_id)
    # legacy session flat names
    legacy_map = {
        "original_oriented.jpg": source_dir(project_id) / SOURCE_ORIENTED,
    }
    if filename in legacy_map:
        p = legacy_map[filename]
        return p if p.is_file() else None
    direct = adir / filename
    if direct.is_file():
        return direct
    return None


def resolve_source_file(project_id: str, filename: str) -> Optional[Path]:
    if filename == "original_oriented.jpg":
        filename = SOURCE_ORIENTED
    path = source_dir(project_id) / filename
    return path if path.is_file() else None


def _migrate_flat_json_project(json_path: Path) -> None:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    pid = str(data.get("sessionId") or "").strip()
    if not pid:
        return
    ensure_project_dirs(pid)
    if not manifest_path(pid).is_file():
        upsert_manifest_from_client_payload(pid, data)
    try:
        json_path.unlink(missing_ok=True)
    except Exception:
        pass
    logger.info("Migrated flat project JSON -> bundle: %s", pid)


def _migrate_legacy_session(session_path: Path) -> None:
    pid = session_path.name
    if not pid or pid.startswith("."):
        return
    ensure_project_dirs(pid)
    sdir = source_dir(pid)
    adir = artifacts_dir(pid)

    # Source files
    for name in ["input.jpg", "input.jpeg", "input.png", "input.webp", "input.gif", "input.bmp"]:
        src = session_path / name
        if src.is_file() and not find_source_input(pid):
            ext = src.suffix or ".jpg"
            shutil.copy2(src, sdir / f"original{ext}")
    oriented_legacy = session_path / "original_oriented.jpg"
    if oriented_legacy.is_file() and not has_oriented_source(pid):
        shutil.copy2(oriented_legacy, sdir / SOURCE_ORIENTED)

    # Artifacts
    if adir.is_dir():
        for item in session_path.iterdir():
            if not item.is_file():
                continue
            if item.name.startswith("input."):
                continue
            dest = adir / item.name
            if not dest.exists():
                shutil.copy2(item, dest)

    if not manifest_path(pid).is_file():
        upsert_manifest_from_client_payload(pid, {"sessionId": pid, "name": pid[:8], "createdAt": int(session_path.stat().st_mtime * 1000)})

    snap = load_session_snapshot(pid)
    if not snap and (adir / "preview.jpg").is_file():
        # minimal rebuild not attempted; user can regenerate
        pass

    logger.info("Migrated legacy session -> project bundle: %s", pid)


def migrate_storage_on_startup() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    for p in list(PROJECTS_DIR.iterdir()):
        if p.is_file() and p.suffix == ".json":
            _migrate_flat_json_project(p)
    if LEGACY_SESSIONS_DIR.is_dir():
        for session_path in list(LEGACY_SESSIONS_DIR.iterdir()):
            if session_path.is_dir():
                _migrate_legacy_session(session_path)


def new_project_id() -> str:
    return str(uuid.uuid4())
