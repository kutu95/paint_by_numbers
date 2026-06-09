from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.responses import FileResponse, Response, StreamingResponse, JSONResponse
import base64
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, Any
import asyncio
import uuid
import shutil
from datetime import datetime, timedelta
import os
import traceback
import logging
import hashlib
import tempfile
import urllib.request
import urllib.error
import threading
import time
import cv2
import numpy as np
from sklearn.cluster import KMeans
from image_processor import (
    process_image,
    regenerate_pure_mask_from_labels,
    make_quantization_preview_jpeg,
    make_quantization_preview,
)
import project_store
from paint_manager import (
    load_library, save_library, slugify, atomic_write,
    sample_color_from_image, sample_color_from_region, normalize_calibration_samples,
    get_hex_from_calibration, calibration_file_for, migrate_global_calibrations_to_group_scope,
    rgb_to_lab, lab_to_rgb, delta_e_lab, interpolate_lab_from_calibration, CALIBRATION_DIR, PAINT_DIR,
    list_library_groups, get_library_info,
    build_library_export, import_library_data,
    generate_recipes_for_palette, load_recipe_cache, save_recipe_cache, invalidate_recipe_cache,
    load_feedback_bias, save_feedback_bias, _bias_key,
    get_substrate_compensation, _apply_substrate_l_compensation,
    predict_mix_lab_for_paint_ratios,
)
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

RECIPE_PROGRESS_LOCK = threading.Lock()
RECIPE_PROGRESS: dict[str, dict] = {}

# Async recipe jobs: avoid long-running HTTP so proxies (e.g. Cloudflare) don't timeout
RECIPE_JOBS_LOCK = threading.Lock()
RECIPE_JOBS: dict[str, dict] = {}
RECIPE_JOB_TTL_SECONDS = 3600

GAMUT_CACHE_DIR = PAINT_DIR / "gamut_cache"
GAMUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PALETTE_OPTIMIZATION_CACHE_DIR = PAINT_DIR / "palette_optimization_cache"
PALETTE_OPTIMIZATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SPOT_TEST_DIR = PAINT_DIR / "spot_test"
SPOT_TEST_DIR.mkdir(parents=True, exist_ok=True)

# CORS middleware - allow origins from environment or default to localhost
# Always include localhost so local dev works even when CORS_ORIGINS is set for production
default_origins = "http://localhost:3000,http://localhost:3001,http://localhost:3002,http://localhost:3003"
cors_origins_str = os.getenv("CORS_ORIGINS", default_origins)
allowed_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]
for origin in default_origins.split(","):
    if origin.strip() not in allowed_origins:
        allowed_origins.append(origin.strip())
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers for CORS
)

CALIBRATION_MIGRATION_MARKER = PAINT_DIR / ".calibration_scope_migration_v1.done"


def _cors_file_response(file_path: Path, request: Request) -> FileResponse:
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    origin = request.headers.get("origin")
    if origin:
        if origin in allowed_origins or "layerpainter.margies.app" in origin or "margies.app" in origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def _parse_form_bool(value: str) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _validate_processing_params(
    n_colors: int,
    overpaint_mm: float,
    order_mode: str,
    max_side: int,
    saturation_boost: float,
    detail_level: float,
    mask_dilation_px: int,
) -> None:
    if n_colors < 2 or n_colors > 100:
        raise HTTPException(status_code=400, detail="n_colors must be between 2 and 100")
    if overpaint_mm < 0 or overpaint_mm > 50:
        raise HTTPException(status_code=400, detail="overpaint_mm must be between 0 and 50")
    if order_mode not in ["auto", "largest", "smallest", "manual", "lightest"]:
        raise HTTPException(status_code=400, detail="order_mode must be auto, largest, smallest, manual, or lightest")
    if max_side < 100 or max_side > 5000:
        raise HTTPException(status_code=400, detail="max_side must be between 100 and 5000")
    if saturation_boost < 0.5 or saturation_boost > 5.0:
        raise HTTPException(status_code=400, detail="saturation_boost must be between 0.5 and 5.0")
    if detail_level < 0.0 or detail_level > 1.0:
        raise HTTPException(status_code=400, detail="detail_level must be between 0.0 and 1.0")
    if mask_dilation_px < 0 or mask_dilation_px > 20:
        raise HTTPException(status_code=400, detail="mask_dilation_px must be between 0 and 20")


def _validate_easy_painting_params(easy_simplify: float) -> float:
    s = float(easy_simplify)
    if s < 0.0 or s > 1.0:
        raise HTTPException(status_code=400, detail="easy_simplify must be between 0.0 and 1.0")
    return s


def _validate_skin_tone_strength(skin_tone_strength: float) -> float:
    s = float(skin_tone_strength)
    if s < 0.0 or s > 1.0:
        raise HTTPException(status_code=400, detail="skin_tone_strength must be between 0.0 and 1.0")
    return s


def _validate_priority_region_strength(strength: float) -> float:
    s = float(strength)
    if s < 0.0 or s > 1.0:
        raise HTTPException(status_code=400, detail="priority_region_strength must be between 0.0 and 1.0")
    return s


def _parse_must_include_colors(raw: str) -> list[str]:
    from image_processor import _normalize_must_include_hex_list

    if not raw or not str(raw).strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="must_include_colors must be a JSON array of hex strings"
        )
    if not isinstance(data, list):
        raise HTTPException(
            status_code=400, detail="must_include_colors must be a JSON array of hex strings"
        )
    return _normalize_must_include_hex_list([str(x) for x in data])


def _session_must_include_colors(project_id: str) -> list[str]:
    """Recover must-include picks from generated palette when manifest was not persisted."""
    from image_processor import _normalize_must_include_hex_list

    snap = project_store.load_session_snapshot(project_id)
    if not snap:
        return []
    palette = snap.get("palette")
    if not isinstance(palette, list):
        return []
    hexes: list[str] = []
    for entry in palette:
        if isinstance(entry, dict) and entry.get("must_include"):
            h = entry.get("hex")
            if h:
                hexes.append(str(h))
    return _normalize_must_include_hex_list(hexes)


def _manifest_must_include_colors(project_id: str) -> list[str]:
    from image_processor import _normalize_must_include_hex_list

    manifest = project_store.load_manifest(project_id)
    if not manifest:
        return _session_must_include_colors(project_id)
    raw = manifest.get("mustIncludeColors")
    if raw is None:
        raw = (manifest.get("processing") or {}).get("mustIncludeColors")
    colors: list[str] = []
    if isinstance(raw, list):
        colors = _normalize_must_include_hex_list([str(x) for x in raw])
    if colors:
        return colors
    return _session_must_include_colors(project_id)


def _manifest_favor_skin_tones(project_id: str) -> bool | None:
    manifest = project_store.load_manifest(project_id)
    if not manifest:
        return None
    if manifest.get("favorSkinTones") is not None:
        return bool(manifest.get("favorSkinTones"))
    processing = manifest.get("processing") if isinstance(manifest.get("processing"), dict) else {}
    if processing.get("favorSkinTones") is not None:
        return bool(processing.get("favorSkinTones"))
    return None


def _manifest_skin_tone_strength(project_id: str) -> float | None:
    manifest = project_store.load_manifest(project_id)
    if not manifest:
        return None
    if manifest.get("skinToneStrength") is not None:
        return float(manifest.get("skinToneStrength"))
    processing = manifest.get("processing") if isinstance(manifest.get("processing"), dict) else {}
    if processing.get("skinToneStrength") is not None:
        return float(processing.get("skinToneStrength"))
    return None


async def _handle_priority_region_upload(
    project_id: str,
    mask_file: Optional[UploadFile],
    clear_mask: bool,
) -> Optional[Path]:
    """Persist or remove user-drawn priority region mask on the project bundle."""
    if clear_mask:
        project_store.delete_priority_region(project_id)
        return None
    if mask_file is None or not mask_file.filename:
        existing = project_store.priority_region_path(project_id)
        return existing if existing.is_file() else None
    content = await mask_file.read()
    if not content:
        project_store.delete_priority_region(project_id)
        return None
    return project_store.save_priority_region(project_id, content)


def _parse_figure_detail_params(
    easy_face_detail: str,
    detail_eyes: str = "true",
    detail_face: str = "true",
    detail_body_outline: str = "true",
) -> tuple[bool, bool, bool, bool]:
    """Master switch + per-region auto-detection toggles."""
    master = _parse_form_bool(easy_face_detail)
    if not master:
        return False, False, False, False
    eyes = _parse_form_bool(detail_eyes)
    face = _parse_form_bool(detail_face)
    outline = _parse_form_bool(detail_body_outline)
    if not (eyes or face or outline):
        eyes = face = outline = True
    return master, eyes, face, outline


def _run_project_processing(
    project_id: str,
    image_path: Path,
    n_colors: int,
    overpaint_mm: float,
    order_mode: str,
    max_side: int,
    saturation_boost: float,
    detail_level: float,
    mask_dilation_px: int,
    canvas_width_cm: float,
    canvas_height_cm: float,
    manifest_updates: Optional[dict] = None,
    easy_painting: bool = False,
    easy_simplify: float = 0.65,
    easy_face_detail: bool = False,
    detail_eyes: bool = True,
    detail_face: bool = True,
    detail_body_outline: bool = True,
    style_preset: str = "none",
    favor_skin_tones: bool = True,
    skin_tone_strength: float = 0.65,
    priority_region_strength: float = 0.7,
    must_include_hex: Optional[list[str]] = None,
) -> dict:
    from image_processor import normalize_style_preset, resolve_image_style

    preset = normalize_style_preset(style_preset or "none")
    skin_tone_strength = _validate_skin_tone_strength(skin_tone_strength)
    style = resolve_image_style(
        preset,
        easy_painting=easy_painting if preset == "none" else None,
        easy_simplify=easy_simplify,
        easy_face_detail=easy_face_detail,
        detail_eyes=detail_eyes,
        detail_face=detail_face,
        detail_body_outline=detail_body_outline,
    )
    easy_painting = style.easy_painting
    easy_simplify = _validate_easy_painting_params(style.easy_simplify) if style.easy_painting else 0.65
    project_store.ensure_project_dirs(project_id)
    if manifest_updates:
        project_store.upsert_manifest_from_client_payload(project_id, manifest_updates)
    project_store.clear_artifacts(project_id)
    # Reset UI state so old done-layer / layer indices do not apply to a new layer stack.
    project_store.save_state(
        project_id,
        {
            "currentLayer": 0,
            "doneLayers": [],
            "projectionScale": 1.0,
        },
    )
    root = project_store.project_root(project_id)
    result = process_image(
        str(image_path),
        root,
        project_id,
        n_colors,
        overpaint_mm,
        order_mode,
        max_side,
        saturation_boost,
        detail_level,
        mask_dilation_px,
        easy_painting=easy_painting,
        easy_simplify=easy_simplify,
        easy_face_detail=easy_face_detail,
        detail_eyes=detail_eyes,
        detail_face=detail_face,
        detail_body_outline=detail_body_outline,
        style_preset=preset,
        favor_skin_tones=favor_skin_tones,
        skin_tone_strength=skin_tone_strength,
        priority_region_path=(
            str(project_store.priority_region_path(project_id))
            if project_store.has_priority_region(project_id)
            else None
        ),
        priority_region_strength=priority_region_strength,
        must_include_hex=must_include_hex,
    )
    result["session_id"] = project_id
    result["canvas_width_cm"] = max(0.0, float(canvas_width_cm))
    result["canvas_height_cm"] = max(0.0, float(canvas_height_cm))
    project_store.save_session_snapshot(project_id, result)
    project_store.apply_processing_to_manifest(
        project_id,
        {
            "nColors": n_colors,
            "overpaintMm": overpaint_mm,
            "orderMode": order_mode,
            "maxSide": max_side,
            "saturationBoost": saturation_boost,
            "detailLevel": detail_level,
            "maskDilationPx": mask_dilation_px,
            "easyPainting": easy_painting,
            "easySimplify": easy_simplify,
            "easyFaceDetail": easy_face_detail,
            "stylePreset": preset,
            "detailEyes": detail_eyes,
            "detailFace": detail_face,
            "detailBodyOutline": detail_body_outline,
            "favorSkinTones": favor_skin_tones,
            "skinToneStrength": skin_tone_strength,
            "priorityRegionStrength": priority_region_strength,
            "mustIncludeColors": must_include_hex or [],
        },
    )
    manifest = project_store.load_manifest(project_id)
    if manifest:
        updated = manifest.get("updatedAt") or manifest.get("createdAt")
        if updated:
            result["artifacts_version"] = int(updated)
    return result


def _upsert_group_calibration(group: str, paint_id: str, calibration: dict) -> None:
    lib = load_library(group)
    paints = lib.get("paints", [])
    has_paint = any(isinstance(p, dict) and p.get("id") == paint_id for p in paints)
    if not has_paint:
        return
    cal_map = lib.get("calibration_data")
    if not isinstance(cal_map, dict):
        cal_map = {}
    cal_map[paint_id] = calibration
    lib["calibration_data"] = cal_map
    save_library(lib, group)


def run_calibration_scope_migration_once() -> None:
    """Run one-time calibration migration on server startup."""
    if CALIBRATION_MIGRATION_MARKER.exists():
        return
    try:
        result = migrate_global_calibrations_to_group_scope(delete_legacy=True)
        marker_payload = {
            "migration": "calibration_scope_v1",
            "applied_at": datetime.now().isoformat(),
            "result": result,
        }
        atomic_write(CALIBRATION_MIGRATION_MARKER, marker_payload)
        logger.info("Calibration scope migration completed on startup: %s", result)
    except Exception as e:
        logger.exception("Calibration scope migration failed on startup: %s", e)


@app.on_event("startup")
async def startup_event():
    run_calibration_scope_migration_once()
    project_store.migrate_storage_on_startup()


def _project_info_payload(project_id: str) -> dict:
    if not project_store.project_root(project_id).is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    has_source = project_store.find_source_input(project_id) is not None
    has_oriented = project_store.has_oriented_source(project_id)
    original_url = project_store.source_url(project_id, project_store.SOURCE_ORIENTED) if has_oriented else None
    pr_url = None
    if project_store.has_priority_region(project_id):
        pr_url = project_store.source_url(project_id, project_store.SOURCE_PRIORITY_REGION)
    return {
        "session_id": project_id,
        "project_id": project_id,
        "original_url": original_url,
        "priority_region_url": pr_url,
        "has_stored_image": has_source,
        "has_artifacts": project_store.load_session_snapshot(project_id) is not None,
        "has_priority_region": project_store.has_priority_region(project_id),
        "must_include_colors": _manifest_must_include_colors(project_id),
        "favor_skin_tones": _manifest_favor_skin_tones(project_id),
        "skin_tone_strength": _manifest_skin_tone_strength(project_id),
    }


@app.post("/api/sessions")
async def create_session(
    image: UploadFile = File(...),
    n_colors: int = Form(16),
    overpaint_mm: float = Form(5.0),
    order_mode: str = Form("largest"),
    max_side: int = Form(1920),
    saturation_boost: float = Form(1.0),
    detail_level: float = Form(0.5),
    mask_dilation_px: int = Form(0),
    easy_painting: str = Form("false"),
    easy_simplify: float = Form(0.65),
    easy_face_detail: str = Form("false"),
    detail_eyes: str = Form("true"),
    detail_face: str = Form("true"),
    detail_body_outline: str = Form("true"),
    style_preset: str = Form(""),
    favor_skin_tones: str = Form("true"),
    skin_tone_strength: float = Form(0.65),
    priority_region_mask: Optional[UploadFile] = File(None),
    clear_priority_region: str = Form("false"),
    priority_region_strength: float = Form(0.7),
    must_include_colors: str = Form(""),
    canvas_width_cm: float = Form(0),
    canvas_height_cm: float = Form(0),
    project_id: str = Form(""),
    name: str = Form(""),
    library_group: str = Form("default"),
):
    """Create a project bundle, store source image, and generate layers."""
    easy = _parse_form_bool(easy_painting)
    favor_skin = _parse_form_bool(favor_skin_tones)
    skin_strength = _validate_skin_tone_strength(skin_tone_strength)
    pr_strength = _validate_priority_region_strength(priority_region_strength)
    preserve, d_eyes, d_face, d_outline = _parse_figure_detail_params(
        easy_face_detail, detail_eyes, detail_face, detail_body_outline
    )
    from image_processor import normalize_style_preset, resolve_image_style

    preset = normalize_style_preset((style_preset or "").strip() or "none")
    style = resolve_image_style(
        preset,
        easy_painting=easy if preset == "none" else None,
        easy_simplify=easy_simplify,
        easy_face_detail=preserve,
        detail_eyes=d_eyes,
        detail_face=d_face,
        detail_body_outline=d_outline,
    )
    easy = style.easy_painting
    easy_simp = _validate_easy_painting_params(style.easy_simplify) if easy else 0.65
    _validate_processing_params(
        n_colors, overpaint_mm, order_mode, max_side, saturation_boost, detail_level, mask_dilation_px
    )
    pid = (project_id or "").strip() or project_store.new_project_id()
    if project_store.project_exists(pid) and (project_id or "").strip():
        project_store.clear_artifacts(pid)
    else:
        project_store.ensure_project_dirs(pid)
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    image_path = project_store.save_uploaded_source(pid, content, image.filename or "image.jpg")
    await _handle_priority_region_upload(
        pid, priority_region_mask, _parse_form_bool(clear_priority_region)
    )
    manifest_updates = {
        "sessionId": pid,
        "name": (name or "").strip() or "Untitled",
        "imageFileName": image.filename or "image",
        "libraryGroup": library_group or "default",
        "canvasWidthCm": canvas_width_cm,
        "canvasHeightCm": canvas_height_cm,
        "saturationBoost": saturation_boost,
        "detailLevel": detail_level,
        "easyPainting": easy,
        "easySimplify": easy_simp,
        "easyFaceDetail": preserve,
        "stylePreset": preset,
        "detailEyes": d_eyes,
        "detailFace": d_face,
        "detailBodyOutline": d_outline,
        "favorSkinTones": favor_skin,
        "skinToneStrength": skin_strength,
        "priorityRegionStrength": pr_strength,
        "mustIncludeColors": _parse_must_include_colors(must_include_colors),
        "nColors": n_colors,
        "overpaintMm": overpaint_mm,
        "orderMode": order_mode,
        "maxSide": max_side,
        "createdAt": int(datetime.now().timestamp() * 1000),
    }
    must_include = _parse_must_include_colors(must_include_colors)
    if must_include and len(must_include) >= n_colors:
        raise HTTPException(
            status_code=400,
            detail=f"Too many must-include colours ({len(must_include)}) for palette size {n_colors}",
        )
    try:
        return _run_project_processing(
            pid,
            image_path,
            n_colors,
            overpaint_mm,
            order_mode,
            max_side,
            saturation_boost,
            detail_level,
            mask_dilation_px,
            canvas_width_cm,
            canvas_height_cm,
            manifest_updates,
            easy_painting=easy,
            easy_simplify=easy_simp,
            easy_face_detail=preserve,
            detail_eyes=d_eyes,
            detail_face=d_face,
            detail_body_outline=d_outline,
            style_preset=preset,
            favor_skin_tones=favor_skin,
            skin_tone_strength=skin_strength,
            priority_region_strength=pr_strength,
            must_include_hex=must_include or None,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/preview/quantize")
async def preview_quantize(
    image: Optional[UploadFile] = File(None),
    project_id: str = Form(""),
    n_colors: int = Form(16),
    max_side: int = Form(1024),
    saturation_boost: float = Form(1.0),
    easy_painting: str = Form("false"),
    easy_simplify: float = Form(0.65),
    easy_face_detail: str = Form("false"),
    detail_eyes: str = Form("true"),
    detail_face: str = Form("true"),
    detail_body_outline: str = Form("true"),
    style_preset: str = Form("none"),
    favor_skin_tones: str = Form("true"),
    skin_tone_strength: float = Form(0.65),
    priority_region_mask: Optional[UploadFile] = File(None),
    clear_priority_region: str = Form("false"),
    priority_region_strength: float = Form(0.7),
    detail_level: float = Form(0.5),
    include_palette: str = Form("false"),
    must_include_colors: str = Form(""),
):
    """JPEG preview of quantized colours (fast; no layer masks)."""
    if n_colors < 2 or n_colors > 100:
        raise HTTPException(status_code=400, detail="n_colors must be between 2 and 100")
    if max_side < 200 or max_side > 2400:
        raise HTTPException(status_code=400, detail="max_side must be between 200 and 2400")
    if saturation_boost < 0.5 or saturation_boost > 5.0:
        raise HTTPException(status_code=400, detail="saturation_boost must be between 0.5 and 5.0")
    if detail_level < 0.0 or detail_level > 1.0:
        raise HTTPException(status_code=400, detail="detail_level must be between 0.0 and 1.0")

    from image_processor import normalize_style_preset, resolve_image_style

    preserve, d_eyes, d_face, d_outline = _parse_figure_detail_params(
        easy_face_detail, detail_eyes, detail_face, detail_body_outline
    )
    legacy_easy = _parse_form_bool(easy_painting)
    favor_skin = _parse_form_bool(favor_skin_tones)
    skin_strength = _validate_skin_tone_strength(skin_tone_strength)
    pr_strength = _validate_priority_region_strength(priority_region_strength)
    mask_detail = float(detail_level)
    preset = normalize_style_preset((style_preset or "").strip() or "none")
    style = resolve_image_style(
        preset,
        easy_painting=legacy_easy if preset == "none" else None,
        easy_simplify=easy_simplify,
        easy_face_detail=preserve,
        detail_eyes=d_eyes,
        detail_face=d_face,
        detail_body_outline=d_outline,
    )
    easy_simp = _validate_easy_painting_params(style.easy_simplify) if style.easy_painting else 0.0
    from image_processor import (
        load_priority_region_mask,
        load_priority_region_mask_bytes,
        load_rgb_image_normalized,
    )

    tmp_path: Optional[Path] = None
    preview_pid = ""
    try:
        if image is not None and image.filename:
            content = await image.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded image is empty")
            fd, tmp_name = tempfile.mkstemp(suffix=".jpg", prefix="pbn_preview_")
            os.close(fd)
            tmp_path = Path(tmp_name)
            tmp_path.write_bytes(content)
            image_path = tmp_path
        else:
            preview_pid = (project_id or "").strip()
            if not preview_pid:
                raise HTTPException(status_code=400, detail="Provide an image or project_id")
            image_path = project_store.find_source_input(preview_pid)
            if not image_path:
                raise HTTPException(status_code=404, detail="No stored source image for this project")

        if preview_pid:
            await _handle_priority_region_upload(
                preview_pid,
                priority_region_mask,
                _parse_form_bool(clear_priority_region),
            )

        quantize_side = int(max_side)
        priority_mask = None
        if priority_region_mask is not None and priority_region_mask.filename and not preview_pid:
            mask_bytes = await priority_region_mask.read()
            if mask_bytes:
                norm = load_rgb_image_normalized(str(image_path), quantize_side)
                priority_mask = load_priority_region_mask_bytes(
                    mask_bytes, norm.shape[0], norm.shape[1]
                )
        if priority_mask is None and preview_pid and project_store.has_priority_region(preview_pid):
            norm = load_rgb_image_normalized(str(image_path), quantize_side)
            priority_mask = load_priority_region_mask(
                str(project_store.priority_region_path(preview_pid)),
                norm.shape[0],
                norm.shape[1],
            )

        must_include = _parse_must_include_colors(must_include_colors)
        if not must_include and preview_pid:
            must_include = _manifest_must_include_colors(preview_pid)
        if must_include and len(must_include) >= n_colors:
            raise HTTPException(
                status_code=400,
                detail=f"Too many must-include colours ({len(must_include)}) for palette size {n_colors}",
            )

        preview_kwargs = dict(
            image_path=str(image_path),
            n_colors=n_colors,
            max_side=max_side,
            saturation_boost=saturation_boost,
            easy_painting=style.easy_painting,
            easy_simplify=easy_simp,
            easy_face_detail=preserve,
            detail_eyes=d_eyes,
            detail_face=d_face,
            detail_body_outline=d_outline,
            style_preset=preset,
            favor_skin_tones=favor_skin,
            skin_tone_strength=skin_strength,
            priority_region_mask=priority_mask,
            priority_region_strength=pr_strength,
            mask_detail_level=mask_detail,
            must_include_hex=must_include or None,
        )

        if _parse_form_bool(include_palette):
            jpeg, palette = make_quantization_preview(**preview_kwargs)
            return JSONResponse(
                {
                    "jpeg_base64": base64.standard_b64encode(jpeg).decode("ascii"),
                    "palette": palette,
                    "pipeline_version": __import__("image_processor").PIPELINE_VERSION,
                }
            )

        jpeg, _palette = make_quantization_preview(**preview_kwargs)
        return Response(content=jpeg, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview quantize failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@app.get("/api/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """Legacy alias for project info."""
    return _project_info_payload(session_id)


@app.get("/api/projects/{project_id}/info")
async def get_project_info(project_id: str):
    return _project_info_payload(project_id)


@app.get("/api/projects/{project_id}/session")
async def get_project_session(project_id: str):
    """Full layer pipeline data for a project (single source of truth on disk)."""
    session = project_store.build_session_response(project_id)
    if not session:
        raise HTTPException(status_code=404, detail="No generated layers for this project yet")
    return session


@app.get("/api/projects/{project_id}/state")
async def get_project_state(project_id: str):
    if not project_store.project_root(project_id).is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    return project_store.load_state(project_id)


@app.put("/api/projects/{project_id}/state")
async def put_project_state(project_id: str, request: Request):
    if not project_store.project_root(project_id).is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Payload must be an object")
    project_store.save_state(project_id, body)
    return body


@app.post("/api/sessions/{session_id}/reprocess")
async def reprocess_session(
    session_id: str,
    n_colors: int = Form(16),
    overpaint_mm: float = Form(5.0),
    order_mode: str = Form("largest"),
    max_side: int = Form(1920),
    saturation_boost: float = Form(1.0),
    detail_level: float = Form(0.5),
    mask_dilation_px: int = Form(0),
    easy_painting: str = Form("false"),
    easy_simplify: float = Form(0.65),
    easy_face_detail: str = Form("false"),
    detail_eyes: str = Form("true"),
    detail_face: str = Form("true"),
    detail_body_outline: str = Form("true"),
    style_preset: str = Form(""),
    favor_skin_tones: str = Form("true"),
    skin_tone_strength: float = Form(0.65),
    priority_region_mask: Optional[UploadFile] = File(None),
    clear_priority_region: str = Form("false"),
    priority_region_strength: float = Form(0.7),
    must_include_colors: str = Form(""),
    canvas_width_cm: float = Form(0),
    canvas_height_cm: float = Form(0),
    name: str = Form(""),
    library_group: str = Form(""),
):
    """Regenerate layers from stored project source image."""
    easy = _parse_form_bool(easy_painting)
    favor_skin = _parse_form_bool(favor_skin_tones)
    skin_strength = _validate_skin_tone_strength(skin_tone_strength)
    pr_strength = _validate_priority_region_strength(priority_region_strength)
    preserve, d_eyes, d_face, d_outline = _parse_figure_detail_params(
        easy_face_detail, detail_eyes, detail_face, detail_body_outline
    )
    await _handle_priority_region_upload(
        session_id, priority_region_mask, _parse_form_bool(clear_priority_region)
    )
    must_include = _parse_must_include_colors(must_include_colors)
    if not must_include:
        must_include = _manifest_must_include_colors(session_id)
    if must_include and len(must_include) >= n_colors:
        raise HTTPException(
            status_code=400,
            detail=f"Too many must-include colours ({len(must_include)}) for palette size {n_colors}",
        )
    return await _reprocess_project(
        session_id,
        n_colors,
        overpaint_mm,
        order_mode,
        max_side,
        saturation_boost,
        detail_level,
        mask_dilation_px,
        easy,
        _validate_easy_painting_params(easy_simplify) if easy else 0.65,
        preserve,
        d_eyes,
        d_face,
        d_outline,
        canvas_width_cm,
        canvas_height_cm,
        name,
        library_group,
        style_preset=style_preset,
        favor_skin_tones=favor_skin,
        skin_tone_strength=skin_strength,
        priority_region_strength=pr_strength,
        must_include_hex=must_include or None,
    )


@app.post("/api/projects/{project_id}/generate")
async def generate_project(
    project_id: str,
    n_colors: int = Form(16),
    overpaint_mm: float = Form(5.0),
    order_mode: str = Form("largest"),
    max_side: int = Form(1920),
    saturation_boost: float = Form(1.0),
    detail_level: float = Form(0.5),
    mask_dilation_px: int = Form(0),
    easy_painting: str = Form("false"),
    easy_simplify: float = Form(0.65),
    easy_face_detail: str = Form("false"),
    detail_eyes: str = Form("true"),
    detail_face: str = Form("true"),
    detail_body_outline: str = Form("true"),
    style_preset: str = Form(""),
    favor_skin_tones: str = Form("true"),
    skin_tone_strength: float = Form(0.65),
    priority_region_mask: Optional[UploadFile] = File(None),
    clear_priority_region: str = Form("false"),
    priority_region_strength: float = Form(0.7),
    must_include_colors: str = Form(""),
    canvas_width_cm: float = Form(0),
    canvas_height_cm: float = Form(0),
    name: str = Form(""),
    library_group: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    """Regenerate layers; optional image upload replaces project source."""
    if image is not None:
        content = await image.read()
        if content:
            project_store.save_uploaded_source(project_id, content, image.filename or "image.jpg")
    await _handle_priority_region_upload(
        project_id, priority_region_mask, _parse_form_bool(clear_priority_region)
    )
    easy = _parse_form_bool(easy_painting)
    favor_skin = _parse_form_bool(favor_skin_tones)
    skin_strength = _validate_skin_tone_strength(skin_tone_strength)
    pr_strength = _validate_priority_region_strength(priority_region_strength)
    preserve, d_eyes, d_face, d_outline = _parse_figure_detail_params(
        easy_face_detail, detail_eyes, detail_face, detail_body_outline
    )
    must_include = _parse_must_include_colors(must_include_colors)
    if not must_include:
        must_include = _manifest_must_include_colors(project_id)
    if must_include and len(must_include) >= n_colors:
        raise HTTPException(
            status_code=400,
            detail=f"Too many must-include colours ({len(must_include)}) for palette size {n_colors}",
        )
    return await _reprocess_project(
        project_id,
        n_colors,
        overpaint_mm,
        order_mode,
        max_side,
        saturation_boost,
        detail_level,
        mask_dilation_px,
        easy,
        _validate_easy_painting_params(easy_simplify) if easy else 0.65,
        preserve,
        d_eyes,
        d_face,
        d_outline,
        canvas_width_cm,
        canvas_height_cm,
        name,
        library_group,
        style_preset=style_preset,
        favor_skin_tones=favor_skin,
        skin_tone_strength=skin_strength,
        priority_region_strength=pr_strength,
        must_include_hex=must_include or None,
    )


async def _reprocess_project(
    project_id: str,
    n_colors: int,
    overpaint_mm: float,
    order_mode: str,
    max_side: int,
    saturation_boost: float,
    detail_level: float,
    mask_dilation_px: int,
    easy_painting: bool,
    easy_simplify: float,
    easy_face_detail: bool,
    detail_eyes: bool,
    detail_face: bool,
    detail_body_outline: bool,
    canvas_width_cm: float,
    canvas_height_cm: float,
    name: str,
    library_group: str,
    style_preset: str = "",
    favor_skin_tones: bool = True,
    skin_tone_strength: float = 0.65,
    priority_region_strength: float = 0.7,
    must_include_hex: Optional[list[str]] = None,
):
    from image_processor import normalize_style_preset, resolve_image_style

    skin_tone_strength = _validate_skin_tone_strength(skin_tone_strength)
    priority_region_strength = _validate_priority_region_strength(priority_region_strength)
    if must_include_hex and len(must_include_hex) >= n_colors:
        raise HTTPException(
            status_code=400,
            detail=f"Too many must-include colours ({len(must_include_hex)}) for palette size {n_colors}",
        )
    preset = normalize_style_preset((style_preset or "").strip() or "none")
    style = resolve_image_style(
        preset,
        easy_painting=easy_painting if preset == "none" else None,
        easy_simplify=easy_simplify,
        easy_face_detail=easy_face_detail,
        detail_eyes=detail_eyes,
        detail_face=detail_face,
        detail_body_outline=detail_body_outline,
    )
    easy_painting = style.easy_painting
    easy_simplify = _validate_easy_painting_params(style.easy_simplify) if easy_painting else 0.65

    if not project_store.project_root(project_id).is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    image_path = project_store.find_source_input(project_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="No stored source image for this project. Upload a new image.")
    _validate_processing_params(
        n_colors, overpaint_mm, order_mode, max_side, saturation_boost, detail_level, mask_dilation_px
    )
    updates: dict = {
        "canvasWidthCm": canvas_width_cm,
        "canvasHeightCm": canvas_height_cm,
        "saturationBoost": saturation_boost,
        "detailLevel": detail_level,
        "easyPainting": easy_painting,
        "easySimplify": easy_simplify,
        "easyFaceDetail": easy_face_detail,
        "stylePreset": preset,
        "detailEyes": detail_eyes,
        "detailFace": detail_face,
        "detailBodyOutline": detail_body_outline,
        "favorSkinTones": favor_skin_tones,
        "skinToneStrength": skin_tone_strength,
        "priorityRegionStrength": priority_region_strength,
        "mustIncludeColors": must_include_hex or [],
        "nColors": n_colors,
        "overpaintMm": overpaint_mm,
        "orderMode": order_mode,
        "maxSide": max_side,
    }
    if name.strip():
        updates["name"] = name.strip()
    if library_group.strip():
        updates["libraryGroup"] = library_group.strip()
    try:
        return _run_project_processing(
            project_id,
            image_path,
            n_colors,
            overpaint_mm,
            order_mode,
            max_side,
            saturation_boost,
            detail_level,
            mask_dilation_px,
            canvas_width_cm,
            canvas_height_cm,
            updates,
            easy_painting=easy_painting,
            easy_simplify=easy_simplify,
            easy_face_detail=easy_face_detail,
            detail_eyes=detail_eyes,
            detail_face=detail_face,
            detail_body_outline=detail_body_outline,
            style_preset=preset,
            favor_skin_tones=favor_skin_tones,
            skin_tone_strength=skin_tone_strength,
            priority_region_strength=priority_region_strength,
            must_include_hex=must_include_hex,
        )
    except Exception as e:
        logger.error(f"Reprocess failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")


def _serve_project_artifact(project_id: str, filename: str, request: Request) -> FileResponse:
    adir = project_store.artifacts_dir(project_id)
    file_path = project_store.resolve_artifact_file(project_id, filename)
    if not file_path or not file_path.is_file():
        if filename.endswith("_pure_mask.png"):
            try:
                layer_index = int(filename.replace("layer_", "").replace("_pure_mask.png", ""))
            except ValueError:
                layer_index = -1
            if layer_index >= 0 and regenerate_pure_mask_from_labels(adir, layer_index):
                file_path = adir / filename
        if not file_path or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
    elif filename.endswith("_pure_mask.png"):
        try:
            layer_index = int(filename.replace("layer_", "").replace("_pure_mask.png", ""))
        except ValueError:
            layer_index = -1
        if layer_index >= 0:
            regenerate_pure_mask_from_labels(adir, layer_index)
            file_path = adir / filename
    try:
        file_path.resolve().relative_to(project_store.artifacts_dir(project_id).resolve())
    except ValueError:
        try:
            file_path.resolve().relative_to(project_store.source_dir(project_id).resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
    return _cors_file_response(file_path, request)


def _options_cors_response(request: Request) -> Response:
    origin = request.headers.get("origin", "https://layerpainter.margies.app")
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        },
    )


@app.options("/api/projects/{project_id}/artifacts/{filename}")
async def options_project_artifact(project_id: str, filename: str, request: Request):
    return _options_cors_response(request)


@app.options("/api/projects/{project_id}/source/{filename}")
async def options_project_source(project_id: str, filename: str, request: Request):
    return _options_cors_response(request)


@app.options("/api/sessions/{session_id}/{filename}")
async def options_session_file(session_id: str, filename: str, request: Request):
    return _options_cors_response(request)


@app.get("/api/projects/{project_id}/artifacts/{filename}")
async def get_project_artifact(project_id: str, filename: str, request: Request):
    return _serve_project_artifact(project_id, filename, request)


@app.get("/api/projects/{project_id}/source/{filename}")
async def get_project_source(project_id: str, filename: str, request: Request):
    if not project_store.project_root(project_id).is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    file_path = project_store.resolve_source_file(project_id, filename)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
    return _cors_file_response(file_path, request)


@app.get("/api/sessions/{session_id}/{filename}")
async def get_session_file(session_id: str, filename: str, request: Request):
    """Legacy session URLs → project bundle paths."""
    if filename == "original_oriented.jpg":
        return await get_project_source(session_id, project_store.SOURCE_ORIENTED, request)
    if filename.startswith("verify/"):
        verify_name = filename.split("/", 1)[1]
        verify_path = project_store.artifacts_dir(session_id) / "verify" / verify_name
        if verify_path.is_file():
            return _cors_file_response(verify_path, request)
        raise HTTPException(status_code=404, detail="File not found")
    return _serve_project_artifact(session_id, filename, request)


@app.get("/api/sessions/{session_id}/verify/{filename}")
async def get_verification_image_legacy(session_id: str, filename: str, request: Request):
    return await get_session_file(session_id, f"verify/{filename}", request)


# ===== Paint Management Endpoints =====

@app.get("/api/paint/library")
async def get_paint_library(group: str = "default"):
    """Get the paint library for a specific group."""
    return load_library(group)


@app.get("/api/projects")
async def list_projects():
    """List persisted project bundles."""
    return {"projects": project_store.list_projects()}


@app.put("/api/projects/{session_id}")
async def upsert_project(session_id: str, request: Request):
    """Update project manifest (settings/metadata). Creates bundle dir if needed."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Payload must be an object")
    project_store.ensure_project_dirs(session_id)
    manifest = project_store.upsert_manifest_from_client_payload(session_id, body)
    return project_store.manifest_to_list_item(manifest)


@app.delete("/api/projects/{session_id}")
async def delete_project(session_id: str):
    """Delete entire project bundle."""
    deleted = project_store.delete_project(session_id)
    return {"success": True, "deleted": deleted}


@app.get("/api/paint/library/groups")
async def list_paint_library_groups():
    """List all available paint library groups."""
    groups = list_library_groups()
    return {
        "groups": [get_library_info(g) for g in groups],
        "current": "default"  # Default selection
    }


@app.post("/api/paint/library/groups")
async def create_library_group(
    name: str = Form(...)
):
    """Create a new paint library group."""
    group_id = slugify(name)
    
    # Check if group already exists
    existing_groups = list_library_groups()
    if group_id in existing_groups:
        raise HTTPException(status_code=400, detail=f"Library group '{group_id}' already exists")
    
    # Create empty library for the group
    new_library = {
        "version": 1,
        "paints": [],
        "group": group_id,
        "name": name,
        "coverage_mg_per_cm2": None,
        "calibration_data": {},
        "recipes": {},
    }
    save_library(new_library, group_id)
    
    return get_library_info(group_id)


@app.put("/api/paint/library/groups/{group_id}/settings")
async def update_library_settings(
    group_id: str,
    coverage_mg_per_cm2: Optional[float] = Form(None),
):
    """Update library-level settings (e.g. coverage for the whole library)."""
    existing_groups = list_library_groups()
    if group_id not in existing_groups:
        raise HTTPException(status_code=404, detail=f"Library group '{group_id}' not found")
    library = load_library(group_id)
    if coverage_mg_per_cm2 not in (None, ""):
        try:
            val = float(coverage_mg_per_cm2)
            library["coverage_mg_per_cm2"] = val if val > 0 else None
        except (TypeError, ValueError):
            library["coverage_mg_per_cm2"] = None
    else:
        library["coverage_mg_per_cm2"] = None
    save_library(library, group_id)
    return get_library_info(group_id)


@app.get("/api/paint/library/groups/{group_id}/substrate-compensation")
async def get_substrate_compensation_endpoint(group_id: str):
    """Return paper / substrate black-point compensation params for the given library group."""
    existing_groups = list_library_groups()
    if group_id not in existing_groups:
        raise HTTPException(status_code=404, detail=f"Library group '{group_id}' not found")
    return get_substrate_compensation(group_id)


@app.put("/api/paint/library/groups/{group_id}/substrate-compensation")
async def update_substrate_compensation(
    group_id: str,
    body: dict = Body(...),
):
    """Update paper / substrate black-point compensation params for the given library group.

    Body fields (all optional): enabled (bool), L_paper_min (float), L_break (float), alpha_dark (float).
    Recipe cache is invalidated so the next recipe generation re-uses the new params.
    """
    existing_groups = list_library_groups()
    if group_id not in existing_groups:
        raise HTTPException(status_code=404, detail=f"Library group '{group_id}' not found")
    library = load_library(group_id)
    sc = library.get("substrate_compensation")
    sc = dict(sc) if isinstance(sc, dict) else {}
    if "enabled" in body:
        sc["enabled"] = bool(body.get("enabled"))
    for key in ("L_paper_min", "L_break", "alpha_dark"):
        if key in body and body.get(key) is not None:
            try:
                sc[key] = float(body.get(key))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"{key} must be numeric")
    library["substrate_compensation"] = sc
    save_library(library, group_id)
    invalidate_recipe_cache(group_id)
    return get_substrate_compensation(group_id)


@app.put("/api/paint/library/groups/{group_id}")
async def rename_library_group(
    group_id: str,
    name: str = Form(...)
):
    """Rename a paint library group."""
    # Check if group exists
    existing_groups = list_library_groups()
    if group_id not in existing_groups:
        raise HTTPException(status_code=404, detail=f"Library group '{group_id}' not found")
    
    # Load current library
    library = load_library(group_id)
    
    # Update the name
    library["name"] = name
    
    # Save the updated library
    save_library(library, group_id)
    
    return get_library_info(group_id)


@app.get("/api/paint/library/groups/{group_id}/export")
async def export_library_group(group_id: str):
    """Download a full paint library group as JSON (paints, calibrations, recipes, settings)."""
    existing_groups = list_library_groups()
    if group_id not in existing_groups:
        raise HTTPException(status_code=404, detail=f"Library group '{group_id}' not found")
    payload = build_library_export(group_id)
    filename = f"paint_library_{group_id}.json"
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/paint/library/import")
async def import_library_group(
    file: UploadFile = File(...),
    mode: str = Form("replace"),
    target_group: str = Form(""),
):
    """Import a paint library from JSON. Replace an existing group or create a new one."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        raw = await file.read()
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    import_mode = (mode or "replace").strip().lower()
    if import_mode not in ("replace", "new"):
        raise HTTPException(status_code=400, detail="mode must be 'replace' or 'new'")

    try:
        if import_mode == "new":
            info = import_library_data(payload, create_new=True)
        else:
            group = (target_group or "").strip()
            if not group:
                raise ValueError("target_group is required when replacing a library")
            info = import_library_data(payload, target_group=group, create_new=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"success": True, "library": info}


@app.post("/api/paint/library")
async def add_paint(
    name: str = Form(...),
    hex_approx: str = Form(...),
    notes: str = Form(""),
    group: str = Form("default"),
):
    """Add a new paint to the library."""
    library = load_library(group)
    paint_id = slugify(name)
    
    # Check if ID already exists in this group
    existing = [p for p in library['paints'] if p['id'] == paint_id]
    if existing:
        raise HTTPException(status_code=400, detail=f"Paint with ID '{paint_id}' already exists in this library group")
    
    new_paint = {
        "id": paint_id,
        "name": name,
        "type": "base",
        "hex_approx": hex_approx,
        "notes": notes,
    }
    
    library['paints'].append(new_paint)
    save_library(library, group)
    
    return new_paint


@app.put("/api/paint/library/{paint_id}")
async def update_paint(
    paint_id: str,
    name: str = Form(...),
    hex_approx: str = Form(...),
    notes: str = Form(""),
    group: str = Form("default"),
):
    """Update an existing paint. If calibration exists, recalculate hex_approx from the 100% swatch."""
    library = load_library(group)
    paint = next((p for p in library['paints'] if p['id'] == paint_id), None)
    if not paint:
        raise HTTPException(status_code=404, detail="Paint not found")
    
    paint['name'] = name
    paint['notes'] = notes

    # Recalculate hex from calibration 100% swatch when calibration exists; otherwise use form value
    hex_from_cal = get_hex_from_calibration(paint_id, group)
    if hex_from_cal:
        paint['hex_approx'] = hex_from_cal
        logger.info("Update paint: using hex from calibration for %s: %s", paint_id, hex_from_cal)
    else:
        paint['hex_approx'] = hex_approx

    # Save current group
    save_library(library, group)

    return paint


@app.delete("/api/paint/library/{paint_id}")
async def delete_paint(paint_id: str, group: str = "default"):
    """Delete a paint from the library."""
    library = load_library(group)
    library['paints'] = [p for p in library['paints'] if p['id'] != paint_id]
    cal_map = library.get("calibration_data")
    if isinstance(cal_map, dict):
        cal_map.pop(paint_id, None)
        library["calibration_data"] = cal_map
    save_library(library, group)
    
    # Also delete this library's calibration file if it exists
    cal_file = calibration_file_for(group, paint_id)
    if cal_file.exists():
        cal_file.unlink()
    
    return {"success": True}


# Calibration endpoints
@app.post("/api/paint/calibration/upload")
async def upload_calibration_photo(
    image: UploadFile = File(...),
    paint_id: str = Form(...),
    group: str = Form("default"),
):
    """Upload a calibration photo."""
    # Save to temporary location
    temp_dir = CALIBRATION_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    image_id = str(uuid.uuid4())
    image_path = temp_dir / f"{image_id}.jpg"
    
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    
    return {
        "image_id": image_id,
        "preview_url": f"/api/paint/calibration/temp/{image_id}.jpg"
    }


@app.get("/api/paint/calibration/temp/{image_id}.jpg")
async def get_calibration_temp_image(image_id: str, request: Request):
    """Serve temporary calibration image with CORS headers."""
    image_path = CALIBRATION_DIR / "temp" / f"{image_id}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    response = FileResponse(image_path)
    origin = request.headers.get("origin")
    if origin:
        origin_lower = origin.lower()
        if (origin in allowed_origins or 
            "layerpainter.margies.app" in origin_lower or 
            "margies.app" in origin_lower or
            origin_lower.startswith("https://layerpainter") or
            origin_lower.startswith("http://localhost")):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Expose-Headers"] = "*"
    return response


@app.post("/api/paint/calibration/sample")
async def sample_calibration_colors(
    image_id: str = Form(...),
    paint_id: str = Form(...),
    points: str = Form(None),  # legacy: JSON [{x,y}, ...]
    ratios: str = Form(...),  # JSON string of [ratio, ...]
    regions: str = Form(None),  # JSON [{x1,y1,x2,y2}, ...] - user-drawn rectangles (preferred)
    reference_points: str = Form(None),  # legacy: JSON [white, mid_grey, black] each {x,y}
    reference_regions: str = Form(None),  # JSON [white, mid_grey, black] each {x1,y1,x2,y2}
    group: str = Form("default"),
):
    """Sample colors from calibration photo and save calibration.
    Use regions (and reference_regions) to average over user-selected rectangles; falls back to points if provided.
    """
    image_path = CALIBRATION_DIR / "temp" / f"{image_id}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    ratios_list = json.loads(ratios)
    samples = []
    path = str(image_path)

    if regions:
        regions_list = json.loads(regions)
        if len(regions_list) != len(ratios_list):
            raise HTTPException(status_code=400, detail="Regions and ratios must have same length")
        for reg, ratio in zip(regions_list, ratios_list):
            rgb, lab = sample_color_from_region(path, reg['x1'], reg['y1'], reg['x2'], reg['y2'])
            samples.append({"ratio": ratio, "rgb": rgb, "lab": lab})
    elif points:
        points_list = json.loads(points)
        if len(points_list) != len(ratios_list):
            raise HTTPException(status_code=400, detail="Points and ratios must have same length")
        for point, ratio in zip(points_list, ratios_list):
            rgb, lab = sample_color_from_image(path, point['x'], point['y'])
            samples.append({"ratio": ratio, "rgb": rgb, "lab": lab})
    else:
        raise HTTPException(status_code=400, detail="Provide either regions or points")

    reference = {}
    if reference_regions:
        ref_list = json.loads(reference_regions)
        if len(ref_list) >= 3:
            for key, reg in zip(("reference_white", "reference_mid_grey", "reference_black"), ref_list[:3]):
                rgb, lab = sample_color_from_region(path, reg['x1'], reg['y1'], reg['x2'], reg['y2'])
                reference[key] = {"rgb": rgb, "lab": lab}
            samples = normalize_calibration_samples(samples, reference)
    elif reference_points:
        ref_list = json.loads(reference_points)
        if len(ref_list) >= 3:
            for key, point in zip(("reference_white", "reference_mid_grey", "reference_black"), ref_list[:3]):
                rgb, lab = sample_color_from_image(path, point['x'], point['y'])
                reference[key] = {"rgb": rgb, "lab": lab}
            samples = normalize_calibration_samples(samples, reference)
    
    # Save calibration
    calibration = {
        "paint_id": paint_id,
        "ratios": ratios_list,
        "samples": samples,
        "created_at": datetime.now().isoformat(),
        "notes": ""
    }
    if reference:
        calibration["reference_strip"] = reference
    
    cal_file = calibration_file_for(group, paint_id)
    atomic_write(cal_file, calibration)
    _upsert_group_calibration(group, paint_id, calibration)
    
    # Update the paint's approximate color (hex_approx) from the 100% (1.0) swatch in every library group that has this paint
    sample_100 = next((s for s in samples if s.get("ratio", 0) >= 0.99), None)
    if sample_100 and sample_100.get("rgb") and len(sample_100["rgb"]) >= 3:
        r, g, b = sample_100["rgb"][0], sample_100["rgb"][1], sample_100["rgb"][2]
        hex_from_calibration = "#{:02x}{:02x}{:02x}".format(
            max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
        )
        lib = load_library(group)
        for p in lib.get("paints", []):
            if p.get("id") == paint_id:
                p["hex_approx"] = hex_from_calibration
                cal_map = lib.get("calibration_data")
                if not isinstance(cal_map, dict):
                    cal_map = {}
                cal_map[paint_id] = calibration
                lib["calibration_data"] = cal_map
                save_library(lib, group)
                break
    
    return {
        "samples": samples,
        "reference_strip": reference if reference else None,
        "calibration_saved": True
    }


@app.get("/api/paint/calibration/{paint_id}")
async def get_calibration(paint_id: str, group: str = "default"):
    """Get calibration data for a paint."""
    library = load_library(group)
    cal_map = library.get("calibration_data")
    if isinstance(cal_map, dict):
        embedded = cal_map.get(paint_id)
        if isinstance(embedded, dict):
            return embedded

    cal_file = calibration_file_for(group, paint_id)
    if not cal_file.exists():
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    with open(cal_file, 'r') as f:
        return json.load(f)


@app.get("/api/paint/calibration-export")
async def export_library_calibrations(group: str = "default"):
    """Download all calibration data for paints in the selected library group."""
    library = load_library(group)
    paints = library.get("paints", [])
    cal_map = library.get("calibration_data")
    if not isinstance(cal_map, dict):
        cal_map = {}

    export_paints = []
    for paint in paints:
        paint_id = paint.get("id")
        calibration = cal_map.get(paint_id) if paint_id else None
        if paint_id:
            if calibration is None:
                cal_file = calibration_file_for(group, paint_id)
                if cal_file.exists():
                    try:
                        with open(cal_file, "r") as f:
                            calibration = json.load(f)
                    except Exception:
                        calibration = None
        export_paints.append({
            "paint_id": paint_id,
            "paint_name": paint.get("name"),
            "hex_approx": paint.get("hex_approx"),
            "calibration": calibration,
        })

    payload = {
        "generated_at": datetime.now().isoformat(),
        "library_group": group,
        "library_name": library.get("name"),
        "paint_count": len(paints),
        "paints": export_paints,
    }

    filename = f"calibration_export_{group}.json"
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/paint/calibration/migrate-scoped")
async def migrate_calibrations_to_scoped(delete_legacy: bool = True):
    """One-time migration from legacy global calibration files to library-scoped calibration files."""
    try:
        result = migrate_global_calibrations_to_group_scope(delete_legacy=bool(delete_legacy))
        return {
            "success": True,
            **result,
            "message": "Calibration migration completed",
        }
    except Exception as e:
        logger.exception("Calibration migration failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Calibration migration failed: {e}")


@app.get("/api/paint/library/recipes")
async def list_library_recipes(
    group: str = "default",
    page: int = 1,
    page_size: int = 50,
):
    """List cached recipes for a library, sorted by hex ascending, with pagination."""
    page = max(1, int(page))
    page_size = max(1, min(200, int(page_size)))

    cache = load_recipe_cache(group)
    rows: list[dict] = []
    for hex_key in sorted(cache.keys()):
        entry = cache.get(hex_key)
        if not isinstance(entry, dict):
            continue
        recipe = entry.get("recipe")
        if not isinstance(recipe, dict):
            continue
        ingredients_in = recipe.get("ingredients")
        ingredients_out = []
        if isinstance(ingredients_in, list):
            for ing in ingredients_in:
                if not isinstance(ing, dict):
                    continue
                ingredients_out.append({
                    "paint_id": ing.get("paint_id"),
                    "paint_name": ing.get("paint_name"),
                    "percentage": ing.get("percentage"),
                })
        rows.append({
            "hex": str(hex_key).upper(),
            "last_modified": entry.get("updated_at") or recipe.get("updated_at") or recipe.get("created_at"),
            "type": entry.get("type") or recipe.get("type"),
            "delta_e": recipe.get("error"),
            "ingredients": ingredients_out,
        })

    total = len(rows)
    total_pages = max(1, (total + page_size - 1) // page_size)
    safe_page = min(page, total_pages)
    start = (safe_page - 1) * page_size
    end = start + page_size
    return {
        "group": group,
        "page": safe_page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "recipes": rows[start:end],
    }


def _gamut_library_signature(group: str, library: dict) -> str:
    paints = library.get("paints", [])
    records = []
    for p in sorted(paints, key=lambda x: str(x.get("id", ""))):
        pid = p.get("id", "")
        cal = calibration_file_for(group, pid)
        cal_sig = ""
        if cal.exists():
            stat = cal.stat()
            cal_sig = f"{stat.st_size}:{stat.st_mtime_ns}"
        records.append({
            "id": pid,
            "name": p.get("name", ""),
            "hex_approx": p.get("hex_approx", ""),
            "calibration": cal_sig,
        })
    payload = {
        "group": group,
        "records": records,
        "coverage_mg_per_cm2": library.get("coverage_mg_per_cm2"),
        "gamut_version": "v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _optimize_image_from_upload_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _optimize_image_from_path(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _downsample_longest_side(image_rgb: np.ndarray, longest_side: int = 150) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if max(h, w) <= longest_side:
        return image_rgb
    scale = float(longest_side) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _paint_name_lookup(group: str) -> dict[str, str]:
    library = load_library(group)
    out: dict[str, str] = {}
    for p in library.get("paints", []):
        pid = str(p.get("id", "")).strip()
        if not pid:
            continue
        out[pid] = str(p.get("name") or pid)
    return out


def _find_white_paint_id(group: str) -> str:
    library = load_library(group)
    for p in library.get("paints", []):
        pid = str(p.get("id", "")).lower()
        name = str(p.get("name", "")).lower()
        if "white" in pid or "white" in name:
            return str(p.get("id"))
    return "white"


def _recipe_ingredients_from_solver_recipe(
    raw_recipe: Optional[dict[str, Any]],
    paint_names: dict[str, str],
    white_paint_id: str,
) -> list[dict[str, Any]]:
    if not raw_recipe:
        return []

    components: list[tuple[str, float]] = []
    white_ratio = raw_recipe.get("white_ratio")
    if white_ratio is not None:
        try:
            wr = float(white_ratio)
            if wr > 0:
                components.append((white_paint_id, wr))
        except Exception:
            pass

    if raw_recipe.get("pigment_id") and raw_recipe.get("pigment_ratio") is not None:
        try:
            components.append((str(raw_recipe["pigment_id"]), float(raw_recipe["pigment_ratio"])))
        except Exception:
            pass
    elif raw_recipe.get("pigment1_id") and raw_recipe.get("pigment1_ratio") is not None:
        try:
            components.append((str(raw_recipe["pigment1_id"]), float(raw_recipe["pigment1_ratio"])))
        except Exception:
            pass
        if raw_recipe.get("pigment2_id") and raw_recipe.get("pigment2_ratio") is not None:
            try:
                components.append((str(raw_recipe["pigment2_id"]), float(raw_recipe["pigment2_ratio"])))
            except Exception:
                pass
    elif isinstance(raw_recipe.get("pigment_ids"), list) and isinstance(raw_recipe.get("pigment_ratios"), list):
        for pid, ratio in zip(raw_recipe.get("pigment_ids", []), raw_recipe.get("pigment_ratios", [])):
            try:
                components.append((str(pid), float(ratio)))
            except Exception:
                continue

    total = sum(max(0.0, ratio) for _, ratio in components)
    if total <= 0:
        return []

    out: list[dict[str, Any]] = []
    for pid, ratio in components:
        if ratio <= 0:
            continue
        pct = round((ratio / total) * 100.0, 2)
        out.append({
            "paint_id": pid,
            "paint_name": paint_names.get(pid, pid),
            "percentage": pct,
        })
    pct_sum = round(sum(i["percentage"] for i in out), 2)
    drift = round(100.0 - pct_sum, 2)
    if out and abs(drift) >= 0.01:
        out[-1]["percentage"] = round(out[-1]["percentage"] + drift, 2)
    return out


def _solver_recipe_pigment_count(raw_recipe: Optional[dict[str, Any]]) -> int:
    if not raw_recipe:
        return 0
    if raw_recipe.get("pigment_id"):
        return 1
    if raw_recipe.get("pigment1_id"):
        return 2 if raw_recipe.get("pigment2_id") else 1
    pigment_ids = raw_recipe.get("pigment_ids")
    if isinstance(pigment_ids, list):
        return len([pid for pid in pigment_ids if pid])
    return 0


@app.post("/api/paint/optimize-palette")
async def optimize_palette_size(
    image: Optional[UploadFile] = File(None),
    session_id: str = Form(""),
    target_delta_e: float = Form(5.0),
    max_palette_size: int = Form(16),
    library_group: str = Form("default"),
    prefer_simpler: str = Form("false"),
):
    """Find smallest palette size whose reconstructed error is <= target_delta_e."""
    if target_delta_e < 1 or target_delta_e > 15:
        raise HTTPException(status_code=400, detail="target_delta_e must be in range 1..15")
    if max_palette_size < 4 or max_palette_size > 24:
        raise HTTPException(status_code=400, detail="max_palette_size must be in range 4..24")

    use_prefer_simpler = str(prefer_simpler).lower() == "true"
    image_rgb: Optional[np.ndarray] = None
    image_signature_bytes: Optional[bytes] = None

    if image is not None:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")
        try:
            image_rgb = _optimize_image_from_upload_bytes(image_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image upload: {e}")
        image_signature_bytes = image_bytes
    else:
        sid = (session_id or "").strip()
        if not sid:
            raise HTTPException(status_code=400, detail="Provide either image or session_id")
        if not project_store.project_root(sid).is_dir():
            raise HTTPException(status_code=404, detail="Project not found")
        stored = project_store.find_source_input(sid)
        if not stored:
            raise HTTPException(status_code=404, detail="No stored source image for this project")
        try:
            image_rgb = _optimize_image_from_path(stored)
            image_signature_bytes = stored.read_bytes()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load stored image: {e}")

    assert image_rgb is not None
    assert image_signature_bytes is not None

    library = load_library(library_group)
    fingerprint = _gamut_library_signature(library_group, library)
    cache_key_payload = {
        "image_sha": hashlib.sha256(image_signature_bytes).hexdigest(),
        "target_delta_e": round(float(target_delta_e), 4),
        "max_palette_size": int(max_palette_size),
        "library_group": library_group,
        "library_signature": fingerprint,
        "prefer_simpler": use_prefer_simpler,
        "algo_version": "palette_opt_v1",
    }
    cache_key = hashlib.sha256(json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    cache_file = PALETTE_OPTIMIZATION_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    downsampled = _downsample_longest_side(image_rgb, longest_side=150)
    height, width = downsampled.shape[:2]
    rgb_norm = downsampled.astype(np.float32) / 255.0
    lab_img = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2LAB)
    lab_pixels = lab_img.reshape(-1, 3).astype(np.float32)
    lab_pixels = np.nan_to_num(lab_pixels, nan=50.0, posinf=100.0, neginf=0.0)
    lab_pixels[:, 0] = np.clip(lab_pixels[:, 0], 0, 100)
    lab_pixels[:, 1] = np.clip(lab_pixels[:, 1], -128, 127)
    lab_pixels[:, 2] = np.clip(lab_pixels[:, 2], -128, 127)
    total_pixels = lab_pixels.shape[0]
    paint_names = _paint_name_lookup(library_group)
    white_paint_id = _find_white_paint_id(library_group)

    def _kmeans_for_size(palette_size: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Cluster colours; return labels, centers, mean quant ΔE, max quant ΔE."""
        kmeans = KMeans(n_clusters=palette_size, random_state=42, n_init=3, init="k-means++")
        labels = kmeans.fit_predict(lab_pixels)
        centers = kmeans.cluster_centers_.astype(np.float32)
        per_pixel = np.zeros(total_pixels, dtype=np.float32)
        for idx, center in enumerate(centers):
            mask = labels == idx
            if not np.any(mask):
                continue
            diffs = lab_pixels[mask] - center
            per_pixel[mask] = np.sqrt(np.sum(diffs * diffs, axis=1))
        avg_q = float(np.mean(per_pixel)) if total_pixels > 0 else 999.0
        max_q = float(np.max(per_pixel)) if total_pixels > 0 else 999.0
        return labels, centers, avg_q, max_q

    def _build_result(
        palette_size: int,
        labels: np.ndarray,
        centers: np.ndarray,
    ) -> dict[str, Any]:
        solver_palette = []
        for idx, center in enumerate(centers):
            rgb_center = lab_to_rgb([float(center[0]), float(center[1]), float(center[2])])
            rgb_i = [int(max(0, min(255, round(c)))) for c in rgb_center]
            solver_palette.append({"index": idx, "rgb": rgb_i})

        logger.info("Palette optimisation: solving recipes for %s colours", palette_size)
        solver_results = generate_recipes_for_palette(
            "palette_optimization",
            solver_palette,
            library_group,
            None,
        )
        by_index = {
            int(item.get("palette_index", -1)): item
            for item in solver_results
            if item.get("palette_index") is not None
        }

        per_pixel_delta = np.zeros(total_pixels, dtype=np.float32)
        palette_rows: list[dict[str, Any]] = []

        for idx, center in enumerate(centers):
            mask = labels == idx
            if not np.any(mask):
                continue
            center_diffs = lab_pixels[mask] - center
            quant_err = np.sqrt(np.sum(center_diffs * center_diffs, axis=1))

            solver_item = by_index.get(idx, {})
            raw_recipe = solver_item.get("recipe")
            solver_err = 50.0
            if isinstance(raw_recipe, dict):
                try:
                    solver_err = float(raw_recipe.get("error", 50.0))
                except Exception:
                    solver_err = 50.0
            if use_prefer_simpler:
                pigment_count = _solver_recipe_pigment_count(raw_recipe)
                if pigment_count > 2:
                    solver_err += pigment_count * 0.5

            combined = np.sqrt((quant_err * quant_err) + (solver_err * solver_err))
            per_pixel_delta[mask] = combined.astype(np.float32)

            rgb_center = lab_to_rgb([float(center[0]), float(center[1]), float(center[2])])
            rgb_i = [int(max(0, min(255, round(c)))) for c in rgb_center]
            target_hex = "#{:02X}{:02X}{:02X}".format(rgb_i[0], rgb_i[1], rgb_i[2])
            ingredients = _recipe_ingredients_from_solver_recipe(raw_recipe, paint_names, white_paint_id)
            recipe_error = None
            if isinstance(raw_recipe, dict) and raw_recipe.get("error") is not None:
                try:
                    recipe_error = float(raw_recipe.get("error"))
                except Exception:
                    recipe_error = None
            coverage = float(np.sum(mask)) / float(total_pixels)
            palette_rows.append({
                "index": idx,
                "target_hex": target_hex,
                "coverage": round(coverage * 100.0, 2),
                "lab": [round(float(center[0]), 3), round(float(center[1]), 3), round(float(center[2]), 3)],
                "recipe": {
                    "ingredients": ingredients,
                    "error": round(recipe_error, 3) if recipe_error is not None else None,
                    "type": solver_item.get("type"),
                    "error_text": solver_item.get("error"),
                },
            })

        avg_delta = float(np.mean(per_pixel_delta)) if total_pixels > 0 else 999.0
        max_delta = float(np.max(per_pixel_delta)) if total_pixels > 0 else 999.0

        priorities: list[tuple[int, float]] = []
        for row in palette_rows:
            lab = row["lab"]
            area_score = float(row["coverage"]) / 100.0
            lightness_score = max(0.0, min(1.0, float(lab[0]) / 100.0))
            chroma = (float(lab[1]) ** 2 + float(lab[2]) ** 2) ** 0.5
            base_color_bonus = max(0.0, min(1.0, chroma / 181.0))
            priority = 0.6 * area_score + 0.3 * lightness_score + 0.1 * base_color_bonus
            priorities.append((int(row["index"]), float(priority)))
        paint_order = [idx for idx, _ in sorted(priorities, key=lambda x: x[1], reverse=True)]

        return {
            "optimal_palette_size": int(palette_size),
            "average_delta_e": round(avg_delta, 3),
            "maximum_delta_e": round(max_delta, 3),
            "target_delta_e": round(float(target_delta_e), 3),
            "max_palette_size": int(max_palette_size),
            "library_group": library_group,
            "prefer_simpler": use_prefer_simpler,
            "downsample": {"width": int(width), "height": int(height)},
            "palette": sorted(palette_rows, key=lambda x: x["index"]),
            "paint_order": paint_order,
            "met_target": bool(avg_delta <= float(target_delta_e)),
        }

    def _compute() -> dict[str, Any]:
        logger.info(
            "Palette optimisation: scanning sizes 2..%s (quant pass, then one recipe solve)",
            max_palette_size,
        )
        chosen_size = int(max_palette_size)
        chosen_labels: Optional[np.ndarray] = None
        chosen_centers: Optional[np.ndarray] = None
        fallback_size = 2
        fallback_labels: Optional[np.ndarray] = None
        fallback_centers: Optional[np.ndarray] = None

        for palette_size in range(2, int(max_palette_size) + 1):
            labels, centers, avg_q, _ = _kmeans_for_size(palette_size)
            logger.info("Palette optimisation: size=%s quant_avg=%.3f", palette_size, avg_q)
            fallback_size, fallback_labels, fallback_centers = palette_size, labels, centers
            chosen_size, chosen_labels, chosen_centers = palette_size, labels, centers
            if avg_q <= float(target_delta_e):
                break

        if chosen_labels is None or chosen_centers is None:
            chosen_size = fallback_size
            chosen_labels = fallback_labels
            chosen_centers = fallback_centers
        if chosen_labels is None or chosen_centers is None:
            raise RuntimeError("Palette optimisation failed to produce a result")

        return _build_result(chosen_size, chosen_labels, chosen_centers)

    try:
        result = await asyncio.to_thread(_compute)
    except Exception as e:
        logger.exception("Palette optimisation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Palette optimisation failed: {e}")

    try:
        atomic_write(cache_file, result)
    except Exception:
        pass
    return result


@app.get("/api/paint/gamut/slice")
async def get_palette_gamut_slice(
    group: str = "default",
    l: int = 50,
    refresh: bool = False,
):
    """Return a gamut heatmap slice for fixed L over a,b grid using recipe solver error."""
    if l < 0 or l > 100 or (l % 5 != 0):
        raise HTTPException(status_code=400, detail="l must be in 0..100 with step 5")

    library = load_library(group)
    signature = _gamut_library_signature(group, library)
    cache_file = GAMUT_CACHE_DIR / f"{group}_L{l}.json"

    if not refresh and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            if cached.get("signature") == signature:
                return cached
        except Exception:
            pass

    a_values = list(range(-100, 101, 5))
    b_values = list(range(-100, 101, 5))
    palette = []
    idx = 0
    for b_val in b_values:
        for a_val in a_values:
            rgb = lab_to_rgb([float(l), float(a_val), float(b_val)])
            rgb_i = [int(max(0, min(255, round(c)))) for c in rgb]
            hex_color = "#{:02X}{:02X}{:02X}".format(rgb_i[0], rgb_i[1], rgb_i[2])
            palette.append({
                "index": idx,
                "hex": hex_color,
                "rgb": rgb_i,
            })
            idx += 1

    # Generate recipes in chunks to avoid timeouts and ensure every cell gets a result.
    GAMUT_CHUNK_SIZE = 150
    recipe_by_idx: dict[int, dict] = {}
    for chunk_start in range(0, len(palette), GAMUT_CHUNK_SIZE):
        chunk = palette[chunk_start : chunk_start + GAMUT_CHUNK_SIZE]
        chunk_palette = [{"index": i, "hex": c["hex"], "rgb": c["rgb"]} for i, c in enumerate(chunk)]
        try:
            recipe_resp = await _compute_recipes_async(
                chunk_palette,
                group,
                force=False,
                use_ai=False,
                progress_key=None,
                quality_mode="balanced",
                cache_only=False,
            )
        except Exception as e:
            logger.exception("Gamut slice recipe chunk failed at %d: %s", chunk_start, e)
            for i, c in enumerate(chunk):
                recipe_by_idx[chunk_start + i] = {
                    "palette_index": chunk_start + i,
                    "recipe": None,
                    "error": f"Recipe generation failed: {e}",
                }
            continue
        recipe_items = recipe_resp.get("recipes", [])
        for i, item in enumerate(recipe_items):
            orig_idx = chunk_start + i
            recipe_by_idx[orig_idx] = dict(item, palette_index=orig_idx)

    # Ensure every cell has an entry (solver may omit some indices on error).
    for idx in range(len(palette)):
        if idx not in recipe_by_idx:
            recipe_by_idx[idx] = {
                "palette_index": idx,
                "recipe": None,
                "error": "Recipe generation failed: no solver output",
            }

    cells = []
    idx = 0
    for b_val in b_values:
        for a_val in a_values:
            rgb = lab_to_rgb([float(l), float(a_val), float(b_val)])
            rgb_i = [int(max(0, min(255, round(c)))) for c in rgb]
            hex_color = "#{:02X}{:02X}{:02X}".format(rgb_i[0], rgb_i[1], rgb_i[2])
            rec = recipe_by_idx.get(idx, {})
            recipe_obj = rec.get("recipe")
            error = None
            if isinstance(recipe_obj, dict):
                try:
                    e = recipe_obj.get("error")
                    error = float(e) if e is not None else None
                except Exception:
                    error = None
            if error is None:
                band = "unknown"
            elif error < 2:
                band = "excellent"
            elif error < 5:
                band = "good"
            elif error > 10:
                band = "poor"
            else:
                band = "mid"
            cells.append({
                "index": idx,
                "a": a_val,
                "b": b_val,
                "target_hex": hex_color,
                "error": error,
                "band": band,
                "recipe_data": rec,
            })
            idx += 1

    payload = {
        "generated_at": datetime.now().isoformat(),
        "group": group,
        "l": l,
        "step": 5,
        "a_min": -100,
        "a_max": 100,
        "b_min": -100,
        "b_max": 100,
        "signature": signature,
        "cells": cells,
    }
    try:
        atomic_write(cache_file, payload)
    except Exception:
        pass
    return payload


# Recipe generation (shared async runner for both sync endpoint and async jobs)
async def _compute_recipes_async(
    palette_list: list,
    library_group: str,
    force: bool,
    use_ai: bool,
    progress_key: Optional[str] = None,
    quality_mode: str = "balanced",
    cache_only: bool = False,
) -> dict:
    """Run recipe computation; returns {"recipes": [...]}. Used by sync endpoint and job worker."""
    run_started_at = time.perf_counter()
    library = load_library(library_group)
    paints = library.get("paints", [])

    def _set_progress(completed: int, total: int, status: str, current_index: int = -1, message: str = ""):
        if not progress_key:
            return
        with RECIPE_PROGRESS_LOCK:
            RECIPE_PROGRESS[progress_key] = {
                "completed": max(0, int(completed)),
                "total": max(0, int(total)),
                "status": status,
                "current_index": int(current_index),
                "message": message,
                "updated_at": datetime.now().isoformat(),
            }
        with RECIPE_JOBS_LOCK:
            job = RECIPE_JOBS.get(progress_key)
            if job is not None:
                job["completed"] = max(0, int(completed))
                job["total"] = max(0, int(total))
                job["current_index"] = int(current_index)
                job["message"] = message

    def _is_cancel_requested() -> bool:
        if not progress_key:
            return False
        with RECIPE_JOBS_LOCK:
            job = RECIPE_JOBS.get(progress_key)
            if job is None:
                return False
            return bool(job.get("cancel_requested"))

    _set_progress(0, len(palette_list), "starting", 0, "Preparing recipe generation")

    if not paints:
        _set_progress(len(palette_list), len(palette_list), "completed", -1, "No paints available")
        return {
            "recipes": [
                {
                    "palette_index": color['index'],
                    "recipe": None,
                    "error": f"No paints available in library group '{library_group}'"
                }
                for color in palette_list
            ]
        }

    paints_by_id = {p.get("id"): p for p in paints if p.get("id")}

    def _hex_to_rgb(value: str) -> Optional[list[int]]:
        v = (value or "").strip().lstrip("#")
        if len(v) != 6:
            return None
        try:
            return [int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)]
        except Exception:
            return None

    def _normalize_target_hex(color: dict) -> Optional[str]:
        if color.get("hex"):
            rgb = _hex_to_rgb(str(color["hex"]))
            if rgb is not None:
                return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])
        if color.get("rgb") and len(color["rgb"]) >= 3:
            r, g, b = color["rgb"][0], color["rgb"][1], color["rgb"][2]
            try:
                return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))
            except Exception:
                return None
        return None

    def _safe_index(value) -> int:
        try:
            return int(value)
        except Exception:
            return -1

    partial_by_index: dict[int, dict] = {}

    def _publish_partial_item(item: dict) -> None:
        if not progress_key:
            return
        idx = _safe_index(item.get("palette_index"))
        if idx < 0:
            return
        partial_by_index[idx] = item
        with RECIPE_JOBS_LOCK:
            job = RECIPE_JOBS.get(progress_key)
            if job is None:
                return
            job["partial_recipes"] = [partial_by_index[k] for k in sorted(partial_by_index.keys())]

    def _is_white_paint(paint: dict) -> bool:
        pid = str(paint.get("id", "")).lower()
        name = str(paint.get("name", "")).lower()
        if "white" in pid or "white" in name:
            return True
        rgb = _hex_to_rgb(str(paint.get("hex_approx", "")))
        if rgb is None:
            return False
        return rgb[0] > 240 and rgb[1] > 240 and rgb[2] > 240

    white_paint_id = next((p["id"] for p in paints if p.get("id") and _is_white_paint(p)), "white")

    def _library_fingerprint() -> str:
        # Invalidate cached recipes when paint definitions or calibrations change.
        records = []
        for p in sorted(paints, key=lambda x: str(x.get("id", ""))):
            pid = p.get("id", "")
            cal = calibration_file_for(library_group, pid)
            cal_sig = ""
            if cal.exists():
                stat = cal.stat()
                cal_sig = f"{stat.st_size}:{stat.st_mtime_ns}"
            records.append({
                "id": pid,
                "name": p.get("name", ""),
                "type": p.get("type", ""),
                "hex_approx": p.get("hex_approx", ""),
                "notes": p.get("notes", ""),
                "calibration": cal_sig,
            })
        payload = {
            "group": library_group,
            "coverage_mg_per_cm2": library.get("coverage_mg_per_cm2"),
            "records": records,
            "solver_version": "deterministic_v2",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]

    def _build_ingredients(components: list[tuple[str, float]], total_grams: Optional[float]) -> list[dict]:
        merged: dict[str, float] = {}
        for pid, ratio in components:
            if ratio is None:
                continue
            r = float(ratio)
            if r <= 0:
                continue
            merged[str(pid)] = merged.get(str(pid), 0.0) + r
        positive = list(merged.items())
        total_ratio = sum(r for _, r in positive)
        if total_ratio <= 0:
            return []
        out = []
        for paint_id, ratio in positive:
            pct = (ratio / total_ratio) * 100.0
            paint = paints_by_id.get(paint_id, {})
            ingredient = {
                "paint_id": paint_id,
                "paint_name": paint.get("name", paint_id),
                "percentage": round(pct, 2),
            }
            if total_grams is not None and total_grams > 0:
                ingredient["grams"] = round((pct / 100.0) * total_grams, 2)
            out.append(ingredient)
        # Force exact 100.00 after rounding by adjusting last ingredient.
        pct_sum = round(sum(i["percentage"] for i in out), 2)
        drift = round(100.0 - pct_sum, 2)
        if out and abs(drift) >= 0.01:
            out[-1]["percentage"] = round(out[-1]["percentage"] + drift, 2)
        if total_grams is not None and total_grams > 0 and out:
            # Force exact total grams after rounding by solving last ingredient.
            prev_sum = round(sum(i.get("grams", 0.0) for i in out[:-1]), 2)
            out[-1]["grams"] = round(total_grams - prev_sum, 2)
        return out

    def _predict_mix_hex(components: list[tuple[str, float]]) -> Optional[str]:
        """Predict expected mixed color (same model as recipe solver)."""
        merged: dict[str, float] = {}
        for pid, ratio in components:
            if ratio is None:
                continue
            try:
                r = float(ratio)
            except Exception:
                continue
            if r <= 0:
                continue
            key = str(pid)
            merged[key] = merged.get(key, 0.0) + r

        if not merged:
            return None

        total = sum(merged.values())
        if total <= 0:
            return None

        white_ratio = 0.0
        pigment_ratios: dict[str, float] = {}
        for paint_id, ratio in merged.items():
            paint = paints_by_id.get(paint_id, {})
            if _is_white_paint(paint):
                white_ratio += ratio
            else:
                pigment_ratios[paint_id] = pigment_ratios.get(paint_id, 0.0) + ratio

        if sum(pigment_ratios.values()) <= 1e-9:
            return "#FFFFFF"

        pigment_ids = list(pigment_ratios.keys())
        ratios_list = [pigment_ratios[k] for k in pigment_ids]
        lab = predict_mix_lab_for_paint_ratios(
            library_group, pigment_ids, ratios_list, white_ratio
        )
        if lab is None:
            return None
        r8, g8, b8 = (int(round(c)) for c in lab_to_rgb(lab))
        return "#{:02X}{:02X}{:02X}".format(
            max(0, min(255, r8)), max(0, min(255, g8)), max(0, min(255, b8))
        )

    def _recipe_to_structured(
        target_hex: str,
        generated: dict,
        total_grams: Optional[float],
    ) -> dict:
        r = generated.get("recipe")
        if not r:
            return {}

        components: list[tuple[str, float]] = []
        if r.get("white_ratio") is not None:
            components.append((white_paint_id, float(r.get("white_ratio", 0.0))))
        if r.get("pigment_ratio") is not None and r.get("pigment_id"):
            components.append((str(r.get("pigment_id")), float(r.get("pigment_ratio", 0.0))))
        if r.get("pigment1_ratio") is not None and r.get("pigment1_id"):
            components.append((str(r.get("pigment1_id")), float(r.get("pigment1_ratio", 0.0))))
        if r.get("pigment2_ratio") is not None and r.get("pigment2_id"):
            components.append((str(r.get("pigment2_id")), float(r.get("pigment2_ratio", 0.0))))
        if r.get("pigment_ids") and r.get("pigment_ratios"):
            for pid, ratio in zip(r.get("pigment_ids", []), r.get("pigment_ratios", [])):
                components.append((str(pid), float(ratio)))

        ingredients = _build_ingredients(components, total_grams)
        predicted_hex = _predict_mix_hex(components)
        error_value = r.get("error")
        try:
            error_value = float(error_value) if error_value is not None else None
        except Exception:
            error_value = None

        return {
            "target_hex": target_hex,
            "ingredients": ingredients,
            "type": "deterministic",
            "solver_recipe_type": generated.get("type", "unknown"),
            "error": error_value,
            "uncalibrated": bool(r.get("uncalibrated", False)),
            "predicted_hex": predicted_hex,
        }

    def _structured_from_ingredients(
        target_hex: str,
        ingredients_in: list[dict],
        total_grams: Optional[float],
        recipe_type: str = "ai_refined",
    ) -> Optional[dict]:
        components: list[tuple[str, float]] = []
        for ing in ingredients_in:
            if not isinstance(ing, dict):
                continue
            pid = str(ing.get("paint_id", "")).strip()
            if not pid or pid not in paints_by_id:
                continue
            pct = ing.get("percentage")
            try:
                ratio = float(pct) / 100.0
            except Exception:
                continue
            if ratio <= 0:
                continue
            components.append((pid, ratio))
        if not components:
            return None
        ingredients = _build_ingredients(components, total_grams)
        if not ingredients:
            return None
        predicted_hex = _predict_mix_hex(components)
        return {
            "target_hex": target_hex,
            "ingredients": ingredients,
            "type": recipe_type,
            "solver_recipe_type": recipe_type,
            "error": None,
            "uncalibrated": False,
            "predicted_hex": predicted_hex,
        }

    def _load_calibration_context() -> dict:
        out: dict[str, dict] = {}
        for p in paints:
            pid = p.get("id")
            if not pid:
                continue
            cal_file = calibration_file_for(library_group, pid)
            if not cal_file.exists():
                continue
            try:
                with open(cal_file, "r") as f:
                    cal = json.load(f)
                samples = cal.get("samples", [])
                compact_samples = []
                for s in samples:
                    if not isinstance(s, dict):
                        continue
                    compact_samples.append({
                        "ratio": s.get("ratio"),
                        "rgb": s.get("rgb"),
                        "lab": s.get("lab"),
                    })
                out[pid] = {
                    "samples": compact_samples,
                    "reference_strip": cal.get("reference_strip"),
                }
            except Exception:
                continue
        return out

    def _call_ai_refiner(target_hex: str, deterministic_recipe: dict, total_grams: Optional[float], calibration_ctx: dict) -> Optional[dict]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"

        paints_context = []
        for p in paints:
            pid = p.get("id")
            if not pid:
                continue
            paints_context.append({
                "paint_id": pid,
                "name": p.get("name"),
                "hex_approx": p.get("hex_approx"),
                "calibrated": pid in calibration_ctx,
            })

        payload = {
            "model": model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You refine paint-mixing recipes. Return strict JSON with key 'ingredients'. "
                        "ingredients must be a list of objects: {paint_id, percentage}. "
                        "Percentages should sum to 100. Use only provided paint_id values."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "Refine deterministic paint recipe with full calibration context.",
                            "target_hex": target_hex,
                            "deterministic_recipe": deterministic_recipe,
                            "paints": paints_context,
                            "calibration_data": calibration_ctx,
                            "constraints": {
                                "max_ingredients": 4,
                                "must_sum_to_100": True,
                            },
                        }
                    ),
                },
            ],
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
            body = json.loads(raw)
            content = body["choices"][0]["message"]["content"]
            ai_json = json.loads(content)
            ingredients = ai_json.get("ingredients", [])
            return _structured_from_ingredients(target_hex, ingredients, total_grams, "ai_refined")
        except (KeyError, ValueError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

    def _ensure_predicted_hex(recipe_obj: object) -> object:
        if not isinstance(recipe_obj, dict):
            return recipe_obj
        ingredients = recipe_obj.get("ingredients")
        if not isinstance(ingredients, list):
            return recipe_obj
        components: list[tuple[str, float]] = []
        for ing in ingredients:
            if not isinstance(ing, dict):
                continue
            paint_id = ing.get("paint_id")
            percentage = ing.get("percentage")
            if paint_id is None or percentage is None:
                continue
            try:
                ratio = float(percentage) / 100.0
            except Exception:
                continue
            components.append((str(paint_id), ratio))
        predicted_hex = _predict_mix_hex(components)
        if not predicted_hex:
            return recipe_obj
        updated = dict(recipe_obj)
        updated["predicted_hex"] = predicted_hex
        return updated

    def _apply_total_grams(recipe_obj: object, total_grams: Optional[float]) -> object:
        """Recalculate ingredient grams from percentages for current target total."""
        if not isinstance(recipe_obj, dict):
            return recipe_obj
        ingredients = recipe_obj.get("ingredients")
        if not isinstance(ingredients, list):
            return recipe_obj

        if total_grams is None or total_grams <= 0:
            # Drop stale grams when no current total is available.
            updated = dict(recipe_obj)
            new_ingredients = []
            for ing in ingredients:
                if not isinstance(ing, dict):
                    continue
                ing_new = dict(ing)
                ing_new.pop("grams", None)
                new_ingredients.append(ing_new)
            updated["ingredients"] = new_ingredients
            return updated

        updated = dict(recipe_obj)
        new_ingredients = []
        for ing in ingredients:
            if not isinstance(ing, dict):
                continue
            ing_new = dict(ing)
            try:
                pct = float(ing_new.get("percentage", 0.0))
            except Exception:
                pct = 0.0
            ing_new["grams"] = round((pct / 100.0) * float(total_grams), 2)
            new_ingredients.append(ing_new)
        if new_ingredients:
            prev_sum = round(sum(float(i.get("grams", 0.0)) for i in new_ingredients[:-1]), 2)
            new_ingredients[-1]["grams"] = round(float(total_grams) - prev_sum, 2)
        updated["ingredients"] = new_ingredients
        return updated

    try:
        phase_cache_load_ms = 0.0
        phase_cache_lookup_ms = 0.0
        phase_solver_ms = 0.0
        phase_postprocess_ms = 0.0
        phase_cache_persist_ms = 0.0
        phase_ai_refine_ms = 0.0

        fingerprint = _library_fingerprint()
        recipes_by_index: dict[int, dict] = {}
        missing_for_solver: list[dict] = []
        completed_prefix = 0
        recipe_cache: dict[str, dict] = {}
        pending_recipe_cache_updates: dict[str, dict] = {}
        if not force:
            cache_load_started_at = time.perf_counter()
            try:
                loaded_cache = load_recipe_cache(library_group)
                if isinstance(loaded_cache, dict):
                    recipe_cache = loaded_cache
            except Exception:
                recipe_cache = {}
            phase_cache_load_ms = (time.perf_counter() - cache_load_started_at) * 1000.0
    
        cancelled = False

        # Normalize and load from cache where valid.
        cache_lookup_started_at = time.perf_counter()
        for color in palette_list:
            if _is_cancel_requested():
                cancelled = True
                break
            idx = color.get("index")
            target_hex = _normalize_target_hex(color)
            if idx is None or not target_hex:
                safe_idx = _safe_index(idx)
                recipes_by_index[safe_idx] = {
                    "palette_index": idx,
                    "recipe": None,
                    "error": "Color format error: missing or invalid hex/rgb"
                }
                completed_prefix += 1
                _set_progress(completed_prefix, len(palette_list), "running", min(completed_prefix, len(palette_list) - 1))
                continue
    
            color["__target_hex"] = target_hex
            color["__target_rgb"] = _hex_to_rgb(target_hex)
            target_grams = color.get("target_grams")
            try:
                target_grams = float(target_grams) if target_grams is not None else None
            except Exception:
                target_grams = None
            color["__target_grams"] = target_grams if target_grams is not None and target_grams > 0 else None
    
            if not force:
                cached = recipe_cache.get(target_hex)
                if cached and cached.get("type") == "deterministic" and cached.get("library_fingerprint") == fingerprint:
                    cached_recipe = _ensure_predicted_hex(cached.get("recipe"))
                    cached_recipe = _apply_total_grams(cached_recipe, color.get("__target_grams"))
                    safe_idx = _safe_index(idx)
                    recipes_by_index[safe_idx] = {
                        "palette_index": safe_idx,
                        "recipe": cached_recipe,
                        "type": "deterministic"
                    }
                    _publish_partial_item(recipes_by_index[safe_idx])
                    completed_prefix += 1
                    _set_progress(completed_prefix, len(palette_list), "running", min(completed_prefix, len(palette_list) - 1))
                    continue
            missing_for_solver.append(color)
        phase_cache_lookup_ms = (time.perf_counter() - cache_lookup_started_at) * 1000.0

        if cache_only:
            ordered = []
            for color in palette_list:
                idx = _safe_index(color.get("index", -1))
                item = recipes_by_index.get(idx, {
                    "palette_index": idx,
                    "recipe": None,
                })
                if isinstance(item, dict) and isinstance(item.get("recipe"), dict):
                    item = dict(item)
                    item["recipe"] = _apply_total_grams(item.get("recipe"), color.get("__target_grams"))
                ordered.append(item)
            return {"recipes": ordered}

        if missing_for_solver and not cancelled:
            solver_palette = []
            missing_meta: dict[int, tuple[str, Optional[float]]] = {}
            for color in missing_for_solver:
                if _is_cancel_requested():
                    cancelled = True
                    break
                if color.get("__target_rgb") is None:
                    safe_idx = _safe_index(color.get("index"))
                    recipes_by_index[safe_idx] = {
                        "palette_index": safe_idx,
                        "recipe": None,
                        "error": "Color format error: invalid RGB"
                    }
                    _publish_partial_item(recipes_by_index[safe_idx])
                    completed_prefix += 1
                    _set_progress(completed_prefix, len(palette_list), "running", min(completed_prefix, len(palette_list) - 1))
                    continue
                safe_idx = _safe_index(color.get("index"))
                missing_meta[safe_idx] = (color.get("__target_hex"), color.get("__target_grams"))
                solver_palette.append({
                    "index": safe_idx,
                    "rgb": color["__target_rgb"],
                })
    
            if not cancelled:
                solver_started_at = time.perf_counter()
                try:
                    def _solver_progress(done_missing: int, total_missing: int, status: str):
                        overall_done = completed_prefix + max(0, int(done_missing))
                        current_idx = min(max(0, overall_done), max(0, len(palette_list) - 1))
                        _set_progress(overall_done, len(palette_list), "running" if status != "completed" else "finalizing", current_idx)

                    def _solver_recipe_cb(generated_item: dict):
                        idx = _safe_index(generated_item.get("palette_index"))
                        meta = missing_meta.get(idx)
                        if idx < 0 or meta is None:
                            return
                        target_hex, target_grams = meta
                        if not target_hex:
                            return
                        if not generated_item.get("recipe"):
                            partial_item = {
                                "palette_index": idx,
                                "recipe": None,
                                "error": generated_item.get("error", "Recipe generation failed")
                            }
                        else:
                            partial_item = {
                                "palette_index": idx,
                                "recipe": _recipe_to_structured(target_hex, generated_item, target_grams),
                                "type": "deterministic",
                            }
                        _publish_partial_item(partial_item)

                    generated_list = await asyncio.to_thread(
                        generate_recipes_for_palette,
                        "runtime",
                        solver_palette,
                        library_group,
                        _solver_progress,
                        _solver_recipe_cb,
                        _is_cancel_requested,
                        quality_mode,
                    )
                except Exception as e:
                    logger.exception("Recipe solver failed for library_group=%s: %s", library_group, e)
                    generated_list = []
                    for color in missing_for_solver:
                        idx = _safe_index(color.get("index", -1))
                        recipes_by_index[idx] = {
                            "palette_index": idx,
                            "recipe": None,
                            "error": f"Recipe solver crashed for this color: {e}"
                        }
                        _publish_partial_item(recipes_by_index[idx])
                phase_solver_ms = (time.perf_counter() - solver_started_at) * 1000.0
            else:
                generated_list = []
            generated_by_index = {int(r.get("palette_index")): r for r in generated_list if r.get("palette_index") is not None}
    
            postprocess_started_at = time.perf_counter()
            for color in missing_for_solver:
                if _is_cancel_requested():
                    cancelled = True
                    break
                idx = _safe_index(color.get("index"))
                target_hex = color["__target_hex"]
                target_grams = color["__target_grams"]
                generated = generated_by_index.get(idx)
                if not generated:
                    recipes_by_index[idx] = {
                        "palette_index": idx,
                        "recipe": None,
                        "error": "Recipe generation failed: no solver output"
                    }
                    _publish_partial_item(recipes_by_index[idx])
                    continue
                if not generated.get("recipe"):
                    recipes_by_index[idx] = {
                        "palette_index": idx,
                        "recipe": None,
                        "error": generated.get("error", "Recipe generation failed")
                    }
                    _publish_partial_item(recipes_by_index[idx])
                    continue
    
                recipe_storage = _recipe_to_structured(target_hex, generated, target_grams)
                pending_recipe_cache_updates[target_hex] = {
                    "type": "deterministic",
                    "library_fingerprint": fingerprint,
                    "recipe": recipe_storage,
                    "updated_at": datetime.now().isoformat(),
                }
                recipes_by_index[idx] = {
                    "palette_index": idx,
                    "recipe": recipe_storage,
                    "type": "deterministic"
                }
                _publish_partial_item(recipes_by_index[idx])
            phase_postprocess_ms = (time.perf_counter() - postprocess_started_at) * 1000.0

        if pending_recipe_cache_updates:
            cache_persist_started_at = time.perf_counter()
            try:
                recipe_cache.update(pending_recipe_cache_updates)
                save_recipe_cache(library_group, recipe_cache)
            except Exception as e:
                logger.exception("Failed to persist recipe cache for group=%s: %s", library_group, e)
            phase_cache_persist_ms = (time.perf_counter() - cache_persist_started_at) * 1000.0
    
        ordered = []
        if cancelled:
            for idx in sorted(recipes_by_index.keys()):
                color = next((c for c in palette_list if _safe_index(c.get("index", -1)) == idx), None)
                item = recipes_by_index[idx]
                if color is not None and isinstance(item, dict) and isinstance(item.get("recipe"), dict):
                    item = dict(item)
                    item["recipe"] = _apply_total_grams(item.get("recipe"), color.get("__target_grams"))
                ordered.append(item)
        else:
            for color in palette_list:
                idx = _safe_index(color.get("index", -1))
                item = recipes_by_index.get(idx, {
                    "palette_index": idx,
                    "recipe": None,
                    "error": "Recipe generation failed: missing result"
                })
                if isinstance(item, dict) and isinstance(item.get("recipe"), dict):
                    item = dict(item)
                    item["recipe"] = _apply_total_grams(item.get("recipe"), color.get("__target_grams"))
                ordered.append(item)
    
        if use_ai:
            ai_started_at = time.perf_counter()
            # AI second pass only for poor deterministic recipes to control latency/cost.
            _set_progress(len(palette_list), len(palette_list), "finalizing", max(0, len(palette_list) - 1), "Refining poor recipes with AI")
            calibration_ctx = _load_calibration_context()
            try:
                ai_refine_limit = max(0, int(os.getenv("AI_RECIPE_REFINE_LIMIT", "6")))
            except Exception:
                ai_refine_limit = 6
            refined_count = 0
            for item in ordered:
                if refined_count >= ai_refine_limit:
                    break
                recipe = item.get("recipe")
                if not isinstance(recipe, dict):
                    continue
                if recipe.get("type") != "deterministic":
                    continue
                err = recipe.get("error")
                try:
                    err_val = float(err) if err is not None else None
                except Exception:
                    err_val = None
                if err_val is not None and err_val < 6.0:
                    continue
                target_hex = recipe.get("target_hex")
                if not target_hex:
                    continue
                idx = _safe_index(item.get("palette_index"))
                grams = None
                for c in palette_list:
                    if _safe_index(c.get("index")) == idx:
                        grams = c.get("__target_grams")
                        break
                refined = _call_ai_refiner(target_hex, recipe, grams, calibration_ctx)
                if refined:
                    item["recipe"] = refined
                    item["type"] = "ai_refined"
                    refined_count += 1
            phase_ai_refine_ms = (time.perf_counter() - ai_started_at) * 1000.0

        total_ms = (time.perf_counter() - run_started_at) * 1000.0
        logger.info(
            (
                "Recipe timing group=%s colors=%d missing=%d force=%s use_ai=%s "
                "cache_load_ms=%.1f cache_lookup_ms=%.1f solver_ms=%.1f "
                "postprocess_ms=%.1f cache_persist_ms=%.1f ai_refine_ms=%.1f total_ms=%.1f"
            ),
            library_group,
            len(palette_list),
            len(missing_for_solver),
            force,
            use_ai,
            phase_cache_load_ms,
            phase_cache_lookup_ms,
            phase_solver_ms,
            phase_postprocess_ms,
            phase_cache_persist_ms,
            phase_ai_refine_ms,
            total_ms,
        )
        return {"recipes": ordered, "cancelled": cancelled}
    except Exception as e:
        logger.exception("Recipe generation failed: %s", e)
        return {
            "recipes": [
                {"palette_index": palette_list[i].get("index", i), "recipe": None, "error": f"Recipe generation failed: {e}"}
                for i in range(len(palette_list))
            ]
        }


def _hex_to_rgb_list(hex_str: str) -> Optional[list]:
    """Parse #RRGGBB or RRGGBB to [r, g, b] 0-255, or None if invalid."""
    s = (hex_str or "").strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        return [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)]
    except ValueError:
        return None


def _predict_mix_hex_standalone(library_group: str, components: list) -> Optional[str]:
    """Predict mix hex from (paint_id, ratio) components using same calibration logic as recipe generation.
    components: list of (paint_id, ratio) where ratio is proportional (e.g. 0.96, 0.04 or 9.6, 0.4).
    """
    library = load_library(library_group)
    paints = library.get("paints", [])
    if not paints:
        return None
    paints_by_id = {p.get("id"): p for p in paints if p.get("id")}

    def _hex_to_rgb_local(value: str) -> Optional[list]:
        v = (value or "").strip().lstrip("#")
        if len(v) != 6:
            return None
        try:
            return [int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)]
        except Exception:
            return None

    def _is_white_paint(paint: dict) -> bool:
        pid = str(paint.get("id", "")).lower()
        name = str(paint.get("name", "")).lower()
        if "white" in pid or "white" in name:
            return True
        rgb = _hex_to_rgb_local(str(paint.get("hex_approx", "")))
        if rgb is None:
            return False
        return rgb[0] > 240 and rgb[1] > 240 and rgb[2] > 240

    merged = {}
    for item in components:
        if not isinstance(item, (list, tuple)) and isinstance(item, dict):
            pid = item.get("paint_id") or item.get("id")
            ratio = item.get("ratio")
            if pid is None or ratio is None:
                continue
            try:
                r = float(ratio)
            except Exception:
                continue
            if r <= 0:
                continue
            key = str(pid)
            merged[key] = merged.get(key, 0.0) + r
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pid, ratio = item[0], item[1]
            try:
                r = float(ratio)
            except Exception:
                continue
            if r <= 0:
                continue
            merged[str(pid)] = merged.get(str(pid), 0.0) + r
    if not merged:
        return None

    total = sum(merged.values())
    if total <= 0:
        return None

    white_ratio = 0.0
    pigment_ratios: dict[str, float] = {}
    for paint_id, ratio in merged.items():
        paint = paints_by_id.get(paint_id, {})
        if _is_white_paint(paint):
            white_ratio += ratio
        else:
            pigment_ratios[paint_id] = pigment_ratios.get(paint_id, 0.0) + ratio

    if sum(pigment_ratios.values()) <= 1e-9:
        return "#FFFFFF"

    pigment_ids = list(pigment_ratios.keys())
    ratios_list = [pigment_ratios[k] for k in pigment_ids]
    lab = predict_mix_lab_for_paint_ratios(
        library_group, pigment_ids, ratios_list, white_ratio
    )
    if lab is None:
        return None
    r8, g8, b8 = (int(round(c)) for c in lab_to_rgb(lab))
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, r8)), max(0, min(255, g8)), max(0, min(255, b8))
    )


@app.post("/api/paint/recipes/predict-mix")
async def predict_mix(body: dict = Body(...)):
    """Predict mix color and ΔE to target using same calibration as recipe generation (for virtual mixer)."""
    library_group = (body.get("library_group") or "default").strip()
    target_hex = (body.get("target_hex") or "").strip()
    components_in = body.get("components")
    if not isinstance(components_in, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'components' array")
    components = []
    for item in components_in:
        if isinstance(item, dict):
            pid = item.get("paint_id") or item.get("id")
            ratio = item.get("ratio")
            if pid is not None and ratio is not None:
                try:
                    components.append((str(pid), float(ratio)))
                except Exception:
                    pass
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                components.append((str(item[0]), float(item[1])))
            except Exception:
                pass
    if not components:
        raise HTTPException(status_code=400, detail="No valid components (paint_id, ratio)")
    predicted_hex = _predict_mix_hex_standalone(library_group, components)
    if predicted_hex is None:
        raise HTTPException(status_code=500, detail="Could not predict mix")
    target_rgb = _hex_to_rgb_list(target_hex)
    if target_rgb is None:
        raise HTTPException(status_code=400, detail="Invalid target_hex")
    pred_rgb = _hex_to_rgb_list(predicted_hex)
    if pred_rgb is None:
        raise HTTPException(status_code=500, detail="Invalid predicted hex")
    pred_lab = rgb_to_lab([float(c) for c in pred_rgb])
    target_lab = rgb_to_lab([float(c) for c in target_rgb])
    d = delta_e_lab(pred_lab, target_lab)
    return {"predicted_hex": predicted_hex, "delta_e": float(d)}


@app.post("/api/paint/recipes/delta-e")
async def compute_delta_e(body: dict = Body(...)):
    """Compute ΔE between two hex colors using the same Lab/ΔE as recipe generation (for UI consistency)."""
    hex1 = (body.get("hex1") or "").strip()
    hex2 = (body.get("hex2") or "").strip()
    rgb1 = _hex_to_rgb_list(hex1)
    rgb2 = _hex_to_rgb_list(hex2)
    if rgb1 is None or rgb2 is None:
        raise HTTPException(status_code=400, detail="Invalid hex1 or hex2 (expected #RRGGBB or RRGGBB)")
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    d = delta_e_lab(lab1, lab2)
    return {"delta_e": float(d)}


@app.post("/api/paint/recipes/cached")
async def get_cached_recipes(
    body: dict = Body(...),
):
    """Return cached recipes only for the given palette and library. No solver run."""
    palette_list = body.get("palette")
    library_group = (body.get("library_group") or "default").strip()
    if not isinstance(palette_list, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'palette' array")
    try:
        result = await _compute_recipes_async(
            palette_list, library_group, force=False, use_ai=False,
            progress_key=None, quality_mode="balanced", cache_only=True,
        )
        return result
    except Exception as e:
        logger.exception("Cached recipes lookup failed: %s", e)
        return {
            "recipes": [
                {"palette_index": palette_list[i].get("index", i) if i < len(palette_list) else i, "recipe": None, "error": str(e)}
                for i in range(len(palette_list))
            ]
        }


@app.post("/api/paint/recipes/from-palette")
async def generate_recipes_from_palette(
    palette: str = Form(...),
    library_group: str = Form("default"),
    force_regenerate: str = Form("false"),
    use_ai_second_pass: str = Form("false"),
    progress_id: str = Form(""),
    quality_mode: str = Form("balanced"),
):
    """Generate deterministic recipes (sync). For long palettes use POST /api/paint/recipes/jobs to avoid proxy timeouts."""
    try:
        palette_list = json.loads(palette)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid palette JSON: {e}")
    if not isinstance(palette_list, list):
        raise HTTPException(status_code=400, detail="Invalid palette JSON: expected a list")
    force = force_regenerate.lower() == "true"
    use_ai = use_ai_second_pass.lower() == "true"
    progress_key = (progress_id or "").strip() or None
    try:
        return await _compute_recipes_async(palette_list, library_group, force, use_ai, progress_key, quality_mode)
    except Exception as e:
        logger.exception("Recipe generation failed: %s", e)
        return {
            "recipes": [
                {"palette_index": palette_list[i].get("index", i), "recipe": None, "error": f"Recipe generation failed: {e}"}
                for i in range(len(palette_list))
            ]
        }


@app.post("/api/paint/recipes/jobs")
async def start_recipe_job(
    palette: str = Form(...),
    library_group: str = Form("default"),
    force_regenerate: str = Form("false"),
    use_ai_second_pass: str = Form("false"),
    quality_mode: str = Form("balanced"),
):
    """Start recipe generation as a background job. Returns job_id; poll GET /api/paint/recipes/jobs/{job_id} for result."""
    try:
        palette_list = json.loads(palette)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid palette JSON: {e}")
    if not isinstance(palette_list, list):
        raise HTTPException(status_code=400, detail="Invalid palette JSON: expected a list")
    force = force_regenerate.lower() == "true"
    use_ai = use_ai_second_pass.lower() == "true"
    job_id = str(uuid.uuid4())
    with RECIPE_JOBS_LOCK:
        RECIPE_JOBS[job_id] = {
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "total": len(palette_list),
            "completed": 0,
            "current_index": 0,
            "message": "Queued",
            "cancel_requested": False,
            "partial_recipes": [],
        }
    asyncio.create_task(_run_recipe_job(job_id, palette_list, library_group, force, use_ai, quality_mode))
    return Response(status_code=202, content=json.dumps({"job_id": job_id}), media_type="application/json")


async def _run_recipe_job(
    job_id: str,
    palette_list: list,
    library_group: str,
    force: bool,
    use_ai: bool,
    quality_mode: str,
):
    """Background task: run recipe computation and store result in RECIPE_JOBS[job_id]."""
    try:
        with RECIPE_JOBS_LOCK:
            if job_id in RECIPE_JOBS:
                RECIPE_JOBS[job_id]["status"] = "running"
                RECIPE_JOBS[job_id]["message"] = "Running"
        result = await _compute_recipes_async(
            palette_list,
            library_group,
            force,
            use_ai,
            progress_key=job_id,
            quality_mode=quality_mode,
        )
        with RECIPE_JOBS_LOCK:
            if job_id in RECIPE_JOBS:
                cancelled = bool(result.get("cancelled")) or bool(RECIPE_JOBS[job_id].get("cancel_requested"))
                RECIPE_JOBS[job_id]["status"] = "cancelled" if cancelled else "completed"
                RECIPE_JOBS[job_id]["result"] = result
                RECIPE_JOBS[job_id]["completed_at"] = datetime.now().isoformat()
    except Exception as e:
        logger.exception("Recipe job %s failed: %s", job_id, e)
        with RECIPE_JOBS_LOCK:
            if job_id in RECIPE_JOBS:
                RECIPE_JOBS[job_id]["status"] = "failed"
                RECIPE_JOBS[job_id]["error"] = str(e)
                RECIPE_JOBS[job_id]["completed_at"] = datetime.now().isoformat()


@app.get("/api/paint/recipes/jobs/{job_id}")
async def get_recipe_job(job_id: str):
    """Poll for async recipe job result. Returns status (pending|completed|failed) and result when completed."""
    with RECIPE_JOBS_LOCK:
        job = RECIPE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    out = {
        "job_id": job_id,
        "status": job["status"],
        "total": job.get("total", 0),
        "completed": int(job.get("completed", 0) or 0),
        "current_index": int(job.get("current_index", 0) or 0),
        "message": job.get("message", ""),
    }
    with RECIPE_PROGRESS_LOCK:
        progress = RECIPE_PROGRESS.get(job_id)
    if progress:
        out["total"] = int(progress.get("total", out["total"]) or out["total"])
        out["completed"] = int(progress.get("completed", out["completed"]) or out["completed"])
        out["current_index"] = int(progress.get("current_index", out["current_index"]) or out["current_index"])
        out["message"] = progress.get("message", out["message"])
        out["progress_status"] = progress.get("status")
    if job.get("status") in ("completed", "cancelled") and "result" in job:
        out["recipes"] = job["result"].get("recipes", [])
    if isinstance(job.get("partial_recipes"), list):
        out["partial_recipes"] = job["partial_recipes"]
    if job.get("status") == "failed" and "error" in job:
        out["error"] = job["error"]
    return out


@app.post("/api/paint/recipes/jobs/{job_id}/cancel")
async def cancel_recipe_job(job_id: str):
    """Request cancellation of a running recipe job."""
    with RECIPE_JOBS_LOCK:
        job = RECIPE_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        if job.get("status") in ("completed", "failed", "cancelled"):
            return {"job_id": job_id, "status": job.get("status"), "cancel_requested": False}
        job["cancel_requested"] = True
        job["message"] = "Cancellation requested"
    return {"job_id": job_id, "status": "cancelling", "cancel_requested": True}


@app.get("/api/paint/recipes/progress/{progress_id}")
async def get_recipe_progress(progress_id: str):
    with RECIPE_PROGRESS_LOCK:
        progress = RECIPE_PROGRESS.get(progress_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Progress not found")
    return progress


# Feedback bias (spot-test corrections)
@app.get("/api/paint/feedback-bias")
async def get_feedback_bias(group: str = "default"):
    """Return current per-paint Lab bias for the given library group (from spot-test feedback)."""
    group = (group or "default").strip() or "default"
    biases = load_feedback_bias(group)
    return {"group": group, "biases": biases}


@app.post("/api/paint/feedback-bias/reset")
async def reset_feedback_bias(body: dict = Body(...)):
    """Remove a previous spot-test correction: one paint or all for the library. After reset, next Generate recipes will recompute without that bias."""
    group = (body.get("group") or "default").strip() or "default"
    paint_id = (body.get("paint_id") or "").strip() or None
    biases = load_feedback_bias(group)
    if paint_id:
        key = _bias_key(paint_id)
        if key in biases:
            del biases[key]
            save_feedback_bias(group, biases)
    else:
        save_feedback_bias(group, {})
        biases = {}
    invalidate_recipe_cache(group)
    return {"group": group, "removed": paint_id or "all", "biases": biases}


# Verification endpoints
@app.post("/api/paint/verify/upload")
async def upload_verification_photo(
    image: UploadFile = File(...),
    session_id: str = Form(...),
    palette_index: int = Form(...)
):
    """Upload a verification swatch photo."""
    if not project_store.project_root(session_id).is_dir():
        raise HTTPException(status_code=404, detail="Project not found")

    verify_dir = project_store.artifacts_dir(session_id) / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)

    image_id = str(uuid.uuid4())
    image_path = verify_dir / f"{palette_index}_{image_id}.jpg"

    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    return {
        "image_id": image_id,
        "preview_url": f"/api/projects/{session_id}/artifacts/verify/{palette_index}_{image_id}.jpg",
    }


@app.get("/api/projects/{project_id}/artifacts/verify/{filename}")
async def get_verification_image(project_id: str, filename: str, request: Request):
    """Serve verification swatch image with CORS headers."""
    file_path = project_store.artifacts_dir(project_id) / "verify" / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        file_path.resolve().relative_to(project_store.artifacts_dir(project_id).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    return _cors_file_response(file_path, request)


def _parse_recipe_components(recipe_json: Optional[str]) -> list[tuple[str, float]]:
    """Parse recipe JSON into [(paint_id, ratio), ...]. Ratio in 0..1. White excluded from bias updates."""
    if not recipe_json or not recipe_json.strip():
        return []
    try:
        data = json.loads(recipe_json)
    except Exception:
        return []
    out = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            pid = item.get("paint_id") or item.get("id")
            if not pid:
                continue
            r = item.get("ratio")
            if r is None and "percentage" in item:
                r = (item.get("percentage") or 0) / 100.0
            if r is not None:
                try:
                    out.append((str(pid), float(r)))
                except Exception:
                    pass
    return out


# Library spot-test (no session): use existing swatch photos to correct recipe model
@app.post("/api/paint/spot-test/upload")
async def spot_test_upload(
    image: UploadFile = File(...),
    library_group: str = Form("default"),
):
    """Upload a swatch photo for standalone spot test (Paint Library tab)."""
    group = (library_group or "default").strip() or "default"
    group_dir = SPOT_TEST_DIR / group
    group_dir.mkdir(parents=True, exist_ok=True)
    image_id = str(uuid.uuid4())
    image_path = group_dir / f"{image_id}.jpg"
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    return {
        "image_id": image_id,
        "preview_url": f"/api/paint/spot-test/image/{group}/{image_id}",
    }


@app.get("/api/paint/spot-test/image/{group}/{image_id}")
async def spot_test_image(group: str, image_id: str, request: Request):
    """Serve spot-test upload with CORS (image stored as {image_id}.jpg)."""
    path = SPOT_TEST_DIR / group / f"{image_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        path.resolve().relative_to(SPOT_TEST_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    response = FileResponse(path)
    origin = request.headers.get("origin")
    if origin and (origin in allowed_origins or "margies.app" in origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.post("/api/paint/spot-test/sample")
async def spot_test_sample(
    library_group: str = Form("default"),
    image_id: str = Form(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    target_hex: str = Form(""),
    recipe: str = Form(""),
    focus_paint_id: str = Form(""),
    apply_feedback: str = Form("true"),
):
    """Sample region from spot-test image, compare to target_hex, optionally update feedback bias (no session)."""
    group = (library_group or "default").strip() or "default"
    image_path = SPOT_TEST_DIR / group / f"{image_id}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Spot-test image not found")
    measured_rgb, measured_lab = sample_color_from_region(
        str(image_path), x1, y1, x2, y2
    )
    measured_lab = list(measured_lab) if measured_lab else [50.0, 0.0, 0.0]
    result = {
        "measured_rgb": measured_rgb,
        "measured_lab": measured_lab,
        "delta_e": None,
        "feedback_updated": False,
        "paints_updated": [],
    }
    hex_str = (target_hex or "").strip().lstrip("#")
    if len(hex_str) != 6:
        return result
    hex_str = "#" + hex_str
    target_rgb = _hex_to_rgb_list(hex_str)
    if target_rgb is None:
        return result
    target_lab = rgb_to_lab([float(c) for c in target_rgb])
    delta_e = delta_e_lab(measured_lab, target_lab)
    result["target_hex"] = hex_str
    result["target_lab"] = target_lab
    result["delta_e"] = round(float(delta_e), 3)
    if apply_feedback.lower() != "true" or not recipe.strip():
        return result
    components = _parse_recipe_components(recipe)
    white_ids = {"white", "titanium white", "zinc white"}
    pigment_components = [(pid, r) for pid, r in components if str(pid).strip().lower() not in {w.lower() for w in white_ids}]
    if not pigment_components:
        return result
    total_pigment = sum(r for _, r in pigment_components)
    if total_pigment <= 0:
        return result
    error_lab = [
        measured_lab[0] - target_lab[0],
        measured_lab[1] - target_lab[1],
        measured_lab[2] - target_lab[2],
    ]
    alpha = 1.0
    focus_key = _bias_key(focus_paint_id) if (focus_paint_id and str(focus_paint_id).strip()) else None
    biases = load_feedback_bias(group)
    for pid, ratio in pigment_components:
        key = _bias_key(pid)
        if focus_key is not None:
            if key != focus_key:
                continue
            share = 1.0
        else:
            share = ratio / total_pigment
        update = [alpha * share * error_lab[0], alpha * share * error_lab[1], alpha * share * error_lab[2]]
        existing = biases.get(key, [0.0, 0.0, 0.0])
        biases[key] = [
            existing[0] + update[0],
            existing[1] + update[1],
            existing[2] + update[2],
        ]
        result["paints_updated"].append(pid)
    save_feedback_bias(group, biases)
    invalidate_recipe_cache(group)
    result["feedback_updated"] = True
    return result


@app.post("/api/paint/verify/sample")
async def verify_swatch(
    session_id: str = Form(...),
    palette_index: int = Form(...),
    image_id: str = Form(...),
    x: Optional[int] = Form(None),
    y: Optional[int] = Form(None),
    x1: Optional[int] = Form(None),
    y1: Optional[int] = Form(None),
    x2: Optional[int] = Form(None),
    y2: Optional[int] = Form(None),
    library_group: str = Form("default"),
    target_hex: str = Form(""),
    recipe: str = Form(""),
    apply_feedback: str = Form("true"),
    focus_paint_id: str = Form(""),
):
    """Sample verification swatch and compare to target. Average over a region (x1,y1,x2,y2) if provided, else over a small area around (x,y). If target_hex and recipe are provided and apply_feedback is true, updates per-paint feedback bias. If focus_paint_id is set, apply 100%% of the correction to that paint only (so other recipes using that paint learn from this spot test)."""
    if not project_store.project_root(session_id).is_dir():
        raise HTTPException(status_code=404, detail="Project not found")

    verify_dir = project_store.artifacts_dir(session_id) / "verify"
    image_files = list(verify_dir.glob(f"{palette_index}_{image_id}*"))
    if not image_files:
        raise HTTPException(status_code=404, detail="Verification image not found")

    image_path = image_files[0]
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        measured_rgb, measured_lab = sample_color_from_region(
            str(image_path), x1, y1, x2, y2
        )
    else:
        if x is None or y is None:
            raise HTTPException(
                status_code=400,
                detail="Provide either (x, y) or (x1, y1, x2, y2) for the sample area",
            )
        measured_rgb, measured_lab = sample_color_from_image(str(image_path), x, y)
    measured_lab = list(measured_lab) if measured_lab else [50.0, 0.0, 0.0]

    result = {
        "measured_rgb": measured_rgb,
        "measured_lab": measured_lab,
        "delta_e": None,
        "feedback_updated": False,
        "paints_updated": [],
    }

    hex_str = (target_hex or "").strip().lstrip("#")
    if len(hex_str) != 6:
        return result
    hex_str = "#" + hex_str

    target_rgb = _hex_to_rgb_list(hex_str)
    if target_rgb is None:
        return result
    target_lab = rgb_to_lab([float(c) for c in target_rgb])
    delta_e = delta_e_lab(measured_lab, target_lab)
    result["target_hex"] = hex_str
    result["target_lab"] = target_lab
    result["delta_e"] = round(float(delta_e), 3)

    if apply_feedback.lower() != "true" or not recipe.strip():
        return result

    components = _parse_recipe_components(recipe)
    if not components:
        return result

    # Exclude white so we only bias pigment calibrations
    white_ids = {"white", "titanium white", "zinc white"}
    pigment_components = [(pid, r) for pid, r in components if str(pid).strip().lower() not in {w.lower() for w in white_ids}]
    if not pigment_components:
        return result

    total_pigment = sum(r for _, r in pigment_components)
    if total_pigment <= 0:
        return result

    error_lab = [
        measured_lab[0] - target_lab[0],
        measured_lab[1] - target_lab[1],
        measured_lab[2] - target_lab[2],
    ]
    alpha = 1.0  # Full correction so one spot test has a visible effect; bias accumulates over multiple tests
    group = (library_group or "default").strip() or "default"
    focus_key = _bias_key(focus_paint_id) if (focus_paint_id and str(focus_paint_id).strip()) else None
    biases = load_feedback_bias(group)
    for pid, ratio in pigment_components:
        key = _bias_key(pid)
        if focus_key is not None:
            if key != focus_key:
                continue
            share = 1.0
        else:
            share = ratio / total_pigment
        update = [alpha * share * error_lab[0], alpha * share * error_lab[1], alpha * share * error_lab[2]]
        existing = biases.get(key, [0.0, 0.0, 0.0])
        biases[key] = [
            existing[0] + update[0],
            existing[1] + update[1],
            existing[2] + update[2],
        ]
        result["paints_updated"].append(pid)
    save_feedback_bias(group, biases)
    result["feedback_updated"] = True
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
