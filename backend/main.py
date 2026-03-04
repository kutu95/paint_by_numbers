from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional
import uuid
import shutil
from datetime import datetime, timedelta
import os
import traceback
import logging
from image_processor import process_image, regenerate_pure_mask_from_labels
from paint_manager import (
    load_library, save_library, slugify, atomic_write,
    sample_color_from_image, sample_color_from_region, normalize_calibration_samples,
    get_hex_from_calibration,
    rgb_to_lab, delta_e_lab, CALIBRATION_DIR, PAINT_DIR,
    list_library_groups, get_library_info,
    get_cached_recipe, cache_recipe
)
import json
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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

# Data directory relative to backend folder, go up one level to project root
DATA_DIR = Path(__file__).parent.parent / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BACKEND_DIR = Path(__file__).parent
ENV_FILE = BACKEND_DIR / ".env"


def _get_openai_api_key() -> Optional[str]:
    """Return OpenAI API key from environment or from backend/.env file."""
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
                        break
        except Exception:
            pass
    return None

SESSION_CLEANUP_HOURS = 24 * 30  # 30 days so project images persist when reopened


def cleanup_old_sessions():
    """Delete sessions older than SESSION_CLEANUP_HOURS."""
    if not DATA_DIR.exists():
        return
    
    cutoff = datetime.now() - timedelta(hours=SESSION_CLEANUP_HOURS)
    for session_dir in DATA_DIR.iterdir():
        if session_dir.is_dir():
            try:
                mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(session_dir)
            except Exception:
                pass


@app.on_event("startup")
async def startup_event():
    cleanup_old_sessions()


@app.post("/api/sessions")
async def create_session(
    image: UploadFile = File(...),
    n_colors: int = Form(16),
    overpaint_mm: float = Form(5.0),
    order_mode: str = Form("largest"),
    max_side: int = Form(1920),
    saturation_boost: float = Form(1.0),
    detail_level: float = Form(0.5),
    canvas_width_cm: float = Form(0),
    canvas_height_cm: float = Form(0),
):
    """Create a new session and process the image."""
    try:
        logger.info(f"Received request: n_colors={n_colors}, overpaint_mm={overpaint_mm}, order_mode={order_mode}, max_side={max_side}, image={image.filename}")
    except Exception as e:
        logger.error(f"Error logging request: {e}")
    
    # Validate inputs
    if n_colors < 2 or n_colors > 100:
        logger.error(f"Invalid n_colors: {n_colors}")
        raise HTTPException(status_code=400, detail="n_colors must be between 2 and 100")
    if overpaint_mm < 0 or overpaint_mm > 50:
        logger.error(f"Invalid overpaint_mm: {overpaint_mm}")
        raise HTTPException(status_code=400, detail="overpaint_mm must be between 0 and 50")
    if order_mode not in ["largest", "smallest", "manual", "lightest"]:
        logger.error(f"Invalid order_mode: {order_mode}")
        raise HTTPException(status_code=400, detail="order_mode must be largest, smallest, manual, or lightest")
    if max_side < 100 or max_side > 5000:
        logger.error(f"Invalid max_side: {max_side}")
        raise HTTPException(status_code=400, detail="max_side must be between 100 and 5000")
    if saturation_boost < 0.5 or saturation_boost > 5.0:
        logger.error(f"Invalid saturation_boost: {saturation_boost}")
        raise HTTPException(status_code=400, detail="saturation_boost must be between 0.5 and 5.0")
    if detail_level < 0.0 or detail_level > 1.0:
        logger.error(f"Invalid detail_level: {detail_level}")
        raise HTTPException(status_code=400, detail="detail_level must be between 0.0 and 1.0")
    
    # Create session directory
    session_id = str(uuid.uuid4())
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded image - preserve original extension or default to jpg
    file_ext = Path(image.filename).suffix if image.filename else '.jpg'
    if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        file_ext = '.jpg'
    image_path = session_dir / f"input{file_ext}"
    
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    
    logger.info(f"Saved uploaded image to {image_path}, size: {image_path.stat().st_size} bytes")
    
    try:
        # Process image
        logger.info(f"Processing image: {image_path}, n_colors={n_colors}, overpaint_mm={overpaint_mm}, saturation_boost={saturation_boost}")
        result = process_image(
            str(image_path),
            session_dir,
            n_colors,
            overpaint_mm,
            order_mode,
            max_side,
            saturation_boost,
            detail_level,
        )
        
        # Attach session ID and canvas dimensions (original/oriented URLs from process_image)
        result['session_id'] = session_id
        result['canvas_width_cm'] = max(0, float(canvas_width_cm))
        result['canvas_height_cm'] = max(0, float(canvas_height_cm))
        return result
    except Exception as e:
        # Log the full traceback for debugging
        logger.error(f"Processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        # Cleanup on error
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


def _find_stored_input_path(session_dir: Path) -> Optional[Path]:
    """Find the stored original image in a session directory (input.jpg, input.png, etc.)."""
    for name in ["input.jpg", "input.jpeg", "input.png", "input.webp", "input.bmp", "input.gif"]:
        p = session_dir / name
        if p.exists() and p.is_file():
            return p
    # Fallback: any file starting with 'input.'
    for f in session_dir.iterdir():
        if f.is_file() and f.name.startswith("input."):
            return f
    return None


@app.get("/api/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """Return minimal session info (e.g. original_url) so the client can show stored image without re-upload."""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists() or not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found or expired")
    original_path = session_dir / "original_oriented.jpg"
    has_stored = _find_stored_input_path(session_dir) is not None
    return {
        "session_id": session_id,
        "original_url": f"/api/sessions/{session_id}/original_oriented.jpg" if original_path.exists() else None,
        "has_stored_image": has_stored,
    }


@app.post("/api/sessions/{session_id}/reprocess")
async def reprocess_session(
    session_id: str,
    n_colors: int = Form(16),
    overpaint_mm: float = Form(5.0),
    order_mode: str = Form("largest"),
    max_side: int = Form(1920),
    saturation_boost: float = Form(1.0),
    detail_level: float = Form(0.5),
    canvas_width_cm: float = Form(0),
    canvas_height_cm: float = Form(0),
):
    """Reprocess a session using the stored original image (no upload). Use when editing project settings only."""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists() or not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found or expired")
    image_path = _find_stored_input_path(session_dir)
    if not image_path:
        raise HTTPException(status_code=404, detail="No stored original image for this session. Upload a new image instead.")
    # Validate same as create_session
    if n_colors < 2 or n_colors > 100:
        raise HTTPException(status_code=400, detail="n_colors must be between 2 and 100")
    if overpaint_mm < 0 or overpaint_mm > 50:
        raise HTTPException(status_code=400, detail="overpaint_mm must be between 0 and 50")
    if order_mode not in ["largest", "smallest", "manual", "lightest"]:
        raise HTTPException(status_code=400, detail="order_mode must be largest, smallest, manual, or lightest")
    if max_side < 100 or max_side > 5000:
        raise HTTPException(status_code=400, detail="max_side must be between 100 and 5000")
    if saturation_boost < 0.5 or saturation_boost > 5.0:
        raise HTTPException(status_code=400, detail="saturation_boost must be between 0.5 and 5.0")
    if detail_level < 0.0 or detail_level > 1.0:
        raise HTTPException(status_code=400, detail="detail_level must be between 0.0 and 1.0")
    try:
        result = process_image(
            str(image_path),
            session_dir,
            n_colors,
            overpaint_mm,
            order_mode,
            max_side,
            saturation_boost,
            detail_level,
        )
        result["session_id"] = session_id
        result["canvas_width_cm"] = max(0, float(canvas_width_cm))
        result["canvas_height_cm"] = max(0, float(canvas_height_cm))
        return result
    except Exception as e:
        logger.error(f"Reprocess failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")


@app.options("/api/sessions/{session_id}/{filename}")
async def options_session_file(session_id: str, filename: str, request: Request):
    """Handle CORS preflight for session files."""
    origin = request.headers.get("origin", "https://layerpainter.margies.app")
    
    # Always allow our domain - set CORS headers unconditionally
    origin_lower = origin.lower() if origin else ""
    if (not origin or 
        "layerpainter.margies.app" in origin_lower or 
        "margies.app" in origin_lower or
        origin in allowed_origins or
        origin_lower.startswith("https://layerpainter") or
        origin_lower.startswith("http://localhost")):
        cors_origin = origin if origin else "https://layerpainter.margies.app"
    else:
        cors_origin = "https://layerpainter.margies.app"
    
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": cors_origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        }
    )


@app.get("/api/sessions/{session_id}/{filename}")
async def get_session_file(session_id: str, filename: str, request: Request):
    """Serve session files with CORS headers for canvas/image loading."""
    file_path = DATA_DIR / session_id / filename

    session_dir = DATA_DIR / session_id
    if not file_path.exists() or not file_path.is_file():
        # Pure mask: regenerate from labels when possible (exact quantized region, no gaps)
        if filename.endswith("_pure_mask.png"):
            try:
                layer_index = int(filename.replace("layer_", "").replace("_pure_mask.png", ""))
            except ValueError:
                layer_index = -1
            if layer_index >= 0 and regenerate_pure_mask_from_labels(session_dir, layer_index):
                pass  # file_path now exists, fall through to serve
            else:
                raise HTTPException(status_code=404, detail="Pure mask unavailable")
        else:
            raise HTTPException(status_code=404, detail="File not found")
    elif filename.endswith("_pure_mask.png"):
        # Always regenerate pure mask from labels when available so display has no gaps
        try:
            layer_index = int(filename.replace("layer_", "").replace("_pure_mask.png", ""))
        except ValueError:
            layer_index = -1
        if layer_index >= 0:
            regenerate_pure_mask_from_labels(session_dir, layer_index)
    
    # Security check: ensure file is within session directory
    try:
        file_path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Return FileResponse with explicit CORS headers - ALWAYS set for our domain
    # This is required for CSS mask-image and canvas loading to work across origins
    response = FileResponse(file_path)
    
    # Get origin from request header
    origin = request.headers.get("origin")
    
    # Always set CORS headers - use origin if it's from our domain, otherwise default to our domain
    if origin:
        origin_lower = origin.lower()
        if ("layerpainter.margies.app" in origin_lower or 
            "margies.app" in origin_lower or
            origin in allowed_origins or
            origin_lower.startswith("https://layerpainter") or
            origin_lower.startswith("http://localhost")):
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
            # Default to our domain if origin doesn't match
            response.headers["Access-Control-Allow-Origin"] = "https://layerpainter.margies.app"
    else:
        # No origin header (common with CSS mask-image) - allow our domain
        response.headers["Access-Control-Allow-Origin"] = "https://layerpainter.margies.app"
    
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "86400"
    
    return response


# ===== Settings (OpenAI key) =====

@app.get("/api/settings/openai-key/configured")
async def get_openai_key_configured():
    """Return whether an OpenAI API key is configured (without revealing it)."""
    return {"configured": bool(_get_openai_api_key())}


@app.post("/api/settings/openai-key")
async def set_openai_key(request: Request):
    """Save OpenAI API key to backend/.env. Takes effect immediately for recipe generation."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")
    key = body.get("key")
    if key is None:
        raise HTTPException(status_code=400, detail="Missing 'key'")
    key = str(key).strip()
    if not key:
        raise HTTPException(status_code=400, detail="Key cannot be empty")
    try:
        temp_file = ENV_FILE.with_suffix(".env.tmp")
        with open(temp_file, "w") as f:
            f.write(f"OPENAI_API_KEY={key}\n")
        temp_file.replace(ENV_FILE)
    except Exception:
        logger.exception("Failed to write .env")
        raise HTTPException(status_code=500, detail="Failed to save key")
    return {"ok": True}


# ===== Paint Management Endpoints =====

@app.get("/api/paint/library")
async def get_paint_library(group: str = "default"):
    """Get the paint library for a specific group."""
    return load_library(group)


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
    hex_from_cal = get_hex_from_calibration(paint_id)
    if hex_from_cal:
        paint['hex_approx'] = hex_from_cal
        logger.info("Update paint: using hex from calibration for %s: %s", paint_id, hex_from_cal)
    else:
        paint['hex_approx'] = hex_approx

    # Save current group
    save_library(library, group)

    # Sync to other groups that contain this paint
    if hex_from_cal:
        for g in list_library_groups():
            if g == group:
                continue
            lib = load_library(g)
            for p in lib.get('paints', []):
                if p.get('id') == paint_id:
                    p['name'] = paint['name']
                    p['hex_approx'] = paint['hex_approx']
                    p['notes'] = paint['notes']
                    save_library(lib, g)
                    break

    return paint


@app.delete("/api/paint/library/{paint_id}")
async def delete_paint(paint_id: str, group: str = "default"):
    """Delete a paint from the library."""
    library = load_library(group)
    library['paints'] = [p for p in library['paints'] if p['id'] != paint_id]
    save_library(library, group)
    
    # Also delete calibration if it exists
    cal_file = CALIBRATION_DIR / f"{paint_id}.json"
    if cal_file.exists():
        cal_file.unlink()
    
    return {"success": True}


# Calibration endpoints
@app.post("/api/paint/calibration/upload")
async def upload_calibration_photo(
    image: UploadFile = File(...),
    paint_id: str = Form(...)
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
    
    cal_file = CALIBRATION_DIR / f"{paint_id}.json"
    atomic_write(cal_file, calibration)
    
    # Update the paint's approximate color (hex_approx) from the 100% (1.0) swatch in every library group that has this paint
    sample_100 = next((s for s in samples if s.get("ratio", 0) >= 0.99), None)
    if sample_100 and sample_100.get("rgb") and len(sample_100["rgb"]) >= 3:
        r, g, b = sample_100["rgb"][0], sample_100["rgb"][1], sample_100["rgb"][2]
        hex_from_calibration = "#{:02x}{:02x}{:02x}".format(
            max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
        )
        for group in list_library_groups():
            lib = load_library(group)
            for p in lib.get("paints", []):
                if p.get("id") == paint_id:
                    p["hex_approx"] = hex_from_calibration
                    save_library(lib, group)
                    break
    
    return {
        "samples": samples,
        "reference_strip": reference if reference else None,
        "calibration_saved": True
    }


@app.get("/api/paint/calibration/{paint_id}")
async def get_calibration(paint_id: str):
    """Get calibration data for a paint."""
    cal_file = CALIBRATION_DIR / f"{paint_id}.json"
    if not cal_file.exists():
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    with open(cal_file, 'r') as f:
        return json.load(f)


# Recipe generation
@app.post("/api/paint/recipes/from-palette")
async def generate_recipes_from_palette(
    palette: str = Form(...),  # JSON string of palette
    library_group: str = Form("default"),  # Library group to use
    force_regenerate: str = Form("false")  # Force regeneration, ignore cache
):
    """Generate recipes from a provided palette using ChatGPT."""
    palette_list = json.loads(palette)
    force = force_regenerate.lower() == "true"
    
    # Load paints from the specified library group
    library = load_library(library_group)
    paints = library.get('paints', [])
    
    if not paints:
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
    
    # Initialize OpenAI client (key from env or backend/.env)
    api_key = _get_openai_api_key()
    if not api_key:
        logger.error("OPENAI API key not set (env or .env)")
        return {
            "recipes": [
                {
                    "palette_index": color['index'],
                    "recipe": None,
                    "error": "OpenAI API key not configured"
                }
                for color in palette_list
            ]
        }
    
    client = OpenAI(api_key=api_key)
    
    recipes = []
    for color in palette_list:
        try:
            # Support both 'hex' and 'rgb' format (for backward compatibility)
            if 'hex' in color:
                target_hex = color['hex']
            elif 'rgb' in color:
                # Convert RGB to hex
                r, g, b = color['rgb']
                target_hex = f"#{r:02x}{g:02x}{b:02x}"
            else:
                logger.error(f"No hex or rgb in color data: {color}")
                recipes.append({
                    "palette_index": color['index'],
                    "recipe": None,
                    "error": "Color format error: missing hex or rgb"
                })
                continue
            
            # Check if recipe is already cached (unless forcing regeneration)
            # Recipes are cached by COLOR (hex value), not by layer/palette index
            # This ensures recipes are reused across images when colors match
            if not force:
                cached = get_cached_recipe(library_group, target_hex)
                if cached:
                    # Validate that cached recipe is from ChatGPT (not old algorithm)
                    cached_type = cached.get("type", "")
                    if cached_type == "chatgpt" and cached.get("recipe", {}).get("type") == "chatgpt":
                        logger.info(f"Using cached ChatGPT recipe for color {target_hex} in group {library_group}")
                        recipes.append({
                            "palette_index": color['index'],
                            "recipe": cached.get("recipe"),
                            "type": "chatgpt"
                        })
                        continue
                    else:
                        # Old cached recipe from previous algorithm - ignore and regenerate
                        logger.warning(f"Found old cached recipe (type: {cached_type}) for color {target_hex}, regenerating with ChatGPT")
            
            # Recipe not in cache (or force regenerate), generate new one with ChatGPT
            if force:
                logger.info(f"Force regenerating recipe for color {target_hex} in group {library_group}")
            else:
                logger.info(f"Generating new recipe for color {target_hex} in group {library_group}")
            
            # Build paint set structure for function calling
            paint_set_paints = []
            for paint in paints:
                paint_set_paints.append({
                    "id": paint.get('id', paint.get('name', 'Unknown').upper()[:1]),
                    "name": paint.get('name', paint.get('id', 'Unknown')),
                    "pigments": paint.get('notes', 'Unknown')
                })
            
            # Build paint set info (get library name if available)
            library_name = library.get('name', library_group.replace("-", " ").title())
            paint_set = {
                "brand": library_name,  # Use library name as brand
                "range": f"{library_name} Acrylics",
                "paints": paint_set_paints
            }
            
            # System prompt
            system_prompt = """You are an expert paint mixer for artist acrylics and mural work.

Your task is to generate practical, repeatable paint mixing recipes that approximate given target colors using ONLY the provided paint set.

You MUST call the provided function `return_paint_recipes` exactly once and return structured data that conforms strictly to the function schema. Do not output any prose or explanation outside the function call.

Rules and constraints:
- Use ONLY the paints provided. Never invent or substitute paints.
- Default output is percentages that sum to 100.00 (±0.05).
- If `total_grams` is provided and greater than zero, also calculate grams for each ingredient such that the total equals `total_grams` (±0.05g). Percentages must still be included.
- Assume subtractive paint mixing; HEX values are approximations.
- Prefer tints (mostly Titanium White) for light colors.
- Use Carbon Black sparingly unless the target is near-black.
- Neutralize saturation primarily with oxides rather than black.
- Keep procedures practical and step-based; mixing order matters.
- Provide a clear mixing strategy, expected result, adjustment ladder (micro tweaks), and practical tips.
- Ingredient percentages and grams must be internally consistent and validated.

Quality guidelines:
- Light colors should be predominantly white.
- Dark colors should start from black or deep blue-black.
- Muted colors should be neutralized carefully to avoid muddiness.
- Adjustment ladders must use very small increments (e.g. 0.01–0.05 parts).

Now generate the recipes using the following inputs and return them via the function call only."""
            
            # Define function schema
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "return_paint_recipes",
                        "description": "Return paint mixing recipes for target HEX colors",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "recipes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "target_hex": {
                                                "type": "string",
                                                "description": "Target color in HEX format (e.g., #FF0000)"
                                            },
                                            "ingredients": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "paint_name": {
                                                            "type": "string",
                                                            "description": "Exact name of the paint as provided in the paint set"
                                                        },
                                                        "percentage": {
                                                            "type": "number",
                                                            "description": "Percentage of this ingredient (0-100)"
                                                        },
                                                        "grams": {
                                                            "type": "number",
                                                            "description": "Grams of this ingredient (only if total_grams was provided)"
                                                        }
                                                    },
                                                    "required": ["paint_name", "percentage"]
                                                },
                                                "description": "List of paint ingredients with percentages"
                                            },
                                            "mixing_strategy": {
                                                "type": "string",
                                                "description": "Step-by-step mixing procedure"
                                            },
                                            "expected_result": {
                                                "type": "string",
                                                "description": "Description of the expected mixed color"
                                            },
                                            "adjustment_ladder": {
                                                "type": "string",
                                                "description": "Micro adjustments for fine-tuning (small increments)"
                                            },
                                            "tips": {
                                                "type": "string",
                                                "description": "Practical tips for mixing this color"
                                            }
                                        },
                                        "required": ["target_hex", "ingredients", "mixing_strategy", "expected_result", "adjustment_ladder", "tips"]
                                    }
                                }
                            },
                            "required": ["recipes"]
                        }
                    }
                }
            ]
            
            # Build user message
            user_message = json.dumps({
                "paint_set": paint_set,
                "targets": [target_hex]
            })
            
            # Call ChatGPT with function calling
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "return_paint_recipes"}},
                    temperature=0.2
                )
                
                if not response or not response.choices or len(response.choices) == 0:
                    raise ValueError("ChatGPT returned empty response")
                
                message = response.choices[0].message
                if not message:
                    raise ValueError("ChatGPT response has no message")
                
                # Check for function call
                if not message.tool_calls or len(message.tool_calls) == 0:
                    # Log the full message to see what we got
                    logger.error(f"ChatGPT did not call function. Message content: {message.content if hasattr(message, 'content') else 'N/A'}")
                    logger.error(f"Full message: {message}")
                    raise ValueError("ChatGPT did not call the required function")
                
                tool_call = message.tool_calls[0]
                if tool_call.function.name != "return_paint_recipes":
                    logger.error(f"ChatGPT called wrong function: {tool_call.function.name}")
                    raise ValueError(f"ChatGPT called wrong function: {tool_call.function.name}")
                
                # Parse function arguments
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse function arguments: {tool_call.function.arguments}")
                    logger.error(f"JSON decode error: {e}")
                    raise ValueError(f"Invalid JSON in function arguments: {e}")
                
                recipes_data = function_args.get("recipes", [])
                
                if not recipes_data or len(recipes_data) == 0:
                    logger.error(f"ChatGPT returned no recipes. Function args: {function_args}")
                    raise ValueError("ChatGPT returned no recipes")
                
                recipe_data = recipes_data[0]  # Get first (and only) recipe
                
                # Validate recipe data
                if not recipe_data:
                    logger.error(f"Recipe data is None or empty: {recipe_data}")
                    raise ValueError("Recipe data is empty")
                
                ingredients = recipe_data.get("ingredients", [])
                if not ingredients or len(ingredients) == 0:
                    logger.error(f"Recipe has no ingredients. Recipe data: {recipe_data}")
                    raise ValueError("Recipe has no ingredients")
                
                # Format recipe for storage (keep structured data)
                recipe_storage = {
                    "target_hex": recipe_data.get("target_hex", target_hex),
                    "ingredients": ingredients,
                    "mixing_strategy": recipe_data.get("mixing_strategy", ""),
                    "expected_result": recipe_data.get("expected_result", ""),
                    "adjustment_ladder": recipe_data.get("adjustment_ladder", ""),
                    "tips": recipe_data.get("tips", ""),
                    "type": "chatgpt"
                }
                
                logger.info(f"ChatGPT generated recipe for {target_hex} with {len(ingredients)} ingredients: {[ing.get('paint_name') for ing in ingredients]}")
                logger.debug(f"Full recipe storage: {json.dumps(recipe_storage, indent=2)}")
                
                # Cache the recipe for future use
                cache_recipe(library_group, target_hex, {
                    "recipe": recipe_storage,
                    "type": "chatgpt"
                })
                
                recipes.append({
                    "palette_index": color['index'],
                    "recipe": recipe_storage,
                    "type": "chatgpt"
                })
                
            except Exception as chatgpt_error:
                error_msg = str(chatgpt_error)
                logger.error(f"ChatGPT API error for color {target_hex}: {error_msg}")
                logger.error(traceback.format_exc())
                
                # NO FALLBACK - throw clear error to user
                recipes.append({
                    "palette_index": color['index'],
                    "recipe": None,
                    "error": f"ChatGPT API error: {error_msg}. Please check OpenAI API key and try again."
                })
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating recipe for color {color.get('hex', 'unknown')}: {error_msg}")
            logger.error(traceback.format_exc())
            recipes.append({
                "palette_index": color['index'],
                "recipe": None,
                "error": f"Recipe generation failed: {error_msg}"
            })
    
    return {"recipes": recipes}


# Verification endpoints
@app.post("/api/paint/verify/upload")
async def upload_verification_photo(
    image: UploadFile = File(...),
    session_id: str = Form(...),
    palette_index: int = Form(...)
):
    """Upload a verification swatch photo."""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    verify_dir = session_dir / "verify"
    verify_dir.mkdir(exist_ok=True)
    
    image_id = str(uuid.uuid4())
    image_path = verify_dir / f"{palette_index}_{image_id}.jpg"
    
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    
    return {
        "image_id": image_id,
        "preview_url": f"/api/sessions/{session_id}/verify/{palette_index}_{image_id}.jpg"
    }


@app.get("/api/sessions/{session_id}/verify/{filename}")
async def get_verification_image(session_id: str, filename: str, request: Request):
    """Serve verification image with CORS headers."""
    file_path = DATA_DIR / session_id / "verify" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security check
    try:
        file_path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    response = FileResponse(file_path)
    origin = request.headers.get("origin")
    if origin:
        if origin in allowed_origins or "layerpainter.margies.app" in origin or "margies.app" in origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
    return response


@app.post("/api/paint/verify/sample")
async def verify_swatch(
    session_id: str = Form(...),
    palette_index: int = Form(...),
    image_id: str = Form(...),
    x: int = Form(...),
    y: int = Form(...)
):
    """Sample verification swatch and compare to target."""
    session_dir = DATA_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find the image file
    verify_dir = session_dir / "verify"
    image_files = list(verify_dir.glob(f"{palette_index}_{image_id}*"))
    if not image_files:
        raise HTTPException(status_code=404, detail="Verification image not found")
    
    image_path = image_files[0]
    
    # Sample color
    rgb, lab = sample_color_from_image(str(image_path), x, y)
    
    # Get target color from session (would need to load session data)
    # For MVP, return the measured values and let frontend handle comparison
    # TODO: Load session palette and compare
    
    return {
        "measured_rgb": rgb,
        "measured_lab": lab,
        "suggestion": "Compare measured Lab to target Lab. If too light, increase white. If hue off, add small amount of closest pigment."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
