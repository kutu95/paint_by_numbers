from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
import shutil
from datetime import datetime, timedelta
import os
import traceback
import logging
from image_processor import process_image
from paint_manager import (
    load_library, save_library, slugify, atomic_write,
    sample_color_from_image, generate_recipes_for_palette,
    rgb_to_lab, delta_e_lab, CALIBRATION_DIR, PAINT_DIR
)
import json
from paint_manager import (
    load_library, save_library, slugify, atomic_write,
    sample_color_from_image, generate_recipes_for_palette,
    rgb_to_lab, delta_e_lab, CALIBRATION_DIR, PAINT_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware - allow origins from environment or default to localhost
# Default includes common ports for compatibility
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:3002,http://localhost:3003").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory relative to backend folder, go up one level to project root
DATA_DIR = Path(__file__).parent.parent / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SESSION_CLEANUP_HOURS = 24


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
    max_side: int = Form(1920)
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
    if order_mode not in ["largest", "smallest", "manual"]:
        logger.error(f"Invalid order_mode: {order_mode}")
        raise HTTPException(status_code=400, detail="order_mode must be largest, smallest, or manual")
    if max_side < 100 or max_side > 5000:
        logger.error(f"Invalid max_side: {max_side}")
        raise HTTPException(status_code=400, detail="max_side must be between 100 and 5000")
    
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
        logger.info(f"Processing image: {image_path}, n_colors={n_colors}, overpaint_mm={overpaint_mm}")
        result = process_image(
            str(image_path),
            session_dir,
            n_colors,
            overpaint_mm,
            order_mode,
            max_side
        )
        
        result['session_id'] = session_id
        return result
    except Exception as e:
        # Log the full traceback for debugging
        logger.error(f"Processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        # Cleanup on error
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/sessions/{session_id}/{filename}")
async def get_session_file(session_id: str, filename: str):
    """Serve session files."""
    file_path = DATA_DIR / session_id / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security check: ensure file is within session directory
    try:
        file_path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(file_path)


# ===== Paint Management Endpoints =====

@app.get("/api/paint/library")
async def get_paint_library():
    """Get the paint library."""
    return load_library()


@app.post("/api/paint/library")
async def add_paint(
    name: str = Form(...),
    hex_approx: str = Form(...),
    notes: str = Form("")
):
    """Add a new paint to the library."""
    library = load_library()
    paint_id = slugify(name)
    
    # Check if ID already exists
    existing = [p for p in library['paints'] if p['id'] == paint_id]
    if existing:
        raise HTTPException(status_code=400, detail=f"Paint with ID '{paint_id}' already exists")
    
    new_paint = {
        "id": paint_id,
        "name": name,
        "type": "base",
        "hex_approx": hex_approx,
        "notes": notes
    }
    
    library['paints'].append(new_paint)
    save_library(library)
    
    return new_paint


@app.put("/api/paint/library/{paint_id}")
async def update_paint(
    paint_id: str,
    name: str = Form(...),
    hex_approx: str = Form(...),
    notes: str = Form("")
):
    """Update an existing paint."""
    library = load_library()
    paint = next((p for p in library['paints'] if p['id'] == paint_id), None)
    if not paint:
        raise HTTPException(status_code=404, detail="Paint not found")
    
    paint['name'] = name
    paint['hex_approx'] = hex_approx
    paint['notes'] = notes
    
    save_library(library)
    return paint


@app.delete("/api/paint/library/{paint_id}")
async def delete_paint(paint_id: str):
    """Delete a paint from the library."""
    library = load_library()
    library['paints'] = [p for p in library['paints'] if p['id'] != paint_id]
    save_library(library)
    
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
async def get_calibration_temp_image(image_id: str):
    """Serve temporary calibration image."""
    image_path = CALIBRATION_DIR / "temp" / f"{image_id}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


@app.post("/api/paint/calibration/sample")
async def sample_calibration_colors(
    image_id: str = Form(...),
    paint_id: str = Form(...),
    points: str = Form(...),  # JSON string of [{x,y}, ...]
    ratios: str = Form(...)   # JSON string of [ratio, ...]
):
    """Sample colors from calibration photo and save calibration."""
    image_path = CALIBRATION_DIR / "temp" / f"{image_id}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    points_list = json.loads(points)
    ratios_list = json.loads(ratios)
    
    if len(points_list) != len(ratios_list):
        raise HTTPException(status_code=400, detail="Points and ratios must have same length")
    
    samples = []
    for point, ratio in zip(points_list, ratios_list):
        rgb, lab = sample_color_from_image(str(image_path), point['x'], point['y'])
        samples.append({
            "ratio": ratio,
            "rgb": rgb,
            "lab": lab
        })
    
    # Save calibration
    calibration = {
        "paint_id": paint_id,
        "ratios": ratios_list,
        "samples": samples,
        "created_at": datetime.now().isoformat(),
        "notes": ""
    }
    
    cal_file = CALIBRATION_DIR / f"{paint_id}.json"
    atomic_write(cal_file, calibration)
    
    return {
        "samples": samples,
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
    palette: str = Form(...)  # JSON string of palette
):
    """Generate recipes from a provided palette."""
    palette_list = json.loads(palette)
    recipes = generate_recipes_for_palette("", palette_list)
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
async def get_verification_image(session_id: str, filename: str):
    """Serve verification image."""
    file_path = DATA_DIR / session_id / "verify" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


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

