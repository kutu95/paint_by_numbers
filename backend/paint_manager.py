import json
import cv2
import numpy as np
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
from datetime import datetime
import os
import re
import time
import logging

logger = logging.getLogger(__name__)


# Data directories
PAINT_DIR = Path(__file__).parent.parent / "data" / "paint"
PAINT_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_DIR = PAINT_DIR / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_FILE = PAINT_DIR / "library.json"
LIBRARIES_DIR = PAINT_DIR / "libraries"
LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)
RECIPES_CACHE_DIR = PAINT_DIR / "recipes_cache"
RECIPES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_BIAS_DIR = PAINT_DIR / "feedback_bias"
FEEDBACK_BIAS_DIR.mkdir(parents=True, exist_ok=True)

_CALIBRATION_CACHE: Dict[str, Tuple[int, Dict[str, Any]]] = {}

# ----- Substrate (paper/canvas) black-point compensation defaults -----
# Real paint on real paper cannot reach L*=0; predicted darks otherwise read too dark
# vs. what the user actually sees painted. We lift predicted L* monotonically below L_break
# without touching mids/lights. Per-library overrides come from library["substrate_compensation"].
_SUBSTRATE_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "L_paper_min": 10.0,   # darkest L* achievable on the substrate (output floor)
    "L_break": 35.0,       # L* threshold; values >= L_break pass through unchanged
    "alpha_dark": 1.25,    # >1 lifts the deepest darks more (concave); 1.0 = linear remap
}
_SUBSTRATE_PARAMS_CACHE: Dict[str, Tuple[int, Dict[str, float]]] = {}


def _feedback_bias_file(group: str) -> Path:
    safe = (group or "default").strip() or "default"
    return FEEDBACK_BIAS_DIR / f"{safe}.json"


def _bias_key(paint_id: str) -> str:
    """Canonical key for feedback bias so recipe and library lookups match (case-insensitive)."""
    return str(paint_id or "").strip().lower()


def load_feedback_bias(group: str) -> Dict[str, List[float]]:
    """Load per-paint Lab bias from feedback (L, a, b corrections). Keys normalized to lower case."""
    path = _feedback_bias_file(group)
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        out = {}
        for pid, v in raw.items():
            key = _bias_key(pid)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                out[key] = [float(v[0]), float(v[1]), float(v[2])]
            elif isinstance(v, dict) and "L" in v and "a" in v and "b" in v:
                out[key] = [float(v["L"]), float(v["a"]), float(v["b"])]
        return out
    except Exception:
        return {}


def save_feedback_bias(group: str, biases: Dict[str, List[float]]) -> None:
    """Save per-paint Lab bias (L, a, b). Keys normalized so lookup matches library paint ids."""
    path = _feedback_bias_file(group)
    payload = {_bias_key(pid): {"L": b[0], "a": b[1], "b": b[2]} for pid, b in biases.items()}
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save feedback bias for %s: %s", group, e)


def get_paint_bias(group: str, paint_id: str) -> Optional[List[float]]:
    """Return [L, a, b] bias for a paint, or None if none. Lookup is case-insensitive."""
    biases = load_feedback_bias(group)
    return biases.get(_bias_key(paint_id))


def calibration_file_for(group: str, paint_id: str) -> Path:
    safe_group = (group or "default").strip() or "default"
    safe_paint = (paint_id or "").strip()
    return CALIBRATION_DIR / f"{safe_group}__{safe_paint}.json"


def _load_calibration_cached(paint_id: str, group: str = "default") -> Optional[Dict[str, Any]]:
    """Load calibration JSON with mtime-based in-process cache."""
    cache_key = f"{group}:{paint_id}"
    cal_file = calibration_file_for(group, paint_id)
    if not cal_file.exists():
        _CALIBRATION_CACHE.pop(cache_key, None)
        return None
    try:
        mtime_ns = cal_file.stat().st_mtime_ns
        cached = _CALIBRATION_CACHE.get(cache_key)
        if cached and cached[0] == mtime_ns:
            return cached[1]
        with open(cal_file, 'r') as f:
            calibration = json.load(f)
        if isinstance(calibration, dict):
            _CALIBRATION_CACHE[cache_key] = (mtime_ns, calibration)
            return calibration
    except Exception:
        return None
    return None


def _empty_library(group: str) -> Dict:
    return {
        "version": 1,
        "paints": [],
        "group": group,
        "calibration_data": {},
        "recipes": {},
    }


def _coerce_library_shape(data: object, group: str) -> Dict:
    """Normalize library payload to a safe shape."""
    if not isinstance(data, dict):
        return _empty_library(group)

    paints = data.get("paints", [])
    if not isinstance(paints, list):
        paints = []
    paints = [p for p in paints if isinstance(p, dict)]

    normalized = dict(data)
    normalized["version"] = int(normalized.get("version", 1) or 1)
    normalized["group"] = str(normalized.get("group") or group)
    normalized["paints"] = paints
    calibration_data = normalized.get("calibration_data", {})
    normalized["calibration_data"] = calibration_data if isinstance(calibration_data, dict) else {}
    recipes = normalized.get("recipes", {})
    normalized["recipes"] = recipes if isinstance(recipes, dict) else {}
    return normalized


def _target_wants_umber_white_floor(target_lab: List[float]) -> bool:
    """Whether burnt-umber dilution should raise min white (warm mid earth tones only)."""
    lab = _coerce_lab_to_cielab(target_lab)
    lightness = float(lab[0])
    a, b = float(lab[1]), float(lab[2])
    chroma = (a * a + b * b) ** 0.5
    if lightness < 28.0 or lightness > 78.0 or chroma > 38.0:
        return False
    # Warm orange/brown: positive a* and modest yellow (not cyans/blues: b* << 0).
    return a > 8.0 and b > -5.0


def get_white_mix_limits(
    target_lab: List[float],
    library_group: str = "default",
) -> Tuple[float, float]:
    """Return (min_white_ratio, max_total_pigment) based on target lightness."""
    lab = _coerce_lab_to_cielab(target_lab)
    lightness = float(lab[0])
    chroma = (float(lab[1]) ** 2 + float(lab[2]) ** 2) ** 0.5
    if lightness < 25.0:
        min_white_ratio = 0.01
    elif lightness < 40.0:
        min_white_ratio = 0.03
    elif lightness < 55.0:
        min_white_ratio = 0.08
    elif lightness < 70.0:
        min_white_ratio = 0.12 if chroma > 30.0 else 0.15
    else:
        # Light saturated hues need pigment headroom; high min-white washed out cyans.
        min_white_ratio = 0.05 if chroma > 28.0 else 0.12

    # Burnt-umber calibration floor: only for warm mid-tone earth colors (not #41C3D8-style cyans).
    if _target_wants_umber_white_floor(target_lab):
        cal_umber = _load_calibration_cached("burnt-umber", library_group)
        if cal_umber is not None:
            r_inv = invert_calibration_ratio_for_L(cal_umber, lightness)
            if r_inv is not None:
                curve_white = 1.0 - float(r_inv)
                min_white_ratio = max(min_white_ratio, curve_white * 0.18)

    max_total_pigment = 1.0 - min_white_ratio
    return min_white_ratio, max_total_pigment


def _value_paint_ratio_cap(
    paint_id: str,
    calibration: Optional[Dict],
    target_lab: List[float],
    library_group: str,
) -> Optional[float]:
    """Max mass fraction for a value paint from inverted L* on its calibration curve."""
    pid = (paint_id or "").strip().lower()
    if pid not in _VALUE_PAINT_IDS or calibration is None:
        return None
    lightness = float(_coerce_lab_to_cielab(target_lab)[0])
    if lightness < 20.0 or lightness > 85.0:
        return None
    r_inv = invert_calibration_ratio_for_L(calibration, lightness, group=library_group, paint_id=paint_id)
    if r_inv is None:
        return None
    # Stay near the calibration L* inversion (physical mixes run darker than the model).
    return min(0.75, float(r_inv) * 1.02)


def _apply_physical_value_caps_to_recipe(
    recipe: Dict,
    target_lab: List[float],
    library_group: str,
) -> Dict:
    """Cap value-paint mass using calibration L* curves; excess goes to white."""
    if not isinstance(recipe, dict):
        return recipe
    out = dict(recipe)
    if out.get("pigment_ids") and out.get("pigment_ratios"):
        ids = list(out["pigment_ids"])
        ratios = [float(r) for r in out["pigment_ratios"]]
        white = float(out.get("white_ratio", 0.0))
        freed = 0.0
        for i, pid in enumerate(ids):
            cal = _load_calibration_cached(pid, library_group)
            cap = _value_paint_ratio_cap(pid, cal, target_lab, library_group)
            if cap is not None and ratios[i] > cap + 1e-6:
                freed += ratios[i] - cap
                ratios[i] = cap
        if freed > 0:
            white += freed
        out["pigment_ratios"] = ratios
        out["white_ratio"] = min(1.0, white)
        return out
    for key in ("pigment_ratio", "pigment1_ratio", "pigment2_ratio"):
        if key not in out:
            continue
        pid_key = key.replace("_ratio", "_id")
        pid = out.get(pid_key) or out.get("pigment_id")
        if not pid:
            continue
        cal = _load_calibration_cached(str(pid), library_group)
        cap = _value_paint_ratio_cap(str(pid), cal, target_lab, library_group)
        if cap is not None and float(out[key]) > cap:
            freed = float(out[key]) - cap
            out[key] = cap
            out["white_ratio"] = min(1.0, float(out.get("white_ratio", 0.0)) + freed)
    return out


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text


def atomic_write(filepath: Path, data: dict):
    """Write JSON file atomically (write temp then rename)."""
    temp_file = filepath.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(data, f, indent=2)
    temp_file.replace(filepath)


def load_library(group: str = "default") -> Dict:
    """Load paint library from JSON for a specific group.
    
    Args:
        group: Library group name (e.g., "matisse", "dulux", "default")
    
    Returns:
        Dictionary with version and paints list
    """
    # For backward compatibility, check old library.json first
    if group == "default" and LIBRARY_FILE.exists():
        try:
            with open(LIBRARY_FILE, 'r') as f:
                data = json.load(f)
                # Migrate to new structure if needed
                if isinstance(data, dict) and "groups" not in data:
                    coerced = _coerce_library_shape(data, group)
                    needs_save = "calibration_data" not in data or "recipes" not in data
                    if not coerced.get("calibration_data"):
                        cal_map: Dict[str, Dict[str, Any]] = {}
                        for paint in coerced.get("paints", []):
                            pid = str(paint.get("id", "")).strip()
                            if not pid:
                                continue
                            cal_file = calibration_file_for(group, pid)
                            if not cal_file.exists():
                                continue
                            try:
                                with open(cal_file, "r") as cf:
                                    loaded = json.load(cf)
                                if isinstance(loaded, dict):
                                    cal_map[pid] = loaded
                            except Exception:
                                continue
                        if cal_map:
                            coerced["calibration_data"] = cal_map
                            needs_save = True
                    if needs_save:
                        save_library(coerced, group)
                    return coerced
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning("Failed to load legacy library file %s: %s", LIBRARY_FILE, e)
    
    # Load from group-specific file
    library_file = LIBRARIES_DIR / f"{group}.json"
    if not library_file.exists():
        return _empty_library(group)
    try:
        with open(library_file, 'r') as f:
            data = json.load(f)
            coerced = _coerce_library_shape(data, group)
            needs_save = "calibration_data" not in data or "recipes" not in data
            if not coerced.get("calibration_data"):
                cal_map: Dict[str, Dict[str, Any]] = {}
                for paint in coerced.get("paints", []):
                    pid = str(paint.get("id", "")).strip()
                    if not pid:
                        continue
                    cal_file = calibration_file_for(group, pid)
                    if not cal_file.exists():
                        continue
                    try:
                        with open(cal_file, "r") as cf:
                            loaded = json.load(cf)
                        if isinstance(loaded, dict):
                            cal_map[pid] = loaded
                    except Exception:
                        continue
                if cal_map:
                    coerced["calibration_data"] = cal_map
                    needs_save = True
            if needs_save:
                save_library(coerced, group)
            return coerced
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("Failed to load library group file %s: %s", library_file, e)
        return _empty_library(group)


def save_library(data: Dict, group: str = "default"):
    """Save paint library to JSON for a specific group.
    
    Args:
        data: Library data dictionary
        group: Library group name
    """
    # Ensure group is set in data
    data["group"] = group
    
    # Save to group-specific file
    library_file = LIBRARIES_DIR / f"{group}.json"
    atomic_write(library_file, data)
    
    # For backward compatibility, also save to old location if default
    if group == "default":
        atomic_write(LIBRARY_FILE, data)


def list_library_groups() -> List[str]:
    """List all available library groups."""
    groups = ["default"]  # Always include default
    
    # Scan for group files
    for file in LIBRARIES_DIR.glob("*.json"):
        group_name = file.stem
        if group_name != "default":
            groups.append(group_name)
    
    return sorted(groups)


def get_recipe_cache_file(group: str) -> Path:
    """Get the recipe cache file path for a library group.
    
    Args:
        group: Library group name
    
    Returns:
        Path to the recipe cache file
    """
    return RECIPES_CACHE_DIR / f"{group}_recipes.json"


def _derive_confidence(recipe: Dict) -> str:
    if not isinstance(recipe, dict):
        return "unknown"
    try:
        err = recipe.get("error")
        err_val = float(err) if err is not None else None
    except Exception:
        err_val = None
    if err_val is None:
        return "unknown"
    if err_val < 2.0:
        return "excellent"
    if err_val < 5.0:
        return "good"
    if err_val <= 10.0:
        return "acceptable"
    return "poor"


def load_recipe_cache(group: str) -> Dict[str, Dict]:
    """Load recipe cache for a library group.
    
    Recipes are cached by hex color (normalized to uppercase).
    
    Args:
        group: Library group name
    
    Returns:
        Dictionary mapping hex color to cached recipe data
    """
    library = load_library(group)
    recipes = library.get("recipes", {})
    if isinstance(recipes, dict) and recipes:
        return recipes

    # Backward compatibility: migrate old cache file into library JSON.
    cache_file = get_recipe_cache_file(group)
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        library["recipes"] = data
        save_library(library, group)
        return data
    except (json.JSONDecodeError, IOError):
        return {}


def save_recipe_cache(group: str, cache: Dict[str, Dict]):
    """Save recipe cache for a library group.
    
    Args:
        group: Library group name
        cache: Dictionary mapping hex color to recipe data
    """
    library = load_library(group)
    library["recipes"] = cache if isinstance(cache, dict) else {}
    save_library(library, group)


def invalidate_recipe_cache(group: str) -> None:
    """Clear the recipe cache for a library group so next Generate recipes uses updated bias."""
    save_recipe_cache(group, {})


def get_cached_recipe(group: str, hex_color: str) -> Optional[Dict]:
    """Get a cached recipe for a color and library group.
    
    IMPORTANT: Recipes are cached by COLOR (hex value), NOT by layer number.
    This ensures that:
    - New images reuse recipes for colors that match (same hex value)
    - Different colors in the same layer position don't share recipes
    - Recipes persist across sessions and images
    
    Args:
        group: Library group name (recipes are cached per library group)
        hex_color: Target color hex (e.g., "#FF0000" or "FF0000")
    
    Returns:
        Cached recipe dictionary, or None if not found
    """
    # Normalize hex color to uppercase with # for consistent matching
    hex_normalized = hex_color.upper().lstrip('#')
    if not hex_normalized.startswith('#'):
        hex_normalized = '#' + hex_normalized
    
    cache = load_recipe_cache(group)
    return cache.get(hex_normalized)


def cache_recipe(group: str, hex_color: str, recipe: Dict):
    """Cache a recipe for a color and library group.
    
    IMPORTANT: Recipes are cached by COLOR (hex value), NOT by layer number.
    This allows recipes to be reused across different images when the same color
    appears, regardless of which layer number it occupies.
    
    Args:
        group: Library group name (recipes are cached per library group)
        hex_color: Target color hex (e.g., "#FF0000" or "FF0000")
        recipe: Recipe data dictionary to cache
    """
    # Normalize hex color to uppercase with # for consistent matching
    hex_normalized = hex_color.upper().lstrip('#')
    if not hex_normalized.startswith('#'):
        hex_normalized = '#' + hex_normalized
    
    cache = load_recipe_cache(group)
    entry = dict(recipe) if isinstance(recipe, dict) else {"recipe": recipe}
    entry["updated_at"] = datetime.now().isoformat()
    if "confidence" not in entry:
        entry["confidence"] = _derive_confidence(entry.get("recipe") if isinstance(entry.get("recipe"), dict) else entry)
    cache[hex_normalized] = entry
    save_recipe_cache(group, cache)


def get_library_info(group: str) -> Dict:
    """Get information about a library group."""
    library = load_library(group)
    paints = [p for p in library.get('paints', []) if isinstance(p, dict)]
    cal_map = library.get("calibration_data")
    if not isinstance(cal_map, dict):
        cal_map = {}
    paint_count = len(paints)
    calibrated_count = 0
    for p in paints:
        pid = p.get("id", "")
        if not pid:
            continue
        if isinstance(cal_map.get(pid), dict):
            calibrated_count += 1
            continue
        group_file = calibration_file_for(group, pid)
        if group_file.exists():
            calibrated_count += 1
    
    # Use stored name if available, otherwise generate from group ID
    name = library.get('name', group.replace("-", " ").title())
    
    return {
        "group": group,
        "paint_count": paint_count,
        "calibrated_count": calibrated_count,
        "name": name,
        "coverage_mg_per_cm2": library.get("coverage_mg_per_cm2"),
    }


def _merge_calibration_data_from_files(group: str, library: Dict) -> Dict[str, Dict]:
    """Return calibration_data with any on-disk scoped files merged in."""
    cal_map = library.get("calibration_data")
    if not isinstance(cal_map, dict):
        cal_map = {}
    merged = dict(cal_map)
    for paint in library.get("paints", []):
        if not isinstance(paint, dict):
            continue
        pid = str(paint.get("id", "")).strip()
        if not pid:
            continue
        if isinstance(merged.get(pid), dict):
            continue
        file_cal = _load_calibration_cached(pid, group)
        if isinstance(file_cal, dict):
            merged[pid] = file_cal
    return merged


def _clear_group_calibration_files(group: str) -> None:
    prefix = f"{(group or 'default').strip() or 'default'}__"
    if not CALIBRATION_DIR.exists():
        return
    for path in CALIBRATION_DIR.glob(f"{prefix}*.json"):
        try:
            path.unlink()
        except OSError:
            pass


def _write_group_calibration_files(group: str, cal_map: Dict[str, Dict]) -> None:
    if not isinstance(cal_map, dict):
        return
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    for paint_id, calibration in cal_map.items():
        pid = str(paint_id or "").strip()
        if not pid or not isinstance(calibration, dict):
            continue
        atomic_write(calibration_file_for(group, pid), calibration)


def build_library_export(group: str) -> Dict[str, Any]:
    """Build a portable JSON export of a paint library group."""
    library = load_library(group)
    library = dict(library)
    library["calibration_data"] = _merge_calibration_data_from_files(group, library)
    library["recipes"] = load_recipe_cache(group)
    library["group"] = group
    return {
        "export_format": "layerpainter-paint-library",
        "export_version": 1,
        "exported_at": datetime.now().isoformat(),
        "library": library,
        "feedback_bias": load_feedback_bias(group),
    }


def _parse_library_import_payload(payload: Any) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("Import file must be a JSON object")
    feedback_bias = payload.get("feedback_bias")
    if payload.get("export_format") == "layerpainter-paint-library":
        library = payload.get("library")
        if not isinstance(library, dict):
            raise ValueError("Export file is missing the library object")
        return library, feedback_bias if isinstance(feedback_bias, dict) else None
    if isinstance(payload.get("paints"), list):
        return payload, feedback_bias if isinstance(feedback_bias, dict) else None
    raise ValueError("Unrecognized paint library JSON format")


def import_library_data(
    payload: Any,
    *,
    target_group: Optional[str] = None,
    create_new: bool = False,
) -> Dict[str, Any]:
    """Import paints, calibrations, recipes, and settings from a JSON export."""
    library, feedback_bias = _parse_library_import_payload(payload)
    coerced = _coerce_library_shape(library, target_group or str(library.get("group") or "default"))

    if create_new:
        name = str(library.get("name") or library.get("group") or "Imported Library").strip()
        group_id = slugify(name) or "imported-library"
        base_id = group_id
        suffix = 2
        existing = set(list_library_groups())
        while group_id in existing:
            group_id = f"{base_id}-{suffix}"
            suffix += 1
        coerced["name"] = name
    else:
        group_id = (target_group or "").strip()
        if not group_id:
            raise ValueError("target_group is required when not creating a new library")
        if group_id not in list_library_groups():
            raise ValueError(f"Library group '{group_id}' not found")
        if library.get("name"):
            coerced["name"] = library.get("name")

    coerced["group"] = group_id
    _clear_group_calibration_files(group_id)
    cal_map = coerced.get("calibration_data")
    if isinstance(cal_map, dict):
        _write_group_calibration_files(group_id, cal_map)
    save_library(coerced, group_id)
    if isinstance(feedback_bias, dict) and feedback_bias:
        save_feedback_bias(group_id, {
            pid: v for pid, v in feedback_bias.items()
            if isinstance(v, (list, tuple)) and len(v) >= 3
        })
    return get_library_info(group_id)


def rgb_to_lab(rgb: List[float]) -> List[float]:
    """Convert RGB (0..255) to CIELAB (L* 0..100, a*/b* approx -128..127)."""
    rgb_clamped = [max(0.0, min(255.0, float(c))) for c in rgb]
    rgb_array = np.array([[rgb_clamped]], dtype=np.float32) / 255.0
    lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    return [float(v) for v in lab_array[0, 0].tolist()]


def lab_to_rgb(lab: List[float]) -> List[float]:
    """Convert CIELAB (L* 0..100, a*/b*) to RGB (0..255)."""
    lab_array = np.array([[[
        float(lab[0]),
        float(lab[1]),
        float(lab[2]),
    ]]], dtype=np.float32)
    rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)
    rgb_255 = np.clip(np.round(rgb_array[0, 0] * 255.0), 0, 255).astype(np.uint8)
    return rgb_255.tolist()


def _opencv_u8_lab_to_cielab(lab: List[float]) -> List[float]:
    """Convert OpenCV uint8-style Lab encoding to CIELAB."""
    l, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    return [
        (l * 100.0) / 255.0,
        a - 128.0,
        b - 128.0,
    ]


def _coerce_lab_to_cielab(lab: List[float]) -> List[float]:
    """Best-effort conversion of possibly-legacy Lab values to CIELAB."""
    if len(lab) < 3:
        return [0.0, 0.0, 0.0]
    l, a, b = float(lab[0]), float(lab[1]), float(lab[2])

    # Already plausible CIELAB range.
    if 0.0 <= l <= 100.0 and -128.0 <= a <= 127.0 and -128.0 <= b <= 127.0:
        return [l, a, b]

    # Otherwise, treat as OpenCV uint8-style Lab encoding.
    return _opencv_u8_lab_to_cielab([l, a, b])


def delta_e_lab(lab1: List[float], lab2: List[float]) -> float:
    """Calculate Euclidean distance in Lab space (simple ΔE)."""
    c1 = _coerce_lab_to_cielab(lab1)
    c2 = _coerce_lab_to_cielab(lab2)
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def sample_color_from_image(image_path: str, x: int, y: int, radius: int = 5) -> Tuple[List[int], List[float]]:
    """Sample color from image at given coordinates (average over small area)."""
    from image_processor import apply_exif_orientation
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Apply EXIF orientation if present
    image = apply_exif_orientation(image, image_path)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Clamp coordinates
    x = max(radius, min(w - radius, x))
    y = max(radius, min(h - radius, y))
    
    # Sample small region
    region = image[y-radius:y+radius, x-radius:x+radius]
    rgb_mean = np.mean(region.reshape(-1, 3), axis=0).astype(int).tolist()
    
    # Convert to Lab
    lab = rgb_to_lab([float(c) for c in rgb_mean])
    
    return rgb_mean, lab


def get_hex_from_calibration(paint_id: str, group: str = "default") -> Optional[str]:
    """Get the approximate hex color from a paint's calibration 100% swatch.
    Returns None if no calibration or no valid sample.
    """
    try:
        cal = _load_calibration_cached(paint_id, group)
        if cal is None:
            return None
        samples = cal.get('samples', [])
        if not samples:
            return None
        # Use the sample with the highest ratio (100% = pure paint)
        best = max(samples, key=lambda s: s.get('ratio', 0))
        rgb = best.get('rgb')
        if not rgb or len(rgb) < 3:
            return None
        r, g, b = max(0, min(255, int(rgb[0]))), max(0, min(255, int(rgb[1]))), max(0, min(255, int(rgb[2])))
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    except Exception:
        return None


def migrate_global_calibrations_to_group_scope(delete_legacy: bool = True) -> Dict[str, int]:
    """One-time migration: copy legacy calibration files into group-scoped files and optionally delete legacy files."""
    migrated_files = 0
    updated_library_entries = 0
    groups_seen = 0

    for group in list_library_groups():
        groups_seen += 1
        lib = load_library(group)
        paints = [p for p in lib.get("paints", []) if isinstance(p, dict)]
        cal_map = lib.get("calibration_data")
        if not isinstance(cal_map, dict):
            cal_map = {}
        changed = False

        for paint in paints:
            pid = str(paint.get("id", "")).strip()
            if not pid:
                continue
            scoped_file = calibration_file_for(group, pid)
            calibration = None

            if scoped_file.exists():
                try:
                    with open(scoped_file, "r") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        calibration = loaded
                except Exception:
                    calibration = None
            elif isinstance(cal_map.get(pid), dict):
                calibration = cal_map.get(pid)
                try:
                    atomic_write(scoped_file, calibration)
                    migrated_files += 1
                except Exception:
                    pass
            else:
                legacy_file = CALIBRATION_DIR / f"{pid}.json"
                if legacy_file.exists():
                    try:
                        with open(legacy_file, "r") as f:
                            loaded = json.load(f)
                        if isinstance(loaded, dict):
                            calibration = loaded
                            atomic_write(scoped_file, loaded)
                            migrated_files += 1
                    except Exception:
                        calibration = None

            if isinstance(calibration, dict):
                if cal_map.get(pid) != calibration:
                    cal_map[pid] = calibration
                    updated_library_entries += 1
                    changed = True

        if changed:
            lib["calibration_data"] = cal_map
            save_library(lib, group)

    deleted_legacy_files = 0
    if delete_legacy:
        for file in CALIBRATION_DIR.glob("*.json"):
            stem = file.stem
            # scoped files look like "<group>__<paint_id>"
            if "__" in stem:
                continue
            try:
                file.unlink()
                deleted_legacy_files += 1
            except Exception:
                continue

    return {
        "groups_seen": groups_seen,
        "migrated_files": migrated_files,
        "updated_library_entries": updated_library_entries,
        "deleted_legacy_files": deleted_legacy_files,
    }


def sample_color_from_region(
    image_path: str,
    x1: int, y1: int, x2: int, y2: int
) -> Tuple[List[int], List[float]]:
    """Sample average color from a rectangular region (user-selected swatch area).
    
    Coordinates are normalized to top-left (x1,y1) and bottom-right (x2,y2);
    order does not matter. Region is clamped to image bounds. All pixels in the
    region are averaged, then converted to Lab.
    """
    from image_processor import apply_exif_orientation
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = apply_exif_orientation(image, image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    x_min = max(0, min(x1, x2))
    x_max = min(w, max(x1, x2))
    y_min = max(0, min(y1, y2))
    y_max = min(h, max(y1, y2))
    
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Region has no area")
    
    crop = image[y_min:y_max, x_min:x_max]
    rgb_mean = np.mean(crop.reshape(-1, 3), axis=0).astype(int).tolist()
    lab = rgb_to_lab([float(c) for c in rgb_mean])
    return rgb_mean, lab


def normalize_calibration_samples(
    samples: List[Dict],
    reference_strip: Dict[str, Dict]
) -> List[Dict]:
    """Correct paint sample Lab values using the white / mid-grey / black reference strip.

    Uses the reference strip to correct for lighting and camera response so that
    the stored calibration is consistent across photos. Maps the measured L range
    to a standard 0–50–100 scale and removes a global colour cast using mid-grey.

    Args:
        samples: List of {ratio, rgb, lab} from paint swatches (lab will be modified).
        reference_strip: Dict with reference_white, reference_mid_grey, reference_black
            each with {"rgb": [...], "lab": [L, a, b]}.

    Returns:
        New list of samples with same structure but lab values normalized. RGB unchanged.
    """
    ref_w_raw = reference_strip.get("reference_white", {}).get("lab")
    ref_m_raw = reference_strip.get("reference_mid_grey", {}).get("lab")
    ref_b_raw = reference_strip.get("reference_black", {}).get("lab")
    ref_w = _coerce_lab_to_cielab(ref_w_raw) if ref_w_raw else None
    ref_m = _coerce_lab_to_cielab(ref_m_raw) if ref_m_raw else None
    ref_b = _coerce_lab_to_cielab(ref_b_raw) if ref_b_raw else None
    if not ref_w or not ref_m or not ref_b:
        return samples

    L_w, a_m, b_m = ref_w[0], ref_m[1], ref_m[2]
    L_m, L_b = ref_m[0], ref_b[0]

    # Avoid degenerate cases (wrong order or same values)
    if L_m <= L_b or L_w <= L_m:
        return samples
    denom_low = L_m - L_b
    denom_high = L_w - L_m
    if denom_low <= 0 or denom_high <= 0:
        return samples

    out = []
    for s in samples:
        l_raw = s.get("lab", [0.0, 0.0, 0.0])
        L, a, b = _coerce_lab_to_cielab(l_raw)
        # Piecewise linear L: map [L_b,L_m] -> [0,50], [L_m,L_w] -> [50,100]
        if L <= L_m:
            L_corr = 50.0 * (L - L_b) / denom_low
        else:
            L_corr = 50.0 + 50.0 * (L - L_m) / denom_high
        L_corr = max(0.0, min(100.0, L_corr))
        # Remove global cast by shifting a,b so mid-grey is neutral
        a_corr = a - a_m
        b_corr = b - b_m
        out.append({
            "ratio": s["ratio"],
            "rgb": s["rgb"],
            "lab": [L_corr, a_corr, b_corr],
        })
    return out


def _lab_L_to_Y(L: float) -> float:
    """CIELAB L* to relative luminance Y (0..1)."""
    L = float(L)
    if L > 8.0:
        fy = (L + 16.0) / 116.0
        return float(fy ** 3)
    return L / 903.3


def _lab_L_to_Y_batch(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    out = np.empty_like(L)
    hi = L > 8.0
    out[hi] = ((L[hi] + 16.0) / 116.0) ** 3
    out[~hi] = L[~hi] / 903.3
    return np.clip(out, 1e-6, 1.0)


def _lab_Y_to_L(Y: float) -> float:
    Y = max(1e-6, min(1.0 - 1e-9, float(Y)))
    if Y > 0.008856:
        return 116.0 * (Y ** (1.0 / 3.0)) - 16.0
    return 903.3 * Y


def _lab_Y_to_L_batch(Y: np.ndarray) -> np.ndarray:
    Y = np.clip(np.asarray(Y, dtype=np.float64), 1e-6, 1.0 - 1e-9)
    out = np.empty_like(Y)
    hi = Y > 0.008856
    out[hi] = 116.0 * np.power(Y[hi], 1.0 / 3.0) - 16.0
    out[~hi] = 903.3 * Y[~hi]
    return np.clip(out, 0.0, 100.0)


def _monotonic_calibration_labs(calibration: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ratios asc, labs) with L* non-increasing as pigment ratio increases."""
    samples = calibration.get("samples", [])
    if not samples:
        return np.array([], dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    sorted_samples = sorted(samples, key=lambda x: float(x["ratio"]))
    ratios = np.array([float(s["ratio"]) for s in sorted_samples], dtype=np.float64)
    labs = np.array([_coerce_lab_to_cielab(s["lab"]) for s in sorted_samples], dtype=np.float64)
    # Higher pigment ratio must not be lighter than a lower ratio (fix bad swatches).
    for i in range(1, len(labs)):
        if labs[i, 0] > labs[i - 1, 0]:
            labs[i, 0] = labs[i - 1, 0]
    return ratios, labs


def _interpolate_lab_from_monotonic_arrays(
    ratios: np.ndarray,
    labs: np.ndarray,
    ratio: float,
) -> Optional[List[float]]:
    if ratios.size == 0:
        return None
    ratio = float(ratio)
    if ratio <= ratios[0]:
        return list(labs[0])
    if ratio >= ratios[-1]:
        return list(labs[-1])
    idx = int(np.searchsorted(ratios, ratio, side="right") - 1)
    idx = max(0, min(idx, len(ratios) - 2))
    r0, r1 = ratios[idx], ratios[idx + 1]
    t = (ratio - r0) / (r1 - r0 + 1e-12)
    t = max(0.0, min(1.0, t))
    lab = (1.0 - t) * labs[idx] + t * labs[idx + 1]
    return [float(lab[0]), float(lab[1]), float(lab[2])]


def invert_calibration_ratio_for_L(
    calibration: Dict,
    target_L: float,
    group: Optional[str] = None,
    paint_id: Optional[str] = None,
) -> Optional[float]:
    """Invert monotonic L*(ratio) curve to find pigment/(pigment+white) for target lightness."""
    ratios, labs = _monotonic_calibration_labs(calibration)
    if ratios.size == 0:
        return None
    target_L = float(target_L)
    Ls = labs[:, 0]
    if target_L >= Ls[0]:
        return float(ratios[0])
    if target_L <= Ls[-1]:
        return float(ratios[-1])
    for i in range(len(ratios) - 1):
        L_hi, L_lo = Ls[i], Ls[i + 1]
        if L_lo <= target_L <= L_hi:
            span = L_hi - L_lo
            if abs(span) < 1e-6:
                return float(ratios[i + 1])
            t = (target_L - L_hi) / (L_lo - L_hi + 1e-12)
            t = max(0.0, min(1.0, t))
            return float(ratios[i] + t * (ratios[i + 1] - ratios[i]))
    return None


def _uncalibrated_tint_lab(hex_rgb: List[int], eff_ratio: float) -> List[float]:
    """Subtractive tint of a full-strength paint swatch toward white at eff_ratio."""
    eff_ratio = max(0.0, min(1.0, float(eff_ratio)))
    paint_lab = np.array(rgb_to_lab([float(c) for c in hex_rgb]), dtype=np.float64)
    white_lab = np.array([100.0, 0.0, 0.0], dtype=np.float64)
    if eff_ratio <= 1e-9:
        return [100.0, 0.0, 0.0]
    if eff_ratio >= 1.0 - 1e-9:
        return paint_lab.tolist()
    labs = np.stack([paint_lab, white_lab], axis=0)
    weights = np.array([eff_ratio, 1.0 - eff_ratio], dtype=np.float64)
    return _combine_labs_subtractive(labs, weights)


def _combine_labs_subtractive(labs: np.ndarray, weights: np.ndarray) -> List[float]:
    """Mix calibrated swatch Labs by mass fraction using subtractive Y (1/sum w/Y)."""
    weights = np.asarray(weights, dtype=np.float64)
    labs = np.asarray(labs, dtype=np.float64)
    total = float(weights.sum())
    if total <= 1e-12 or labs.shape[0] == 0:
        return [100.0, 0.0, 0.0]
    w = weights / total
    Y = _lab_L_to_Y_batch(labs[:, 0])
    Y_mix = 1.0 / np.sum(w / Y)
    L_mix = float(_lab_Y_to_L_batch(np.array([Y_mix]))[0])
    a_mix = float(np.sum(w * labs[:, 1]))
    b_mix = float(np.sum(w * labs[:, 2]))
    return [L_mix, a_mix, b_mix]


def _combine_labs_subtractive_batch(labs_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """labs_stack (N, K, 3), weights (N, K) -> (N, 3) CIELAB."""
    N, K, _ = labs_stack.shape
    out = np.zeros((N, 3), dtype=np.float64)
    w = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    Y = _lab_L_to_Y_batch(labs_stack[:, :, 0].reshape(-1)).reshape(N, K)
    inv = np.sum(w / np.maximum(Y, 1e-6), axis=1)
    Y_mix = 1.0 / np.maximum(inv, 1e-6)
    out[:, 0] = _lab_Y_to_L_batch(Y_mix)
    out[:, 1] = np.sum(w * labs_stack[:, :, 1], axis=1)
    out[:, 2] = np.sum(w * labs_stack[:, :, 2], axis=1)
    return out


_VALUE_PAINT_IDS = frozenset({
    "burnt-umber",
    "australian-olive-green",
    "black",
    "carbon-black",
})


def _is_green_value_paint(paint_id: str) -> bool:
    pid = (paint_id or "").strip().lower()
    return "olive" in pid and "green" in pid


def _is_warm_hue_target(lab: List[float]) -> bool:
    """True for orange/brown/red-yellow targets (not green/blue neutrals)."""
    a, b = float(lab[1]), float(lab[2])
    chroma = (a * a + b * b) ** 0.5
    if chroma < 12.0:
        return False
    hue_deg = float(np.degrees(np.arctan2(b, a)) % 360.0)
    return 15.0 <= hue_deg <= 85.0


def _combo_allowed_for_target(target_lab: List[float], paint_ids: List[str]) -> bool:
    """Reject pigment sets that fight the target hue (e.g. olive in a brown)."""
    ids = [(pid or "").strip().lower() for pid in paint_ids]
    if _is_warm_hue_target(target_lab) and any(_is_green_value_paint(pid) for pid in ids):
        return False
    n_value = sum(1 for pid in ids if pid in _VALUE_PAINT_IDS)
    if n_value > 1:
        return False
    return True


def _hue_penalty_coeffs_per_paint(
    target_lab: List[float],
    paint_ids: List[str],
    library_group: str,
) -> np.ndarray:
    """Per-paint penalty weight per unit mass fraction (computed once per combo)."""
    n = len(paint_ids)
    coeffs = np.zeros(n, dtype=np.float64)
    target = _coerce_lab_to_cielab(target_lab)
    ta, tb = float(target[1]), float(target[2])
    target_chroma = (ta * ta + tb * tb) ** 0.5
    if target_chroma < 8.0:
        return coeffs
    target_h = float(np.arctan2(tb, ta))
    warm = _is_warm_hue_target(target)
    for i, pid in enumerate(paint_ids):
        if warm and _is_green_value_paint(pid):
            coeffs[i] = 18.0
            continue
        cal = _load_calibration_cached(pid, library_group)
        if cal is None:
            continue
        lab_full = interpolate_lab_from_calibration(
            cal, 1.0, group=library_group, paint_id=pid
        )
        if lab_full is None:
            continue
        pa, pb = float(lab_full[1]), float(lab_full[2])
        paint_chroma = (pa * pa + pb * pb) ** 0.5
        if paint_chroma < 6.0:
            continue
        paint_h = float(np.arctan2(pb, pa))
        dh = abs(target_h - paint_h)
        dh = min(dh, 2.0 * np.pi - dh)
        if dh > np.radians(55.0):
            coeffs[i] = (dh - np.radians(55.0)) * 40.0
    return coeffs


def _hue_mismatch_penalty_from_coeffs(
    pigment_ratios: List[float],
    coeffs: np.ndarray,
    min_ratio: float = 0.06,
) -> float:
    """Fast hue penalty using precomputed per-paint coeffs."""
    if coeffs.size == 0:
        return 0.0
    r = np.asarray(pigment_ratios, dtype=np.float64)
    mask = r >= min_ratio
    if not np.any(mask):
        return 0.0
    return float(np.dot(coeffs[mask], r[mask]))


def _l_star_seed_ratios(
    target_lab: List[float],
    paint_calibrations: List[Optional[Dict]],
    n_pigments: int,
    min_ratio: float,
    max_total_pigment: float,
    max_ratio_per_pigment: float,
    library_group: str,
    paint_ids: List[str],
) -> Optional[List[float]]:
    """Seed multi-pigment search from L* inversion on the darkest calibrated paint in the set."""
    target_L = float(_coerce_lab_to_cielab(target_lab)[0])
    dark_idx: Optional[int] = None
    dark_L_full = 101.0
    for i, cal in enumerate(paint_calibrations):
        if cal is None:
            continue
        pid = paint_ids[i] if i < len(paint_ids) else ""
        if _is_warm_hue_target(target_lab) and _is_green_value_paint(pid):
            continue
        _, labs = _monotonic_calibration_labs(cal)
        if labs.shape[0] == 0:
            continue
        L_full = float(labs[-1, 0])
        if L_full < dark_L_full:
            dark_L_full = L_full
            dark_idx = i
    if dark_idx is None:
        return None
    cal = paint_calibrations[dark_idx]
    if cal is None:
        return None
    pid = paint_ids[dark_idx] if dark_idx < len(paint_ids) else None
    ratio_dark = invert_calibration_ratio_for_L(cal, target_L, group=library_group, paint_id=pid)
    if ratio_dark is None:
        return None
    seed = [float(min_ratio)] * n_pigments
    seed[dark_idx] = float(np.clip(ratio_dark, min_ratio, max_ratio_per_pigment))
    total = sum(seed)
    if total > max_total_pigment:
        scale = max_total_pigment / total
        seed = [max(min_ratio, r * scale) for r in seed]
    return seed


def interpolate_lab_from_calibration(
    calibration: Dict,
    ratio: float,
    group: Optional[str] = None,
    paint_id: Optional[str] = None,
) -> Optional[List[float]]:
    """Interpolate Lab color for a given ratio from calibration samples.
    If group and paint_id are provided, adds feedback bias (learned from spot tests).
    Uses monotonic L* vs ratio so bad swatches do not invert the curve.
    """
    ratios, labs = _monotonic_calibration_labs(calibration)
    if ratios.size == 0:
        return None
    lab = _interpolate_lab_from_monotonic_arrays(ratios, labs, ratio)
    if lab is None:
        return None
    if group and paint_id:
        bias = get_paint_bias(group, paint_id)
        if bias:
            lab = [lab[0] + bias[0], lab[1] + bias[1], lab[2] + bias[2]]
    return lab


def get_calibration_max_ratio(calibration: Dict) -> float:
    """Return the maximum calibrated pigment ratio available for a paint."""
    samples = calibration.get('samples', [])
    ratios = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        try:
            ratios.append(float(s.get('ratio', 0)))
        except Exception:
            continue
    if not ratios:
        return 0.0
    return max(ratios)


def _interpolate_lab_from_calibration_batch(
    calibration: Optional[Dict],
    ratios: np.ndarray,
    group: Optional[str] = None,
    paint_id: Optional[str] = None,
) -> Optional[np.ndarray]:
    """(N,) ratios -> (N, 3) Lab in CIELAB. Returns None if calibration is None or empty."""
    if calibration is None:
        return None
    ratios_sorted, labs_sorted = _monotonic_calibration_labs(calibration)
    if ratios_sorted.size == 0:
        return None
    ratios_flat = np.asarray(ratios, dtype=np.float64).ravel()
    n = ratios_flat.size
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx_low = np.searchsorted(ratios_sorted, ratios_flat, side="right") - 1
    idx_low = np.clip(idx_low, 0, len(ratios_sorted) - 2)
    idx_high = idx_low + 1
    r_low = ratios_sorted[idx_low]
    r_high = ratios_sorted[idx_high]
    t = (ratios_flat - r_low) / (r_high - r_low + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    lab = (1.0 - t)[:, None] * labs_sorted[idx_low] + t[:, None] * labs_sorted[idx_high]
    if group and paint_id:
        bias = get_paint_bias(group, paint_id)
        if bias:
            lab = lab + np.array(bias, dtype=np.float64)
    return lab


def _rgb_to_linear(rgb: List[float]) -> np.ndarray:
    arr = np.array([max(0.0, min(255.0, float(c))) / 255.0 for c in rgb], dtype=np.float64)
    return np.power(arr, 2.2)


def _linear_to_rgb(linear_rgb: np.ndarray) -> List[float]:
    clamped = np.clip(np.asarray(linear_rgb, dtype=np.float64), 0.0, 1.0)
    srgb = np.power(clamped, 1.0 / 2.2) * 255.0
    return [float(v) for v in np.clip(np.round(srgb), 0, 255)]


# ----- Batch variants for multi-pigment search (reduce Python/call overhead) -----

def _rgb_to_linear_batch(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) RGB 0..255 -> (N, 3) linear."""
    arr = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 255.0) / 255.0
    return np.power(arr, 2.2)


def _linear_to_rgb_batch(linear_rgb: np.ndarray) -> np.ndarray:
    """(N, 3) linear -> (N, 3) RGB 0..255."""
    clamped = np.clip(np.asarray(linear_rgb, dtype=np.float64), 0.0, 1.0)
    srgb = np.power(clamped, 1.0 / 2.2) * 255.0
    return np.clip(np.round(srgb), 0, 255).astype(np.float64)


def _opencv_lab_batch_to_cielab(lab: np.ndarray) -> np.ndarray:
    """(N, 3) OpenCV Lab (L 0..255, a/b 0..255) -> (N, 3) CIELAB."""
    lab = np.asarray(lab, dtype=np.float64)
    out = np.empty_like(lab)
    out[:, 0] = lab[:, 0] * 100.0 / 255.0
    out[:, 1] = lab[:, 1] - 128.0
    out[:, 2] = lab[:, 2] - 128.0
    return out


def _rgb_to_lab_batch(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) RGB 0..255 -> (N, 3) CIELAB. Uses cv2 in batch (uint8 for consistent LAB scale)."""
    n = rgb.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rgb_u8 = np.clip(np.round(np.asarray(rgb, dtype=np.float64)), 0, 255).astype(np.uint8)
    rgb_u8 = rgb_u8.reshape((n, 1, 3))
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    lab = lab.reshape((n, 3)).astype(np.float64)
    return _opencv_lab_batch_to_cielab(lab)


def _lab_to_rgb_batch(lab_cielab: np.ndarray) -> np.ndarray:
    """(N, 3) CIELAB -> (N, 3) RGB 0..255. Input in CIELAB (L 0..100, a/b -128..127)."""
    n = lab_cielab.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    lab = np.asarray(lab_cielab, dtype=np.float32)
    lab[:, 0] = lab[:, 0] * 255.0 / 100.0
    lab[:, 1] = lab[:, 1] + 128.0
    lab[:, 2] = lab[:, 2] + 128.0
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    lab = lab.reshape((n, 1, 3))
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb.reshape((n, 3)).astype(np.float64)


def get_substrate_compensation(library_group: Optional[str] = None) -> Dict[str, float]:
    """Return substrate (paper) compensation params for a library.

    Reads ``library["substrate_compensation"]`` if present and merges over module defaults.
    Cached on library file mtime so per-recipe lookups are cheap.
    """
    group = (library_group or "default").strip() or "default"
    library_file = LIBRARIES_DIR / f"{group}.json"
    if not library_file.exists() and group == "default" and LIBRARY_FILE.exists():
        library_file = LIBRARY_FILE
    try:
        mtime_ns = library_file.stat().st_mtime_ns if library_file.exists() else 0
    except OSError:
        mtime_ns = 0
    cached = _SUBSTRATE_PARAMS_CACHE.get(group)
    if cached and cached[0] == mtime_ns:
        return cached[1]

    merged = dict(_SUBSTRATE_DEFAULTS)
    try:
        if library_file.exists():
            with open(library_file, "r") as f:
                data = json.load(f)
            sc = data.get("substrate_compensation") if isinstance(data, dict) else None
            if isinstance(sc, dict):
                if "enabled" in sc:
                    merged["enabled"] = bool(sc.get("enabled"))
                for key in ("L_paper_min", "L_break", "alpha_dark"):
                    if key in sc:
                        try:
                            merged[key] = float(sc.get(key))
                        except (TypeError, ValueError):
                            pass
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    # Clamp to safe ranges so a bad config can't break the solver.
    merged["L_paper_min"] = max(0.0, min(40.0, float(merged["L_paper_min"])))
    merged["L_break"] = max(merged["L_paper_min"] + 1.0, min(80.0, float(merged["L_break"])))
    merged["alpha_dark"] = max(0.5, min(3.0, float(merged["alpha_dark"])))
    _SUBSTRATE_PARAMS_CACHE[group] = (mtime_ns, merged)
    return merged


def _apply_substrate_l_compensation(L: float, params: Dict[str, float]) -> float:
    """Apply paper black-point compensation to a single L* value (CIELAB).

    For L >= L_break the value is unchanged. For L < L_break, L is mapped from [0, L_break]
    onto [L_paper_min, L_break] using a power curve t**(1/alpha_dark); with alpha_dark>1
    the curve is concave so the deepest darks are lifted toward L_paper_min.
    """
    if not params.get("enabled", True):
        return L
    L_break = float(params["L_break"])
    L_paper_min = float(params["L_paper_min"])
    if L >= L_break:
        return L
    L = max(0.0, float(L))
    t = L / L_break  # in [0, 1]
    alpha = float(params["alpha_dark"])
    t_lifted = t ** (1.0 / alpha) if alpha > 0 else t
    return L_paper_min + (L_break - L_paper_min) * t_lifted


def _apply_substrate_l_compensation_batch(L_arr: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Vectorized version of :func:`_apply_substrate_l_compensation` over an array of L* values."""
    if not params.get("enabled", True):
        return L_arr
    L_break = float(params["L_break"])
    L_paper_min = float(params["L_paper_min"])
    alpha = float(params["alpha_dark"])
    L = np.asarray(L_arr, dtype=np.float64)
    out = L.copy()
    finite = np.isfinite(L)
    mask = finite & (L < L_break)
    if not np.any(mask):
        return out
    t = np.clip(L[mask] / L_break, 0.0, 1.0)
    t_lifted = np.power(t, 1.0 / alpha) if alpha > 0 else t
    out[mask] = L_paper_min + (L_break - L_paper_min) * t_lifted
    return out


def _predict_mix_lab_from_components(
    components: List[Tuple[Optional[Dict], Optional[List[int]], float]],
    white_ratio: float,
    library_group: Optional[str] = None,
    paint_ids: Optional[List[str]] = None,
) -> Optional[List[float]]:
    """Predict mixed Lab from calibration curves using total dilution and subtractive Y mixing.

    Each pigment is evaluated at eff = p_i / (sum_j p_j + white_ratio). Calibration swatches
    are already pigment-in-white at that ratio, so we do not add a separate white component.
    """
    white_ratio = max(0.0, min(1.0, float(white_ratio)))
    total_pigment = sum(max(0.0, float(r)) for _, _, r in components)
    denom = total_pigment + white_ratio
    if denom <= 1e-12:
        return [100.0, 0.0, 0.0]

    labs_list: List[List[float]] = []
    weights: List[float] = []

    for idx, (calibration, hex_rgb, pigment_ratio) in enumerate(components):
        p = max(0.0, float(pigment_ratio))
        if p <= 1e-12:
            continue
        eff_ratio = max(0.0, min(1.0, p / denom))
        pid = paint_ids[idx] if paint_ids and idx < len(paint_ids) else None
        lab: Optional[List[float]] = None
        if calibration is not None:
            lab = interpolate_lab_from_calibration(
                calibration, eff_ratio, group=library_group, paint_id=pid
            )
        elif hex_rgb is not None:
            lab = _uncalibrated_tint_lab(hex_rgb, eff_ratio)
        if lab is None:
            continue
        labs_list.append(lab)
        weights.append(p)

    if not labs_list:
        if white_ratio > 1e-9:
            return [100.0, 0.0, 0.0]
        return None

    mixed_lab = _combine_labs_subtractive(
        np.array(labs_list, dtype=np.float64),
        np.array(weights, dtype=np.float64),
    )
    if library_group is not None:
        params = get_substrate_compensation(library_group)
        mixed_lab[0] = _apply_substrate_l_compensation(mixed_lab[0], params)
    return mixed_lab


def _predict_mix_lab_batch(
    paint_calibrations: List[Optional[Dict]],
    paint_hex_colors: List[Optional[List[int]]],
    paint_max_ratios: List[float],
    ratios_batch: np.ndarray,
    min_white_ratio: float,
    max_total_pigment: float,
    library_group: Optional[str] = None,
    paint_ids: Optional[List[str]] = None,
) -> np.ndarray:
    """Batch mix prediction. ratios_batch (N, n_pigments). Returns (N, 3) CIELAB; invalid rows are NaN.
    If library_group and paint_ids are provided, applies feedback bias per paint.
    """
    ratios_batch = np.asarray(ratios_batch, dtype=np.float64)
    n, n_pigments = ratios_batch.shape
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    total_pigment = ratios_batch.sum(axis=1)
    white_ratio = 1.0 - total_pigment
    valid = (
        (white_ratio >= min_white_ratio)
        & (total_pigment <= max_total_pigment)
        & (total_pigment > 1e-12)
    )
    max_ratios_arr = np.array(paint_max_ratios, dtype=np.float64)
    valid &= (ratios_batch <= max_ratios_arr + 1e-9).all(axis=1)

    denom = total_pigment + white_ratio + 1e-12
    labs_stack = np.zeros((n, n_pigments, 3), dtype=np.float64)
    weights = np.zeros((n, n_pigments), dtype=np.float64)

    slot = 0
    for i in range(n_pigments):
        p = ratios_batch[:, i]
        eff_ratio = np.clip(p / denom, 0.0, 1.0)
        calibration = paint_calibrations[i] if i < len(paint_calibrations) else None
        hex_rgb = paint_hex_colors[i] if i < len(paint_hex_colors) else None
        pid = paint_ids[i] if paint_ids and i < len(paint_ids) else None
        lab_i: Optional[np.ndarray] = None
        if calibration is not None:
            lab_i = _interpolate_lab_from_calibration_batch(
                calibration, eff_ratio, group=library_group, paint_id=pid
            )
        elif hex_rgb is not None:
            lab_i = np.array(
                [_uncalibrated_tint_lab(hex_rgb, float(e)) for e in eff_ratio],
                dtype=np.float64,
            )
        if lab_i is None:
            continue
        labs_stack[:, slot, :] = lab_i
        weights[:, slot] = p
        slot += 1

    if slot == 0:
        mixed_lab = np.full((n, 3), np.nan, dtype=np.float64)
    else:
        mixed_lab = _combine_labs_subtractive_batch(labs_stack[:, :slot, :], weights[:, :slot])
    if library_group is not None:
        params = get_substrate_compensation(library_group)
        if params.get("enabled", True):
            mixed_lab[:, 0] = _apply_substrate_l_compensation_batch(mixed_lab[:, 0], params)
    mixed_lab[~valid] = np.nan
    return mixed_lab


def _delta_e_batch(lab_batch: np.ndarray, target_lab: List[float]) -> np.ndarray:
    """(N, 3) CIELAB batch vs target -> (N,) ΔE. NaN in lab_batch -> inf."""
    target = np.array(_coerce_lab_to_cielab(target_lab), dtype=np.float64)
    diff = lab_batch - target
    out = np.sqrt(np.nansum(diff * diff, axis=1))
    out[np.any(np.isnan(lab_batch), axis=1)] = np.inf
    return out


def find_best_one_pigment_recipe(
    target_lab: List[float],
    paint_id: str,
    paint_hex: str = None,
    library_group: str = "default",
) -> Optional[Dict]:
    """Find best mixing ratio for one pigment + white to match target Lab.
    
    Args:
        target_lab: Target color in Lab color space
        paint_id: Paint ID
        paint_hex: Optional hex color for uncalibrated paints (e.g., '#FF0000')
    
    Returns:
        Recipe dict with pigment_id, pigment_ratio, white_ratio, and error
    """
    min_white_ratio, max_total_pigment = get_white_mix_limits(target_lab, library_group)

    # If calibration exists, use it (more accurate)
    calibration = _load_calibration_cached(paint_id, library_group)
    if calibration is not None:
        samples = calibration.get('samples', [])
        if samples:
            cal_max_ratio = max(0.0, min(1.0, get_calibration_max_ratio(calibration)))
            if cal_max_ratio <= 0.0:
                return None
            # Search for best ratio
            best_ratio = None
            best_error = float('inf')
            
            # Test pigment ratios across target-appropriate range.
            upper_ratio = min(max_total_pigment, cal_max_ratio)
            for test_ratio in np.arange(0.0, upper_ratio + 0.001, 0.01):
                white_r = 1.0 - test_ratio
                predicted_lab = predict_mix_lab_for_paint_ratios(
                    library_group, [paint_id], [test_ratio], white_r
                )
                if predicted_lab:
                    error = delta_e_lab(target_lab, predicted_lab)
                    if error < best_error:
                        best_error = error
                        best_ratio = test_ratio
            
            if best_ratio is not None:
                result = {
                    'pigment_id': paint_id,
                    'pigment_ratio': best_ratio,
                    'white_ratio': 1.0 - best_ratio,
                    'error': best_error
                }
                return _apply_physical_value_caps_to_recipe(result, target_lab, library_group)

    # Fallback: Use approximate color if no calibration (less accurate but better than nothing)
    if paint_hex:
        try:
            # Convert hex to RGB
            hex_clean = paint_hex.lstrip('#')
            paint_rgb = [int(hex_clean[i:i+2], 16) for i in (0, 2, 4)]
            paint_lab = rgb_to_lab(paint_rgb)
            
            # Estimate mixing ratio based on lightness difference
            # If target is lighter than paint, need more white
            target_lightness = target_lab[0]  # L channel (0-100)
            paint_lightness = paint_lab[0]
            
            # Simple estimation: adjust ratio based on lightness difference
            # If target is much lighter, use less pigment
            lightness_diff = target_lightness - paint_lightness
            
            # Estimate pigment ratio over target-appropriate range.
            # If target is lighter, use less pigment
            if lightness_diff > 0:
                # Target is lighter - use less pigment
                estimated_ratio = max(0.02, min(max_total_pigment, max_total_pigment - (lightness_diff / 200.0)))
            else:
                # Target is darker - use more pigment
                estimated_ratio = max(0.05, min(max_total_pigment, max_total_pigment + (abs(lightness_diff) / 200.0)))
            
            # Calculate approximate error (will be higher than calibrated)
            # Use a simple distance metric
            estimated_lab = [
                target_lightness * (1 - estimated_ratio) + paint_lightness * estimated_ratio,
                target_lab[1] * (1 - estimated_ratio) + paint_lab[1] * estimated_ratio,
                target_lab[2] * (1 - estimated_ratio) + paint_lab[2] * estimated_ratio
            ]
            estimated_error = delta_e_lab(target_lab, estimated_lab)
            
            # Add penalty for uncalibrated (so calibrated paints are preferred)
            estimated_error += 3.0
            
            return {
                'pigment_id': paint_id,
                'pigment_ratio': estimated_ratio,
                'white_ratio': 1.0 - estimated_ratio,
                'error': estimated_error,
                'uncalibrated': True  # Flag to indicate this is an estimate
            }
        except Exception as e:
            # If hex conversion fails, return None
            return None
    
    return None


def find_best_two_pigment_recipe(
    target_lab: List[float],
    paint_id1: str,
    paint_id2: str,
    paint1_hex: str = None,
    paint2_hex: str = None,
    library_group: str = "default",
) -> Optional[Dict]:
    """Find best mixing ratio for two pigments + white (approximation).
    
    Args:
        target_lab: Target color in Lab color space
        paint_id1: First paint ID
        paint_id2: Second paint ID
        paint1_hex: Optional hex color for first paint (if uncalibrated)
        paint2_hex: Optional hex color for second paint (if uncalibrated)
    
    Returns:
        Recipe dict with pigment IDs, ratios, white_ratio, and error
    """
    min_white_ratio, max_total_pigment = get_white_mix_limits(target_lab, library_group)
    min_ratio = 0.02
    max_ratio_per_pigment = max_total_pigment - min_ratio
    
    # Try calibrated first if both exist (batched evaluation)
    cal1 = _load_calibration_cached(paint_id1, library_group)
    cal2 = _load_calibration_cached(paint_id2, library_group)
    if cal1 is not None and cal2 is not None:
        cal1_max = max(0.0, min(1.0, get_calibration_max_ratio(cal1)))
        cal2_max = max(0.0, min(1.0, get_calibration_max_ratio(cal2)))
        if cal1_max <= 0.0 or cal2_max <= 0.0:
            return None

        cap1 = _value_paint_ratio_cap(paint_id1, cal1, target_lab, library_group)
        cap2 = _value_paint_ratio_cap(paint_id2, cal2, target_lab, library_group)
        p1_hi = min(max_ratio_per_pigment, cal1_max, cap1 if cap1 is not None else 1.0)
        p2_hi = min(max_ratio_per_pigment, cal2_max, cap2 if cap2 is not None else 1.0)
        p1_axis = np.arange(min_ratio, p1_hi + 0.001, 0.02)
        p2_axis = np.arange(min_ratio, p2_hi + 0.001, 0.02)
        p1_grid, p2_grid = np.meshgrid(p1_axis, p2_axis, indexing='ij')
        p1_flat = p1_grid.ravel()
        p2_flat = p2_grid.ravel()
        valid = (p1_flat + p2_flat <= max_total_pigment) & (1.0 - p1_flat - p2_flat >= min_white_ratio)
        p1_flat = p1_flat[valid]
        p2_flat = p2_flat[valid]
        if p1_flat.size == 0:
            return None
        ratios_batch = np.column_stack([p1_flat, p2_flat])
        lab_batch = _predict_mix_lab_batch(
            [cal1, cal2],
            [None, None],
            [cal1_max, cal2_max],
            ratios_batch,
            min_white_ratio,
            max_total_pigment,
            library_group=library_group,
            paint_ids=[paint_id1, paint_id2],
        )
        errors = _delta_e_batch(lab_batch, target_lab)
        hue2 = _hue_penalty_coeffs_per_paint(
            target_lab, [paint_id1, paint_id2], library_group
        )
        if np.any(hue2 > 0):
            errors += p1_flat * hue2[0] + p2_flat * hue2[1]
        idx = np.argmin(errors)
        best_error = float(errors[idx])
        if not np.isfinite(best_error):
            return None
        p1_best = float(p1_flat[idx])
        p2_best = float(p2_flat[idx])
        result = {
            'pigment1_id': paint_id1,
            'pigment1_ratio': p1_best,
            'pigment2_id': paint_id2,
            'pigment2_ratio': p2_best,
            'white_ratio': 1.0 - p1_best - p2_best,
            'error': best_error,
            'type': 'two_pigment'
        }
        return _apply_physical_value_caps_to_recipe(result, target_lab, library_group)
    
    # Fallback: Use approximate colors if available (batched)
    if paint1_hex and paint2_hex:
        try:
            hex1_clean = paint1_hex.lstrip('#')
            rgb1 = [int(hex1_clean[i:i+2], 16) for i in (0, 2, 4)]
            hex2_clean = paint2_hex.lstrip('#')
            rgb2 = [int(hex2_clean[i:i+2], 16) for i in (0, 2, 4)]

            p1_axis = np.arange(min_ratio, max_ratio_per_pigment + 0.001, 0.02)
            p2_axis = np.arange(min_ratio, max_ratio_per_pigment + 0.001, 0.02)
            p1_grid, p2_grid = np.meshgrid(p1_axis, p2_axis, indexing='ij')
            p1_flat = p1_grid.ravel()
            p2_flat = p2_grid.ravel()
            valid = (p1_flat + p2_flat <= max_total_pigment) & (1.0 - p1_flat - p2_flat >= min_white_ratio)
            p1_flat = p1_flat[valid]
            p2_flat = p2_flat[valid]
            if p1_flat.size == 0:
                return None
            ratios_batch = np.column_stack([p1_flat, p2_flat])
            lab_batch = _predict_mix_lab_batch(
                [None, None],
                [rgb1, rgb2],
                [1.0, 1.0],
                ratios_batch,
                min_white_ratio,
                max_total_pigment,
            )
            errors = _delta_e_batch(lab_batch, target_lab) + 5.0
            idx = np.argmin(errors)
            best_error = float(errors[idx])
            if not np.isfinite(best_error):
                return None
            p1_best = float(p1_flat[idx])
            p2_best = float(p2_flat[idx])
            return {
                'pigment1_id': paint_id1,
                'pigment1_ratio': p1_best,
                'pigment2_id': paint_id2,
                'pigment2_ratio': p2_best,
                'white_ratio': 1.0 - p1_best - p2_best,
                'error': best_error,
                'type': 'two_pigment',
                'uncalibrated': True
            }
        except Exception:
            return None

    return None


def find_best_multi_pigment_recipe(
    target_lab: List[float],
    paint_ids: List[str],
    paint_hexes: List[str],
    library_group: str = "default",
    quality_mode: str = "balanced",
) -> Optional[Dict]:
    """Find best mixing ratio for multiple pigments + white.
    
    Args:
        target_lab: Target color in Lab color space
        paint_ids: List of paint IDs
        paint_hexes: List of hex colors for paints (if uncalibrated)
    
    Returns:
        Recipe dict with pigment IDs, ratios, white_ratio, and error
    """
    n_pigments = len(paint_ids)
    if n_pigments < 3:
        return None
    
    # Store calibration data and hex colors for each paint
    # We'll interpolate calibrated colors at the actual ratio being tested
    paint_calibrations = []
    paint_hex_colors = []
    paint_max_ratios = []
    calibrated_count = 0
    
    for paint_id, paint_hex in zip(paint_ids, paint_hexes):
        calibration = _load_calibration_cached(paint_id, library_group)
        if calibration is not None:
            calibrated_count += 1
        
        paint_calibrations.append(calibration)
        if calibration is not None:
            paint_max_ratios.append(max(0.0, min(1.0, get_calibration_max_ratio(calibration))))
        else:
            paint_max_ratios.append(1.0)
        
        # Store hex color as fallback
        if paint_hex:
            try:
                hex_clean = paint_hex.lstrip('#')
                rgb = [int(hex_clean[i:i+2], 16) for i in (0, 2, 4)]
                paint_hex_colors.append(rgb)
            except:
                return None
        else:
            paint_hex_colors.append(None)
    
    if len(paint_calibrations) != n_pigments:
        return None
    
    # Two-stage search with unbiased ratio enumeration.
    min_ratio = 0.02
    min_white_ratio, max_total_pigment = get_white_mix_limits(target_lab, library_group)
    max_ratio_per_pigment = max_total_pigment - (n_pigments - 1) * min_ratio
    if max_ratio_per_pigment < min_ratio:
        return None

    value_caps = [
        _value_paint_ratio_cap(paint_ids[i], paint_calibrations[i], target_lab, library_group)
        for i in range(n_pigments)
    ]

    ratio_type = {
        3: 'three_pigment',
        4: 'four_pigment',
    }.get(n_pigments, f'{n_pigments}_pigment')

    uncalibrated_penalty = (n_pigments - calibrated_count) * (3.0 if n_pigments == 3 else 4.0)
    _RECIPE_BATCH_SIZE = 2048
    hue_coeffs = _hue_penalty_coeffs_per_paint(target_lab, paint_ids, library_group)
    umber_cap = None
    umber_idx = None
    for i, pid in enumerate(paint_ids):
        if (pid or "").strip().lower() == "burnt-umber":
            umber_idx = i
            umber_cap = value_caps[i]
            break

    def evaluate_batch(ratios_list: List[List[float]]) -> np.ndarray:
        """List of N ratio vectors -> (N,) errors; inf for invalid."""
        if not ratios_list:
            return np.array([], dtype=np.float64)
        ratios_arr = np.array(ratios_list, dtype=np.float64)
        lab_batch = _predict_mix_lab_batch(
            paint_calibrations,
            paint_hex_colors,
            paint_max_ratios,
            ratios_arr,
            min_white_ratio,
            max_total_pigment,
            library_group=library_group,
            paint_ids=paint_ids,
        )
        errors = _delta_e_batch(lab_batch, target_lab)
        errors += uncalibrated_penalty
        if hue_coeffs.size == ratios_arr.shape[1] and np.any(hue_coeffs > 0):
            with np.errstate(over="ignore", invalid="ignore"):
                errors += np.nan_to_num(ratios_arr @ hue_coeffs, nan=0.0, posinf=0.0, neginf=0.0)
        if umber_idx is not None and umber_cap is not None:
            excess = np.maximum(0.0, ratios_arr[:, umber_idx] - float(umber_cap))
            errors += excess * 25.0
        return errors

    def search(step: float, center: Optional[List[float]] = None, radius: float = 0.0) -> Tuple[Optional[List[float]], float]:
        ratio_axes = []
        for i in range(n_pigments):
            low = min_ratio
            high = max_ratio_per_pigment
            if value_caps[i] is not None:
                high = min(high, value_caps[i])
            if center is not None:
                low = max(min_ratio, center[i] - radius)
                high = min(max_ratio_per_pigment, center[i] + radius)
                if value_caps[i] is not None:
                    high = min(high, value_caps[i])
            axis = np.arange(low, high + (step * 0.5), step)
            if len(axis) == 0:
                return None, float('inf')
            ratio_axes.append(axis)

        best_ratios = None
        best_error = float('inf')
        batch_list: List[List[float]] = []

        def flush_batch() -> None:
            nonlocal best_ratios, best_error
            if not batch_list:
                return
            errors = evaluate_batch(batch_list)
            for k, err in enumerate(errors):
                if err < best_error:
                    best_error = float(err)
                    best_ratios = batch_list[k].copy()
            batch_list.clear()

        def walk(depth: int, chosen: List[float], total: float) -> None:
            nonlocal best_ratios, best_error
            remaining = n_pigments - depth
            if depth == n_pigments:
                batch_list.append(chosen.copy())
                if len(batch_list) >= _RECIPE_BATCH_SIZE:
                    flush_batch()
                return

            min_needed_for_rest = (remaining - 1) * min_ratio
            max_this = min(
                ratio_axes[depth][-1],
                max_total_pigment - total - min_needed_for_rest
            )
            if max_this < min_ratio:
                return

            for ratio in ratio_axes[depth]:
                r = float(ratio)
                if r > max_this + 1e-9:
                    break
                walk(depth + 1, chosen + [r], total + r)

        walk(0, [], 0.0)
        flush_batch()
        return best_ratios, best_error

    mode = (quality_mode or "balanced").lower()
    if mode == "high":
        coarse_step = 0.02
        fine_step = 0.008
        fine_radius = 0.035
    elif mode == "server_fast":
        coarse_step = 0.035 if n_pigments >= 3 else 0.025
        fine_step = 0.018
        fine_radius = 0.02
    elif mode == "fast":
        coarse_step = 0.03 if n_pigments >= 3 else 0.02
        fine_step = 0.015
        fine_radius = 0.022
    else:
        # balanced
        coarse_step = 0.025 if n_pigments >= 3 else 0.02
        fine_step = 0.01
        fine_radius = 0.03

    seed = _l_star_seed_ratios(
        target_lab,
        paint_calibrations,
        n_pigments,
        min_ratio,
        max_total_pigment,
        max_ratio_per_pigment,
        library_group,
        paint_ids,
    )
    coarse_ratios: Optional[List[float]] = None
    coarse_error = float("inf")
    if seed is not None:
        coarse_ratios, coarse_error = search(coarse_step, center=seed, radius=0.14)
    if coarse_ratios is None or coarse_error > 0.35:
        full_ratios, full_error = search(coarse_step)
        if full_ratios is not None and full_error < coarse_error:
            coarse_ratios, coarse_error = full_ratios, full_error
    if coarse_ratios is None:
        return None

    fine_ratios, fine_error = search(fine_step, center=coarse_ratios, radius=fine_radius)

    best_ratios = fine_ratios if fine_ratios is not None and fine_error <= coarse_error else coarse_ratios
    best_error = fine_error if fine_ratios is not None and fine_error <= coarse_error else coarse_error
    white_ratio = 1.0 - sum(best_ratios)

    result = {
        'pigment_ids': paint_ids,
        'pigment_ratios': best_ratios,
        'white_ratio': white_ratio,
        'error': best_error,
        'type': ratio_type,
        'uncalibrated': calibrated_count < n_pigments
    }
    return _apply_physical_value_caps_to_recipe(result, target_lab, library_group)


def predict_mix_lab_for_paint_ratios(
    library_group: str,
    pigment_ids: List[str],
    pigment_ratios: List[float],
    white_ratio: float,
) -> Optional[List[float]]:
    """Predict CIELAB for a recipe from paint ids and mass fractions (sums to ~1)."""
    if len(pigment_ids) != len(pigment_ratios):
        return None
    components: List[Tuple[Optional[Dict], Optional[List[int]], float]] = []
    for pid, ratio in zip(pigment_ids, pigment_ratios):
        if ratio <= 1e-12:
            continue
        cal = _load_calibration_cached(pid, library_group)
        hex_rgb: Optional[List[int]] = None
        if cal is None:
            lib = load_library(library_group)
            for p in lib.get("paints", []):
                if p.get("id") == pid:
                    hx = str(p.get("hex_approx", "")).lstrip("#")
                    if len(hx) == 6:
                        hex_rgb = [int(hx[i : i + 2], 16) for i in (0, 2, 4)]
                    break
        components.append((cal, hex_rgb, float(ratio)))
    return _predict_mix_lab_from_components(
        components,
        float(white_ratio),
        library_group=library_group,
        paint_ids=[pid for pid, r in zip(pigment_ids, pigment_ratios) if r > 1e-12],
    )


def generate_recipes_for_palette(
    session_id: str,
    palette: List[Dict],
    library_group: str = "default",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    recipe_cb: Optional[Callable[[Dict], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    quality_mode: str = "balanced",
) -> List[Dict]:
    """Generate paint mixing recipes for each palette color.
    
    Args:
        session_id: Session ID (unused, kept for compatibility)
        palette: List of palette colors with rgb and index
        library_group: Library group to use for recipe generation (default: "default")
    
    Will use calibrated paints if available, otherwise falls back to approximate colors.
    """
    library = load_library(library_group)
    paints = library.get('paints', [])
    
    # Filter to base paints only, excluding white and black (they're mixing components, not color pigments)
    base_paints = [p for p in paints if p.get('type') == 'base']
    
    # Identify white and black paints (by name or color)
    def is_achromatic(paint: Dict) -> bool:
        """Check if a paint is white, black, or essentially achromatic (gray)."""
        paint_id_lower = paint.get('id', '').lower()
        paint_name_lower = paint.get('name', '').lower()
        
        # Check by name
        achromatic_names = ['white', 'black', 'carbon black', 'titanium white', 'zinc white']
        if any(name in paint_id_lower or name in paint_name_lower for name in achromatic_names):
            return True
        
        # Check by color - if it's very close to white, black, or gray
        hex_color = paint.get('hex_approx', '')
        if hex_color:
            try:
                hex_clean = hex_color.lstrip('#')
                r = int(hex_clean[0:2], 16)
                g = int(hex_clean[2:4], 16)
                b = int(hex_clean[4:6], 16)
                
                # Check if it's essentially white (all channels > 240)
                if r > 240 and g > 240 and b > 240:
                    return True
                
                # Check if it's essentially black (all channels < 20)
                if r < 20 and g < 20 and b < 20:
                    return True
                
                # Check if it's essentially gray (all channels similar, within 30 of each other)
                if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
                    # And it's not a very saturated color
                    max_channel = max(r, g, b)
                    min_channel = min(r, g, b)
                    if max_channel - min_channel < 40:  # Low saturation = gray
                        return True
            except:
                pass
        
        return False
    
    def is_white_paint(paint: Dict) -> bool:
        """Check if paint behaves like white and should not be a pigment candidate."""
        paint_id_lower = paint.get('id', '').lower()
        paint_name_lower = paint.get('name', '').lower()
        if 'white' in paint_id_lower or 'white' in paint_name_lower:
            return True

        hex_color = paint.get('hex_approx', '')
        if hex_color:
            try:
                hex_clean = hex_color.lstrip('#')
                r = int(hex_clean[0:2], 16)
                g = int(hex_clean[2:4], 16)
                b = int(hex_clean[4:6], 16)
                return r > 240 and g > 240 and b > 240
            except Exception:
                return False
        return False

    # Separate colored paints from achromatic ones
    colored_paints = [p for p in base_paints if not is_achromatic(p)]
    achromatic_paints = [p for p in base_paints if is_achromatic(p)]
    
    # Use colored paints for matching, but keep achromatic for reference
    if not colored_paints:
        # Fallback: if no colored paints, use all paints but with heavy penalty
        colored_paints = base_paints
    
    if not base_paints:
        # No paints available at all
        return [{
            'palette_index': color['index'],
            'recipe': None,
            'error': 'No paints in library. Add paints to the Paint Library first.'
        } for color in palette]
    
    recipes = []

    def _push_recipe(item: Dict) -> None:
        recipes.append(item)
        if recipe_cb:
            try:
                recipe_cb(item)
            except Exception:
                pass
    mode = (quality_mode or "balanced").lower()
    if mode == "high":
        candidate_paint_limit = 8
        max_pigments_cfg = 4
        early_exit_delta = 0.1
    elif mode == "server_fast":
        candidate_paint_limit = 5
        max_pigments_cfg = 3
        early_exit_delta = 0.45
    elif mode == "fast":
        candidate_paint_limit = 5
        max_pigments_cfg = 3
        early_exit_delta = 0.35
    else:
        # balanced defaults
        try:
            candidate_paint_limit = max(3, int(os.getenv("RECIPE_CANDIDATE_PAINT_LIMIT", "6")))
        except Exception:
            candidate_paint_limit = 6
        try:
            max_pigments_cfg = max(2, int(os.getenv("RECIPE_MAX_PIGMENTS", "4")))
        except Exception:
            max_pigments_cfg = 4
        try:
            early_exit_delta = max(0.05, float(os.getenv("RECIPE_EARLY_EXIT_DELTA", "0.2")))
        except Exception:
            early_exit_delta = 0.2

    total_colors = len(palette)
    for i, color in enumerate(palette):
        if cancel_cb and cancel_cb():
            if progress_cb:
                try:
                    progress_cb(i, total_colors, "cancelled")
                except Exception:
                    pass
            return recipes
        if progress_cb:
            try:
                progress_cb(i, total_colors, "running")
            except Exception:
                pass
        palette_index = color.get('index')
        color_t0 = time.perf_counter()
        try:
            target_rgb = color.get('rgb')
            if not isinstance(target_rgb, (list, tuple)) or len(target_rgb) < 3:
                _push_recipe({
                    'palette_index': palette_index,
                    'recipe': None,
                    'error': 'Color format error: missing or invalid rgb'
                })
                continue

            target_lab = rgb_to_lab([float(c) for c in target_rgb[:3]])
        
            # Exclude white from pigment candidates because white is modeled by white_ratio.
            non_white_base_paints = [p for p in base_paints if not is_white_paint(p)]

            # Try one-pigment recipes first across all non-white paints.
            # Restricting to "colored only" often blocks good dark/neutral corrections.
            search_paints = non_white_base_paints
            if not search_paints:
                search_paints = non_white_base_paints
            best_one_pigment = None
            best_one_error = float('inf')
            one_pigment_candidates = []
        
            for paint in search_paints:
                paint_hex = paint.get('hex_approx', '')
                recipe = find_best_one_pigment_recipe(target_lab, paint['id'], paint_hex, library_group=library_group)
                if recipe:
                    if recipe['error'] < best_one_error:
                        best_one_error = recipe['error']
                        best_one_pigment = recipe
                    one_pigment_candidates.append({
                        'paint': paint,
                        'error': recipe['error']
                    })
        
            # Try multi-pigment recipes and pick the lowest error with a mild complexity tie-break.
            best_multi_pigment = None
            best_multi_error = float('inf')
            best_multi_score = float('inf')

            # Adaptive search depth:
            # - easy colors keep a narrow/fast search
            # - hard colors widen candidate paints and allow more pigments
            if best_one_error <= 1.5:
                local_paint_limit = max(3, candidate_paint_limit - 2)
                local_max_pigments = min(3, max_pigments_cfg)
                local_early_exit = early_exit_delta
            elif best_one_error <= 3.5:
                local_paint_limit = candidate_paint_limit
                local_max_pigments = min(4, max_pigments_cfg)
                local_early_exit = max(0.1, early_exit_delta * 0.75)
            else:
                local_paint_limit = candidate_paint_limit + 2
                local_max_pigments = min(4, max_pigments_cfg)
                local_early_exit = 0.08

            # Limit combinatorial explosion by using closest single-pigment candidates first.
            ranked_paints = [p['paint'] for p in sorted(one_pigment_candidates, key=lambda x: x['error'])]
            if ranked_paints:
                multi_search_paints = ranked_paints[: min(local_paint_limit, len(ranked_paints))]
            else:
                multi_search_paints = search_paints[: min(local_paint_limit, len(search_paints))]

            max_pigments = min(local_max_pigments, len(multi_search_paints))
            for pigment_count in range(2, max_pigments + 1):
                if cancel_cb and cancel_cb():
                    if progress_cb:
                        try:
                            progress_cb(i, total_colors, "cancelled")
                        except Exception:
                            pass
                    return recipes
                if best_multi_error <= local_early_exit:
                    break
                for combo in combinations(multi_search_paints, pigment_count):
                    if cancel_cb and cancel_cb():
                        if progress_cb:
                            try:
                                progress_cb(i, total_colors, "cancelled")
                            except Exception:
                                pass
                        return recipes
                    paint_ids = [p['id'] for p in combo]
                    if not _combo_allowed_for_target(target_lab, paint_ids):
                        continue
                    paint_hexes = [p.get('hex_approx', '') for p in combo]

                    if pigment_count == 2:
                        recipe = find_best_two_pigment_recipe(
                            target_lab,
                            paint_ids[0],
                            paint_ids[1],
                            paint_hexes[0],
                            paint_hexes[1],
                            library_group=library_group,
                        )
                    else:
                        recipe = find_best_multi_pigment_recipe(
                            target_lab,
                            paint_ids,
                            paint_hexes,
                            library_group=library_group,
                            quality_mode=mode,
                        )

                    if not recipe:
                        continue

                    adjusted_error = recipe['error']

                    complexity_penalty = max(0, pigment_count - 2) * 0.15
                    score = adjusted_error + complexity_penalty
                    if score < best_multi_score:
                        best_multi_score = score
                        best_multi_error = adjusted_error
                        best_multi_pigment = recipe
                if best_multi_error <= local_early_exit:
                    break

            # Choose best available recipe by error.
            if best_multi_pigment and (not best_one_pigment or best_multi_error <= best_one_error):
                _push_recipe({
                    'palette_index': palette_index,
                    'recipe': best_multi_pigment,
                    'type': best_multi_pigment.get('type', 'multi_pigment')
                })
            elif best_one_pigment:
                _push_recipe({
                    'palette_index': palette_index,
                    'recipe': best_one_pigment,
                    'type': 'one_pigment'
                })
            else:
                _push_recipe({
                    'palette_index': palette_index,
                    'recipe': None,
                    'error': 'Could not generate recipe (no valid candidate)'
                })
        except Exception as e:
            logger.exception("Recipe generation failed for palette_index=%s: %s", palette_index, e)
            _push_recipe({
                'palette_index': palette_index,
                'recipe': None,
                'error': f'Recipe generation failed for this color: {e}'
            })
        finally:
            elapsed_ms = (time.perf_counter() - color_t0) * 1000.0
            chosen = recipes[-1] if recipes else {}
            err = None
            if isinstance(chosen, dict):
                r = chosen.get("recipe")
                if isinstance(r, dict):
                    err = r.get("error")
            logger.info(
                "Recipe color %s/%s index=%s done in %.0f ms (best dE=%s)",
                i + 1,
                total_colors,
                palette_index,
                elapsed_ms,
                f"{err:.2f}" if isinstance(err, (int, float)) else err,
            )
            if progress_cb:
                try:
                    progress_cb(i + 1, total_colors, "running")
                except Exception:
                    pass

    if progress_cb:
        try:
            progress_cb(total_colors, total_colors, "completed")
        except Exception:
            pass
    
    return recipes
