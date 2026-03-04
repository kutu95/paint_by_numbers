import json
import cv2
import numpy as np
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
import os
import re
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


def _empty_library(group: str) -> Dict:
    return {"version": 1, "paints": [], "group": group}


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
    return normalized


def get_white_mix_limits(target_lab: List[float]) -> Tuple[float, float]:
    """Return (min_white_ratio, max_total_pigment) based on target lightness."""
    lightness = float(target_lab[0]) if target_lab and len(target_lab) > 0 else 60.0
    # Keep constraints permissive; strict minimum white causes large chroma errors.
    if lightness < 25.0:
        min_white_ratio = 0.01
    elif lightness < 40.0:
        min_white_ratio = 0.03
    elif lightness < 55.0:
        min_white_ratio = 0.08
    elif lightness < 70.0:
        min_white_ratio = 0.15
    else:
        min_white_ratio = 0.30
    max_total_pigment = 1.0 - min_white_ratio
    return min_white_ratio, max_total_pigment


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
                    return _coerce_library_shape(data, group)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning("Failed to load legacy library file %s: %s", LIBRARY_FILE, e)
    
    # Load from group-specific file
    library_file = LIBRARIES_DIR / f"{group}.json"
    if not library_file.exists():
        return _empty_library(group)
    try:
        with open(library_file, 'r') as f:
            return _coerce_library_shape(json.load(f), group)
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


def load_recipe_cache(group: str) -> Dict[str, Dict]:
    """Load recipe cache for a library group.
    
    Recipes are cached by hex color (normalized to uppercase).
    
    Args:
        group: Library group name
    
    Returns:
        Dictionary mapping hex color to cached recipe data
    """
    cache_file = get_recipe_cache_file(group)
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
            # Ensure it's a dict with hex keys
            if isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, IOError):
        return {}


def save_recipe_cache(group: str, cache: Dict[str, Dict]):
    """Save recipe cache for a library group.
    
    Args:
        group: Library group name
        cache: Dictionary mapping hex color to recipe data
    """
    cache_file = get_recipe_cache_file(group)
    atomic_write(cache_file, cache)


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
    cache[hex_normalized] = recipe
    save_recipe_cache(group, cache)


def get_library_info(group: str) -> Dict:
    """Get information about a library group."""
    library = load_library(group)
    paints = [p for p in library.get('paints', []) if isinstance(p, dict)]
    paint_count = len(paints)
    calibrated_count = sum(
        1 for p in paints
        if (CALIBRATION_DIR / f"{p.get('id', '')}.json").exists()
    )
    
    # Use stored name if available, otherwise generate from group ID
    name = library.get('name', group.replace("-", " ").title())
    
    return {
        "group": group,
        "paint_count": paint_count,
        "calibrated_count": calibrated_count,
        "name": name,
        "coverage_mg_per_cm2": library.get("coverage_mg_per_cm2"),
    }


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


def get_hex_from_calibration(paint_id: str) -> Optional[str]:
    """Get the approximate hex color from a paint's calibration 100% swatch.
    Returns None if no calibration or no valid sample.
    """
    cal_file = CALIBRATION_DIR / f"{paint_id}.json"
    if not cal_file.exists():
        return None
    try:
        with open(cal_file, 'r') as f:
            cal = json.load(f)
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


def interpolate_lab_from_calibration(calibration: Dict, ratio: float) -> Optional[List[float]]:
    """Interpolate Lab color for a given ratio from calibration samples."""
    samples = calibration.get('samples', [])
    if not samples:
        return None
    
    # Sort by ratio
    sorted_samples = sorted(samples, key=lambda x: x['ratio'])
    ratios = [s['ratio'] for s in sorted_samples]
    labs = [_coerce_lab_to_cielab(s['lab']) for s in sorted_samples]
    
    # Find bounding ratios
    if ratio <= ratios[0]:
        return labs[0]
    if ratio >= ratios[-1]:
        return labs[-1]
    
    # Linear interpolation
    for i in range(len(ratios) - 1):
        if ratios[i] <= ratio <= ratios[i + 1]:
            t = (ratio - ratios[i]) / (ratios[i + 1] - ratios[i])
            lab = [
                labs[i][0] + t * (labs[i + 1][0] - labs[i][0]),
                labs[i][1] + t * (labs[i + 1][1] - labs[i][1]),
                labs[i][2] + t * (labs[i + 1][2] - labs[i][2])
            ]
            return lab
    
    return None


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


def _rgb_to_linear(rgb: List[float]) -> np.ndarray:
    arr = np.array([max(0.0, min(255.0, float(c))) / 255.0 for c in rgb], dtype=np.float64)
    return np.power(arr, 2.2)


def _linear_to_rgb(linear_rgb: np.ndarray) -> List[float]:
    clamped = np.clip(np.asarray(linear_rgb, dtype=np.float64), 0.0, 1.0)
    srgb = np.power(clamped, 1.0 / 2.2) * 255.0
    return [float(v) for v in np.clip(np.round(srgb), 0, 255)]


def _predict_mix_lab_from_components(
    components: List[Tuple[Optional[Dict], Optional[List[int]], float]],
    white_ratio: float,
) -> Optional[List[float]]:
    """Predict mixed Lab from components using calibration curves + white-aware effective ratios.

    Each pigment is first converted to its calibrated tint shade at ratio:
      eff = pigment_ratio / (pigment_ratio + white_ratio)
    then blended with other pigments by pigment share in linear RGB.
    """
    white_ratio = max(0.0, min(1.0, float(white_ratio)))
    total_pigment = sum(max(0.0, float(r)) for _, _, r in components)
    if total_pigment <= 0:
        return [100.0, 0.0, 0.0]

    mixed_linear = np.zeros(3, dtype=np.float64)
    used = 0.0

    for calibration, hex_rgb, pigment_ratio in components:
        p = max(0.0, float(pigment_ratio))
        if p <= 0:
            continue
        share = p / total_pigment
        eff_ratio = p / (p + white_ratio) if (p + white_ratio) > 0 else 0.0
        eff_ratio = max(0.0, min(1.0, eff_ratio))

        rgb: Optional[List[float]] = None
        if calibration is not None:
            lab = interpolate_lab_from_calibration(calibration, eff_ratio)
            if lab is not None:
                rgb = lab_to_rgb(lab)
        elif hex_rgb is not None:
            # Uncalibrated fallback: tint paint hex toward white by effective ratio.
            paint_lin = _rgb_to_linear([float(c) for c in hex_rgb])
            white_lin = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            tint_lin = (paint_lin * eff_ratio) + (white_lin * (1.0 - eff_ratio))
            rgb = _linear_to_rgb(tint_lin)

        if rgb is None:
            continue

        mixed_linear += _rgb_to_linear(rgb) * share
        used += share

    if used <= 0.0:
        return None
    if used < 0.999:
        mixed_linear /= used
    mixed_rgb = _linear_to_rgb(mixed_linear)
    return rgb_to_lab(mixed_rgb)


def find_best_one_pigment_recipe(target_lab: List[float], paint_id: str, paint_hex: str = None) -> Optional[Dict]:
    """Find best mixing ratio for one pigment + white to match target Lab.
    
    Args:
        target_lab: Target color in Lab color space
        paint_id: Paint ID
        paint_hex: Optional hex color for uncalibrated paints (e.g., '#FF0000')
    
    Returns:
        Recipe dict with pigment_id, pigment_ratio, white_ratio, and error
    """
    calibration_file = CALIBRATION_DIR / f"{paint_id}.json"
    
    min_white_ratio, max_total_pigment = get_white_mix_limits(target_lab)

    # If calibration exists, use it (more accurate)
    if calibration_file.exists():
        with open(calibration_file, 'r') as f:
            calibration = json.load(f)
        
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
                predicted_lab = interpolate_lab_from_calibration(calibration, test_ratio)
                if predicted_lab:
                    error = delta_e_lab(target_lab, predicted_lab)
                    if error < best_error:
                        best_error = error
                        best_ratio = test_ratio
            
            if best_ratio is not None:
                return {
                    'pigment_id': paint_id,
                    'pigment_ratio': best_ratio,
                    'white_ratio': 1.0 - best_ratio,
                    'error': best_error
                }

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


def find_best_two_pigment_recipe(target_lab: List[float], paint_id1: str, paint_id2: str, paint1_hex: str = None, paint2_hex: str = None) -> Optional[Dict]:
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
    cal1_file = CALIBRATION_DIR / f"{paint_id1}.json"
    cal2_file = CALIBRATION_DIR / f"{paint_id2}.json"
    min_white_ratio, max_total_pigment = get_white_mix_limits(target_lab)
    min_ratio = 0.02
    max_ratio_per_pigment = max_total_pigment - min_ratio
    
    # Try calibrated first if both exist
    if cal1_file.exists() and cal2_file.exists():
        with open(cal1_file, 'r') as f:
            cal1 = json.load(f)
        with open(cal2_file, 'r') as f:
            cal2 = json.load(f)
        cal1_max = max(0.0, min(1.0, get_calibration_max_ratio(cal1)))
        cal2_max = max(0.0, min(1.0, get_calibration_max_ratio(cal2)))
        if cal1_max <= 0.0 or cal2_max <= 0.0:
            return None
        
        # Fine grid search for better accuracy
        best_error = float('inf')
        best_recipe = None
        
        for p1_ratio in np.arange(min_ratio, min(max_ratio_per_pigment, cal1_max) + 0.001, 0.02):
            for p2_ratio in np.arange(min_ratio, min(max_ratio_per_pigment, cal2_max) + 0.001, 0.02):
                if p1_ratio + p2_ratio > max_total_pigment:
                    continue
                white_ratio = 1.0 - p1_ratio - p2_ratio
                if white_ratio < min_white_ratio:
                    continue

                blended_lab = _predict_mix_lab_from_components(
                    [(cal1, None, p1_ratio), (cal2, None, p2_ratio)],
                    white_ratio,
                )
                if blended_lab is None:
                    continue

                error = delta_e_lab(target_lab, blended_lab)
                if error < best_error:
                    best_error = error
                    best_recipe = {
                        'pigment1_id': paint_id1,
                        'pigment1_ratio': p1_ratio,
                        'pigment2_id': paint_id2,
                        'pigment2_ratio': p2_ratio,
                        'white_ratio': white_ratio,
                        'error': best_error,
                        'type': 'two_pigment'
                    }
        
        if best_recipe:
            return best_recipe
    
    # Fallback: Use approximate colors if available (less accurate)
    if paint1_hex and paint2_hex:
        try:
            # Convert hex to Lab
            hex1_clean = paint1_hex.lstrip('#')
            rgb1 = [int(hex1_clean[i:i+2], 16) for i in (0, 2, 4)]
            lab1 = rgb_to_lab(rgb1)
            
            hex2_clean = paint2_hex.lstrip('#')
            rgb2 = [int(hex2_clean[i:i+2], 16) for i in (0, 2, 4)]
            lab2 = rgb_to_lab(rgb2)
            
            # Fine grid search with approximate colors
            best_error = float('inf')
            best_recipe = None
            
            for p1_ratio in np.arange(min_ratio, max_ratio_per_pigment + 0.001, 0.02):
                for p2_ratio in np.arange(min_ratio, max_ratio_per_pigment + 0.001, 0.02):
                    if p1_ratio + p2_ratio > max_total_pigment:
                        continue
                    white_ratio = 1.0 - p1_ratio - p2_ratio
                    if white_ratio < min_white_ratio:
                        continue

                    blended_lab = _predict_mix_lab_from_components(
                        [(None, rgb1, p1_ratio), (None, rgb2, p2_ratio)],
                        white_ratio,
                    )
                    if blended_lab is None:
                        continue

                    error = delta_e_lab(target_lab, blended_lab)
                    # Add penalty for uncalibrated
                    error += 5.0
                    
                    if error < best_error:
                        best_error = error
                        best_recipe = {
                            'pigment1_id': paint_id1,
                            'pigment1_ratio': p1_ratio,
                            'pigment2_id': paint_id2,
                            'pigment2_ratio': p2_ratio,
                            'white_ratio': white_ratio,
                            'error': best_error,
                            'type': 'two_pigment',
                            'uncalibrated': True
                        }
            
            return best_recipe
        except Exception:
            return None
    
    return None


def find_best_multi_pigment_recipe(target_lab: List[float], paint_ids: List[str], paint_hexes: List[str]) -> Optional[Dict]:
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
        cal_file = CALIBRATION_DIR / f"{paint_id}.json"
        calibration = None
        if cal_file.exists():
            with open(cal_file, 'r') as f:
                calibration = json.load(f)
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
    min_white_ratio, max_total_pigment = get_white_mix_limits(target_lab)
    max_ratio_per_pigment = max_total_pigment - (n_pigments - 1) * min_ratio
    if max_ratio_per_pigment < min_ratio:
        return None

    ratio_type = {
        3: 'three_pigment',
        4: 'four_pigment',
    }.get(n_pigments, f'{n_pigments}_pigment')

    def evaluate(ratios: List[float]) -> Optional[float]:
        white_ratio = 1.0 - sum(ratios)
        if white_ratio < min_white_ratio:
            return None
        if sum(ratios) > max_total_pigment:
            return None

        components: List[Tuple[Optional[Dict], Optional[List[int]], float]] = []
        for calibration, hex_rgb, ratio, max_ratio in zip(paint_calibrations, paint_hex_colors, ratios, paint_max_ratios):
            if ratio - max_ratio > 1e-9:
                return None
            if calibration is None and hex_rgb is None:
                return None
            components.append((calibration, hex_rgb, ratio))

        blended_lab = _predict_mix_lab_from_components(components, white_ratio)
        if blended_lab is None:
            return None

        error = delta_e_lab(target_lab, blended_lab)
        if calibrated_count < n_pigments:
            error += (n_pigments - calibrated_count) * (3.0 if n_pigments == 3 else 4.0)
        return error

    def search(step: float, center: Optional[List[float]] = None, radius: float = 0.0) -> Tuple[Optional[List[float]], float]:
        ratio_axes = []
        for i in range(n_pigments):
            low = min_ratio
            high = max_ratio_per_pigment
            if center is not None:
                low = max(min_ratio, center[i] - radius)
                high = min(max_ratio_per_pigment, center[i] + radius)
            axis = np.arange(low, high + (step * 0.5), step)
            if len(axis) == 0:
                return None, float('inf')
            ratio_axes.append(axis)

        best_ratios = None
        best_error = float('inf')

        def walk(depth: int, chosen: List[float], total: float):
            nonlocal best_ratios, best_error
            remaining = n_pigments - depth
            if depth == n_pigments:
                error = evaluate(chosen)
                if error is not None and error < best_error:
                    best_error = error
                    best_ratios = chosen.copy()
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
        return best_ratios, best_error

    coarse_step = 0.02 if n_pigments == 3 else 0.015
    coarse_ratios, coarse_error = search(coarse_step)
    if coarse_ratios is None:
        return None

    # Keep refinement slightly coarser for runtime stability in API context.
    fine_step = 0.01
    fine_radius = 0.03
    fine_ratios, fine_error = search(fine_step, center=coarse_ratios, radius=fine_radius)

    best_ratios = fine_ratios if fine_ratios is not None and fine_error <= coarse_error else coarse_ratios
    best_error = fine_error if fine_ratios is not None and fine_error <= coarse_error else coarse_error
    white_ratio = 1.0 - sum(best_ratios)

    return {
        'pigment_ids': paint_ids,
        'pigment_ratios': best_ratios,
        'white_ratio': white_ratio,
        'error': best_error,
        'type': ratio_type,
        'uncalibrated': calibrated_count < n_pigments
    }


def generate_recipes_for_palette(
    session_id: str,
    palette: List[Dict],
    library_group: str = "default",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
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

    total_colors = len(palette)
    for i, color in enumerate(palette):
        if progress_cb:
            try:
                progress_cb(i, total_colors, "running")
            except Exception:
                pass
        palette_index = color.get('index')
        try:
            target_rgb = color.get('rgb')
            if not isinstance(target_rgb, (list, tuple)) or len(target_rgb) < 3:
                recipes.append({
                    'palette_index': palette_index,
                    'recipe': None,
                    'error': 'Color format error: missing or invalid rgb'
                })
                continue

            target_lab = rgb_to_lab([float(c) for c in target_rgb[:3]])
        
            # Check if target is essentially achromatic (gray/white/black)
            target_is_achromatic = False
            r, g, b = target_rgb
            if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
                max_channel = max(r, g, b)
                min_channel = min(r, g, b)
                if max_channel - min_channel < 40:  # Low saturation = gray
                    target_is_achromatic = True
        
            # Exclude white from pigment candidates because white is modeled by white_ratio.
            non_white_colored_paints = [p for p in colored_paints if not is_white_paint(p)]
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
                recipe = find_best_one_pigment_recipe(target_lab, paint['id'], paint_hex)
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

            # Limit combinatorial explosion by using closest single-pigment candidates first.
            ranked_paints = [p['paint'] for p in sorted(one_pigment_candidates, key=lambda x: x['error'])]
            if ranked_paints:
                multi_search_paints = ranked_paints[: min(7, len(ranked_paints))]
            else:
                multi_search_paints = search_paints[: min(7, len(search_paints))]

            max_pigments = min(4, len(multi_search_paints))
            for pigment_count in range(2, max_pigments + 1):
                if best_multi_error <= 1.5:
                    break
                for combo in combinations(multi_search_paints, pigment_count):
                    paint_ids = [p['id'] for p in combo]
                    paint_hexes = [p.get('hex_approx', '') for p in combo]

                    if pigment_count == 2:
                        recipe = find_best_two_pigment_recipe(
                            target_lab,
                            paint_ids[0],
                            paint_ids[1],
                            paint_hexes[0],
                            paint_hexes[1]
                        )
                    else:
                        recipe = find_best_multi_pigment_recipe(target_lab, paint_ids, paint_hexes)

                    if not recipe:
                        continue

                    adjusted_error = recipe['error']

                    complexity_penalty = max(0, pigment_count - 2) * 0.15
                    score = adjusted_error + complexity_penalty
                    if score < best_multi_score:
                        best_multi_score = score
                        best_multi_error = adjusted_error
                        best_multi_pigment = recipe

            # Choose best available recipe by error.
            if best_multi_pigment and (not best_one_pigment or best_multi_error <= best_one_error):
                recipes.append({
                    'palette_index': palette_index,
                    'recipe': best_multi_pigment,
                    'type': best_multi_pigment.get('type', 'multi_pigment')
                })
            elif best_one_pigment:
                recipes.append({
                    'palette_index': palette_index,
                    'recipe': best_one_pigment,
                    'type': 'one_pigment'
                })
            else:
                recipes.append({
                    'palette_index': palette_index,
                    'recipe': None,
                    'error': 'Could not generate recipe (no valid candidate)'
                })
        except Exception as e:
            logger.exception("Recipe generation failed for palette_index=%s: %s", palette_index, e)
            recipes.append({
                'palette_index': palette_index,
                'recipe': None,
                'error': f'Recipe generation failed for this color: {e}'
            })
        finally:
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
