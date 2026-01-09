import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re


# Data directories
PAINT_DIR = Path(__file__).parent.parent / "data" / "paint"
PAINT_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_DIR = PAINT_DIR / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_FILE = PAINT_DIR / "library.json"
LIBRARIES_DIR = PAINT_DIR / "libraries"
LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)


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
        with open(LIBRARY_FILE, 'r') as f:
            data = json.load(f)
            # Migrate to new structure if needed
            if "groups" not in data:
                return data
    
    # Load from group-specific file
    library_file = LIBRARIES_DIR / f"{group}.json"
    if not library_file.exists():
        return {"version": 1, "paints": [], "group": group}
    with open(library_file, 'r') as f:
        return json.load(f)


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


def get_library_info(group: str) -> Dict:
    """Get information about a library group."""
    library = load_library(group)
    paint_count = len(library.get('paints', []))
    calibrated_count = sum(1 for p in library.get('paints', []) 
                          if (CALIBRATION_DIR / f"{p.get('id', '')}.json").exists())
    
    return {
        "group": group,
        "paint_count": paint_count,
        "calibrated_count": calibrated_count,
        "name": group.replace("-", " ").title()
    }


def rgb_to_lab(rgb: List[float]) -> List[float]:
    """Convert RGB to Lab color space."""
    rgb_array = np.array([[rgb]], dtype=np.uint8)
    lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    return lab_array[0, 0].tolist()


def lab_to_rgb(lab: List[float]) -> List[float]:
    """Convert Lab to RGB color space."""
    lab_array = np.array([[lab]], dtype=np.uint8)
    rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)
    return rgb_array[0, 0].tolist()


def delta_e_lab(lab1: List[float], lab2: List[float]) -> float:
    """Calculate Euclidean distance in Lab space (simple ΔE)."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def sample_color_from_image(image_path: str, x: int, y: int, radius: int = 5) -> Tuple[List[int], List[float]]:
    """Sample color from image at given coordinates (average over small area)."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
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


def interpolate_lab_from_calibration(calibration: Dict, ratio: float) -> Optional[List[float]]:
    """Interpolate Lab color for a given ratio from calibration samples."""
    samples = calibration.get('samples', [])
    if not samples:
        return None
    
    # Sort by ratio
    sorted_samples = sorted(samples, key=lambda x: x['ratio'])
    ratios = [s['ratio'] for s in sorted_samples]
    labs = [s['lab'] for s in sorted_samples]
    
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
    
    # If calibration exists, use it (more accurate)
    if calibration_file.exists():
        with open(calibration_file, 'r') as f:
            calibration = json.load(f)
        
        samples = calibration.get('samples', [])
        if samples:
            # Search for best ratio
            best_ratio = None
            best_error = float('inf')
            
            # Test ratios from 0 to 0.5 in small steps
            for test_ratio in np.arange(0.0, 0.51, 0.01):
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
            
            # Estimate pigment ratio (0.0 to 0.5 range)
            # If target is lighter, use less pigment
            if lightness_diff > 0:
                # Target is lighter - use less pigment
                estimated_ratio = max(0.05, min(0.5, 0.5 - (lightness_diff / 200.0)))
            else:
                # Target is darker - use more pigment
                estimated_ratio = max(0.1, min(0.5, 0.5 + (abs(lightness_diff) / 200.0)))
            
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
    
    # Try calibrated first if both exist
    if cal1_file.exists() and cal2_file.exists():
        with open(cal1_file, 'r') as f:
            cal1 = json.load(f)
        with open(cal2_file, 'r') as f:
            cal2 = json.load(f)
        
        # Fine grid search for better accuracy
        best_error = float('inf')
        best_recipe = None
        
        # Allow up to 60% total pigment (40% white minimum)
        max_total_pigment = 0.6
        for p1_ratio in np.arange(0.02, 0.35, 0.02):
            for p2_ratio in np.arange(0.02, 0.35, 0.02):
                if p1_ratio + p2_ratio > max_total_pigment:
                    continue
                # Ensure minimum ratio for each pigment (at least 2% each)
                if p1_ratio < 0.02 or p2_ratio < 0.02:
                    continue
                
                # Approximate: blend the two single-pigment Labs at their ratios
                lab1 = interpolate_lab_from_calibration(cal1, p1_ratio)
                lab2 = interpolate_lab_from_calibration(cal2, p2_ratio)
                
                if lab1 and lab2:
                    # Simple weighted blend (approximation)
                    white_ratio = 1.0 - p1_ratio - p2_ratio
                    blended_lab = [
                        lab1[0] * p1_ratio + lab2[0] * p2_ratio + 100.0 * white_ratio * 0.9,  # Approximate white Lab
                        lab1[1] * p1_ratio + lab2[1] * p2_ratio,
                        lab1[2] * p1_ratio + lab2[2] * p2_ratio
                    ]
                    
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
            
            # Allow up to 60% total pigment (40% white minimum)
            max_total_pigment = 0.6
            for p1_ratio in np.arange(0.02, 0.35, 0.02):
                for p2_ratio in np.arange(0.02, 0.35, 0.02):
                    if p1_ratio + p2_ratio > max_total_pigment:
                        continue
                    # Ensure minimum ratio for each pigment (at least 2% each)
                    if p1_ratio < 0.02 or p2_ratio < 0.02:
                        continue
                    
                    white_ratio = 1.0 - p1_ratio - p2_ratio
                    blended_lab = [
                        lab1[0] * p1_ratio + lab2[0] * p2_ratio + 100.0 * white_ratio * 0.9,
                        lab1[1] * p1_ratio + lab2[1] * p2_ratio,
                        lab1[2] * p1_ratio + lab2[2] * p2_ratio
                    ]
                    
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
    # ONLY handle 3+ pigments - no 2-pigment recipes
    if n_pigments < 3:
        return None
    
    # Store calibration data and hex colors for each paint
    # We'll interpolate calibrated colors at the actual ratio being tested
    paint_calibrations = []
    paint_hex_colors = []
    calibrated_count = 0
    
    for paint_id, paint_hex in zip(paint_ids, paint_hexes):
        cal_file = CALIBRATION_DIR / f"{paint_id}.json"
        calibration = None
        if cal_file.exists():
            with open(cal_file, 'r') as f:
                calibration = json.load(f)
            calibrated_count += 1
        
        paint_calibrations.append(calibration)
        
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
    
    # Grid search for best ratios
    # For n pigments, we'll try different combinations
    # Keep total pigment ratio <= 0.5 (white dominant)
    best_error = float('inf')
    best_recipe = None
    
    # Two-stage grid search: coarse first, then refine around best result
    # Coarse grid search - use smaller steps for better accuracy
    coarse_step = 0.02 if n_pigments == 3 else 0.015 if n_pigments == 4 else 0.015
    
    # Generate all combinations of ratios
    # Allow more flexibility - total pigment can be up to 0.6 (white can be 0.4)
    max_total_pigment = 0.6
    max_ratio_per_pigment = max_total_pigment / n_pigments
    
    if n_pigments == 3:
        # Stage 1: Coarse grid search
        best_coarse_ratios = None
        for p1_ratio in np.arange(0.02, max_ratio_per_pigment * 2.5, coarse_step):
            for p2_ratio in np.arange(0.02, max_ratio_per_pigment * 2.5, coarse_step):
                remaining = max_total_pigment - p1_ratio - p2_ratio
                if remaining < 0.02:
                    continue
                p3_ratio = min(max_ratio_per_pigment * 2.5, remaining)
                if p3_ratio < 0.02:
                    continue
                
                white_ratio = 1.0 - p1_ratio - p2_ratio - p3_ratio
                if white_ratio < 0.3:  # Keep at least 30% white
                    continue
                
                # Get actual paint colors at these ratios (for calibrated paints)
                paint_labs_at_ratio = []
                for i, (cal, hex_rgb, ratio) in enumerate(zip(
                    paint_calibrations, 
                    paint_hex_colors, 
                    [p1_ratio, p2_ratio, p3_ratio]
                )):
                    if cal:
                        # Use calibrated color at this specific ratio
                        lab = interpolate_lab_from_calibration(cal, ratio)
                        if lab:
                            paint_labs_at_ratio.append(lab)
                        else:
                            # Fallback to hex
                            if hex_rgb:
                                paint_labs_at_ratio.append(rgb_to_lab(hex_rgb))
                            else:
                                continue
                    else:
                        # Use hex color (uncalibrated)
                        if hex_rgb:
                            paint_labs_at_ratio.append(rgb_to_lab(hex_rgb))
                        else:
                            continue
                
                if len(paint_labs_at_ratio) != 3:
                    continue
                
                # Blend the colors using actual ratios
                # White is approximately Lab(100, 0, 0) but we use 0.9 factor for mixing
                blended_lab = [
                    paint_labs_at_ratio[0][0] * p1_ratio + paint_labs_at_ratio[1][0] * p2_ratio + paint_labs_at_ratio[2][0] * p3_ratio + 100.0 * white_ratio * 0.9,
                    paint_labs_at_ratio[0][1] * p1_ratio + paint_labs_at_ratio[1][1] * p2_ratio + paint_labs_at_ratio[2][1] * p3_ratio,
                    paint_labs_at_ratio[0][2] * p1_ratio + paint_labs_at_ratio[1][2] * p2_ratio + paint_labs_at_ratio[2][2] * p3_ratio
                ]
                
                error = delta_e_lab(target_lab, blended_lab)
                if calibrated_count < n_pigments:
                    error += (n_pigments - calibrated_count) * 3.0  # Penalty for uncalibrated
                
                if error < best_error:
                    best_error = error
                    best_coarse_ratios = [p1_ratio, p2_ratio, p3_ratio]
        
        # Stage 2: Fine refinement around best coarse result
        if best_coarse_ratios:
            refine_step = 0.005  # Much finer step for refinement
            refine_range = 0.03  # Search within ±3% of best coarse result
            
            for p1_ratio in np.arange(
                max(0.02, best_coarse_ratios[0] - refine_range),
                min(max_ratio_per_pigment * 2.5, best_coarse_ratios[0] + refine_range + refine_step),
                refine_step
            ):
                for p2_ratio in np.arange(
                    max(0.02, best_coarse_ratios[1] - refine_range),
                    min(max_ratio_per_pigment * 2.5, best_coarse_ratios[1] + refine_range + refine_step),
                    refine_step
                ):
                    remaining = max_total_pigment - p1_ratio - p2_ratio
                    if remaining < 0.02:
                        continue
                    p3_ratio = min(max_ratio_per_pigment * 2.5, remaining)
                    if p3_ratio < 0.02:
                        continue
                    
                    white_ratio = 1.0 - p1_ratio - p2_ratio - p3_ratio
                    if white_ratio < 0.3:
                        continue
                    
                    # Get actual paint colors at these ratios (for calibrated paints)
                    paint_labs_at_ratio = []
                    for i, (cal, hex_rgb, ratio) in enumerate(zip(
                        paint_calibrations, 
                        paint_hex_colors, 
                        [p1_ratio, p2_ratio, p3_ratio]
                    )):
                        if cal:
                            # Use calibrated color at this specific ratio
                            lab = interpolate_lab_from_calibration(cal, ratio)
                            if lab:
                                paint_labs_at_ratio.append(lab)
                            else:
                                # Fallback to hex
                                if hex_rgb:
                                    paint_labs_at_ratio.append(rgb_to_lab(hex_rgb))
                                else:
                                    break
                        else:
                            # Use hex color (uncalibrated)
                            if hex_rgb:
                                paint_labs_at_ratio.append(rgb_to_lab(hex_rgb))
                            else:
                                break
                    
                    if len(paint_labs_at_ratio) != 3:
                        continue
                    
                    # Blend the colors using actual ratios
                    blended_lab = [
                        paint_labs_at_ratio[0][0] * p1_ratio + paint_labs_at_ratio[1][0] * p2_ratio + paint_labs_at_ratio[2][0] * p3_ratio + 100.0 * white_ratio * 0.9,
                        paint_labs_at_ratio[0][1] * p1_ratio + paint_labs_at_ratio[1][1] * p2_ratio + paint_labs_at_ratio[2][1] * p3_ratio,
                        paint_labs_at_ratio[0][2] * p1_ratio + paint_labs_at_ratio[1][2] * p2_ratio + paint_labs_at_ratio[2][2] * p3_ratio
                    ]
                    
                    error = delta_e_lab(target_lab, blended_lab)
                    if calibrated_count < n_pigments:
                        error += (n_pigments - calibrated_count) * 3.0
                    
                    if error < best_error:
                        best_error = error
                        best_recipe = {
                            'pigment_ids': paint_ids,
                            'pigment_ratios': [p1_ratio, p2_ratio, p3_ratio],
                            'white_ratio': white_ratio,
                            'error': best_error,
                            'type': 'three_pigment',
                            'uncalibrated': calibrated_count < n_pigments
                        }
        
        # If refinement didn't find better, use coarse result
        if best_coarse_ratios and not best_recipe:
            white_ratio = 1.0 - sum(best_coarse_ratios)
            best_recipe = {
                'pigment_ids': paint_ids,
                'pigment_ratios': best_coarse_ratios,
                'white_ratio': white_ratio,
                'error': best_error,
                'type': 'three_pigment',
                'uncalibrated': calibrated_count < n_pigments
            }
    
    elif n_pigments == 4:
        # Stage 1: Coarse grid search (slightly coarser for 4 pigments due to complexity)
        best_coarse_ratios = None
        for p1_ratio in np.arange(0.02, max_ratio_per_pigment * 2.5, coarse_step):
            for p2_ratio in np.arange(0.02, max_ratio_per_pigment * 2.5, coarse_step):
                for p3_ratio in np.arange(0.02, max_ratio_per_pigment * 2.5, coarse_step):
                    remaining = max_total_pigment - p1_ratio - p2_ratio - p3_ratio
                    if remaining < 0.02:
                        continue
                    p4_ratio = min(max_ratio_per_pigment * 2.5, remaining)
                    if p4_ratio < 0.02:
                        continue
                    
                    white_ratio = 1.0 - p1_ratio - p2_ratio - p3_ratio - p4_ratio
                    if white_ratio < 0.3:  # Keep at least 30% white
                        continue
                    
                    # Get actual paint colors at these ratios (for calibrated paints)
                    paint_labs_at_ratio = []
                    for i, (cal, hex_rgb, ratio) in enumerate(zip(
                        paint_calibrations, 
                        paint_hex_colors, 
                        [p1_ratio, p2_ratio, p3_ratio, p4_ratio]
                    )):
                        if cal:
                            # Use calibrated color at this specific ratio
                            lab = interpolate_lab_from_calibration(cal, ratio)
                            if lab:
                                paint_labs_at_ratio.append(lab)
                            else:
                                # Fallback to hex
                                if hex_rgb:
                                    paint_labs_at_ratio.append(rgb_to_lab(hex_rgb))
                                else:
                                    continue
                        else:
                            # Use hex color (uncalibrated)
                            if hex_rgb:
                                paint_labs_at_ratio.append(rgb_to_lab(hex_rgb))
                            else:
                                continue
                    
                    if len(paint_labs_at_ratio) != 4:
                        continue
                    
                    # Blend the colors using actual ratios
                    blended_lab = [
                        paint_labs_at_ratio[0][0] * p1_ratio + paint_labs_at_ratio[1][0] * p2_ratio + 
                        paint_labs_at_ratio[2][0] * p3_ratio + paint_labs_at_ratio[3][0] * p4_ratio + 100.0 * white_ratio * 0.9,
                        paint_labs_at_ratio[0][1] * p1_ratio + paint_labs_at_ratio[1][1] * p2_ratio + 
                        paint_labs_at_ratio[2][1] * p3_ratio + paint_labs_at_ratio[3][1] * p4_ratio,
                        paint_labs_at_ratio[0][2] * p1_ratio + paint_labs_at_ratio[1][2] * p2_ratio + 
                        paint_labs_at_ratio[2][2] * p3_ratio + paint_labs_at_ratio[3][2] * p4_ratio
                    ]
                    
                    error = delta_e_lab(target_lab, blended_lab)
                    if calibrated_count < n_pigments:
                        error += (n_pigments - calibrated_count) * 4.0  # Penalty for uncalibrated
                    
                    if error < best_error:
                        best_error = error
                        best_coarse_ratios = [p1_ratio, p2_ratio, p3_ratio, p4_ratio]
        
        # Stage 2: Fine refinement around best coarse result
        if best_coarse_ratios:
            refine_step = 0.005  # Much finer step for refinement
            refine_range = 0.03  # Search within ±3% of best coarse result
            
            for p1_ratio in np.arange(
                max(0.02, best_coarse_ratios[0] - refine_range),
                min(max_ratio_per_pigment * 2.5, best_coarse_ratios[0] + refine_range + refine_step),
                refine_step
            ):
                for p2_ratio in np.arange(
                    max(0.02, best_coarse_ratios[1] - refine_range),
                    min(max_ratio_per_pigment * 2.5, best_coarse_ratios[1] + refine_range + refine_step),
                    refine_step
                ):
                    for p3_ratio in np.arange(
                        max(0.02, best_coarse_ratios[2] - refine_range),
                        min(max_ratio_per_pigment * 2.5, best_coarse_ratios[2] + refine_range + refine_step),
                        refine_step
                    ):
                        remaining = max_total_pigment - p1_ratio - p2_ratio - p3_ratio
                        if remaining < 0.02:
                            continue
                        p4_ratio = min(max_ratio_per_pigment * 2.5, remaining)
                        if p4_ratio < 0.02:
                            continue
                        
                        white_ratio = 1.0 - p1_ratio - p2_ratio - p3_ratio - p4_ratio
                        if white_ratio < 0.3:
                            continue
                        
                        # Blend the colors
                        blended_lab = [
                            paint_labs[0][0] * p1_ratio + paint_labs[1][0] * p2_ratio + 
                            paint_labs[2][0] * p3_ratio + paint_labs[3][0] * p4_ratio + 100.0 * white_ratio * 0.9,
                            paint_labs[0][1] * p1_ratio + paint_labs[1][1] * p2_ratio + 
                            paint_labs[2][1] * p3_ratio + paint_labs[3][1] * p4_ratio,
                            paint_labs[0][2] * p1_ratio + paint_labs[1][2] * p2_ratio + 
                            paint_labs[2][2] * p3_ratio + paint_labs[3][2] * p4_ratio
                        ]
                        
                        error = delta_e_lab(target_lab, blended_lab)
                        if calibrated_count < n_pigments:
                            error += (n_pigments - calibrated_count) * 4.0
                        
                        if error < best_error:
                            best_error = error
                            best_recipe = {
                                'pigment_ids': paint_ids,
                                'pigment_ratios': [p1_ratio, p2_ratio, p3_ratio, p4_ratio],
                                'white_ratio': white_ratio,
                                'error': best_error,
                                'type': 'four_pigment',
                                'uncalibrated': calibrated_count < n_pigments
                            }
        
        # If refinement didn't find better, use coarse result
        if best_coarse_ratios and not best_recipe:
            white_ratio = 1.0 - sum(best_coarse_ratios)
            best_recipe = {
                'pigment_ids': paint_ids,
                'pigment_ratios': best_coarse_ratios,
                'white_ratio': white_ratio,
                'error': best_error,
                'type': 'four_pigment',
                'uncalibrated': calibrated_count < n_pigments
            }
    else:
        # Reject any other number of pigments (including 2)
        return None
    
    return best_recipe


def generate_recipes_for_palette(session_id: str, palette: List[Dict], library_group: str = "default") -> List[Dict]:
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
    
    for color in palette:
        target_rgb = color['rgb']
        target_lab = rgb_to_lab([float(c) for c in target_rgb])
        
        # Check if target is essentially achromatic (gray/white/black)
        target_is_achromatic = False
        r, g, b = target_rgb
        if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            max_channel = max(r, g, b)
            min_channel = min(r, g, b)
            if max_channel - min_channel < 40:  # Low saturation = gray
                target_is_achromatic = True
        
        # Try one-pigment recipes first (prefer calibrated, fallback to approximate)
        # Only use colored paints for colored targets
        search_paints = colored_paints if not target_is_achromatic else base_paints
        best_one_pigment = None
        best_one_error = float('inf')
        
        for paint in search_paints:
            paint_hex = paint.get('hex_approx', '')
            recipe = find_best_one_pigment_recipe(target_lab, paint['id'], paint_hex)
            if recipe:
                # Add penalty if using achromatic paint for colored target
                if not target_is_achromatic and is_achromatic(paint):
                    recipe['error'] += 20.0  # Heavy penalty
                
                if recipe['error'] < best_one_error:
                    best_one_error = recipe['error']
                    best_one_pigment = recipe
        
        # Try multi-pigment recipes - ALWAYS use at least 3 colors (no 2-color fallback)
        best_multi_pigment = None
        best_multi_error = float('inf')
        best_pigment_count = 0  # Track how many pigments in best recipe
        
        # Strategy: Try MORE pigments first, prefer recipes with more colors
        # Try 4 pigments first if available (most flexibility)
        if len(search_paints) >= 4:
            for i, paint1 in enumerate(search_paints):
                for j, paint2 in enumerate(search_paints[i+1:], i+1):
                    for k, paint3 in enumerate(search_paints[j+1:], j+1):
                        for paint4 in search_paints[k+1:]:
                            # Don't allow too many achromatic paints
                            if not target_is_achromatic:
                                achromatic_count = sum([is_achromatic(p) for p in [paint1, paint2, paint3, paint4]])
                                if achromatic_count >= 2:
                                    continue
                            
                            recipe = find_best_multi_pigment_recipe(
                                target_lab,
                                [paint1['id'], paint2['id'], paint3['id'], paint4['id']],
                                [paint1.get('hex_approx', ''), paint2.get('hex_approx', ''), 
                                 paint3.get('hex_approx', ''), paint4.get('hex_approx', '')]
                            )
                            if recipe:
                                # Add penalty if using achromatic paints
                                if not target_is_achromatic:
                                    for paint in [paint1, paint2, paint3, paint4]:
                                        if is_achromatic(paint):
                                            recipe['error'] += 8.0
                                
                                # Prefer 4-pigment recipes: accept if error is similar (within 2.0) or better
                                adjusted_error = recipe['error']
                                if best_pigment_count < 4:
                                    # If we don't have a 4-pigment recipe yet, or this is better, use it
                                    if best_multi_pigment is None or adjusted_error < best_multi_error + 2.0:
                                        best_multi_error = adjusted_error
                                        best_multi_pigment = recipe
                                        best_pigment_count = 4
                                elif adjusted_error < best_multi_error:
                                    # If we already have 4-pigment, only replace if significantly better
                                    best_multi_error = adjusted_error
                                    best_multi_pigment = recipe
        
        # Try 3 pigments - REQUIRED minimum (no 2-color recipes)
        if len(search_paints) >= 3:
            for i, paint1 in enumerate(search_paints):
                for j, paint2 in enumerate(search_paints[i+1:], i+1):
                    for paint3 in search_paints[j+1:]:
                        # Don't allow too many achromatic paints for colored targets
                        if not target_is_achromatic:
                            achromatic_count = sum([is_achromatic(p) for p in [paint1, paint2, paint3]])
                            if achromatic_count >= 2:  # Don't allow 2+ achromatic in 3-pigment mix
                                continue
                        
                        recipe = find_best_multi_pigment_recipe(
                            target_lab,
                            [paint1['id'], paint2['id'], paint3['id']],
                            [paint1.get('hex_approx', ''), paint2.get('hex_approx', ''), paint3.get('hex_approx', '')]
                        )
                        if recipe:
                            # Add penalty if using achromatic paints
                            if not target_is_achromatic:
                                for paint in [paint1, paint2, paint3]:
                                    if is_achromatic(paint):
                                        recipe['error'] += 10.0
                            
                            # Use 3-pigment if we don't have 4-pigment, or if it's better
                            if best_pigment_count < 3:
                                if best_multi_pigment is None or recipe['error'] < best_multi_error:
                                    best_multi_error = recipe['error']
                                    best_multi_pigment = recipe
                                    best_pigment_count = 3
                            elif best_pigment_count == 3 and recipe['error'] < best_multi_error:
                                best_multi_error = recipe['error']
                                best_multi_pigment = recipe
        
        # NO 2-PIGMENT FALLBACK - we require at least 3 colors
        
        # Choose best recipe - REQUIRE multi-pigment (at least 3 colors)
        # Only use one-pigment if we couldn't generate any multi-pigment recipe
        if best_multi_pigment and best_pigment_count >= 3:
            # We have a valid 3+ pigment recipe - use it
            recipes.append({
                'palette_index': color['index'],
                'recipe': best_multi_pigment,
                'type': best_multi_pigment.get('type', 'multi_pigment')
            })
        elif best_one_pigment:
            # Fallback to one-pigment only if we couldn't generate 3+ pigment recipe
            # This should rarely happen if we have 3+ paints available
            recipes.append({
                'palette_index': color['index'],
                'recipe': best_one_pigment,
                'type': 'one_pigment'
            })
        else:
            # This shouldn't happen now, but just in case
            recipes.append({
                'palette_index': color['index'],
                'recipe': None,
                'error': 'Could not generate recipe (unexpected error)'
            })
    
    return recipes

