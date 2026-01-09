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


def load_library() -> Dict:
    """Load paint library from JSON."""
    if not LIBRARY_FILE.exists():
        return {"version": 1, "paints": []}
    with open(LIBRARY_FILE, 'r') as f:
        return json.load(f)


def save_library(data: Dict):
    """Save paint library to JSON."""
    atomic_write(LIBRARY_FILE, data)


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


def find_best_one_pigment_recipe(target_lab: List[float], paint_id: str) -> Optional[Dict]:
    """Find best mixing ratio for one pigment + white to match target Lab."""
    calibration_file = CALIBRATION_DIR / f"{paint_id}.json"
    if not calibration_file.exists():
        return None
    
    with open(calibration_file, 'r') as f:
        calibration = json.load(f)
    
    samples = calibration.get('samples', [])
    if not samples:
        return None
    
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
    
    if best_ratio is None:
        return None
    
    return {
        'pigment_id': paint_id,
        'pigment_ratio': best_ratio,
        'white_ratio': 1.0 - best_ratio,
        'error': best_error
    }


def find_best_two_pigment_recipe(target_lab: List[float], paint_id1: str, paint_id2: str) -> Optional[Dict]:
    """Find best mixing ratio for two pigments + white (approximation)."""
    cal1_file = CALIBRATION_DIR / f"{paint_id1}.json"
    cal2_file = CALIBRATION_DIR / f"{paint_id2}.json"
    
    if not cal1_file.exists() or not cal2_file.exists():
        return None
    
    with open(cal1_file, 'r') as f:
        cal1 = json.load(f)
    with open(cal2_file, 'r') as f:
        cal2 = json.load(f)
    
    # Coarse grid search
    best_error = float('inf')
    best_recipe = None
    
    for p1_ratio in np.arange(0.0, 0.31, 0.05):
        for p2_ratio in np.arange(0.0, 0.31, 0.05):
            if p1_ratio + p2_ratio > 0.5:  # Keep white dominant
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
                        'error': best_error
                    }
    
    return best_recipe


def generate_recipes_for_palette(session_id: str, palette: List[Dict]) -> List[Dict]:
    """Generate paint mixing recipes for each palette color."""
    library = load_library()
    paints = library.get('paints', [])
    
    recipes = []
    
    for color in palette:
        target_rgb = color['rgb']
        target_lab = rgb_to_lab([float(c) for c in target_rgb])
        
        # Try one-pigment recipes first
        best_one_pigment = None
        best_one_error = float('inf')
        
        for paint in paints:
            if paint.get('type') == 'base':
                recipe = find_best_one_pigment_recipe(target_lab, paint['id'])
                if recipe and recipe['error'] < best_one_error:
                    best_one_error = recipe['error']
                    best_one_pigment = recipe
        
        # Try two-pigment recipes (if one-pigment error is high)
        best_two_pigment = None
        if best_one_error > 5.0:  # Threshold for trying two pigments
            for i, paint1 in enumerate(paints):
                if paint1.get('type') != 'base':
                    continue
                for paint2 in paints[i+1:]:
                    if paint2.get('type') != 'base':
                        continue
                    recipe = find_best_two_pigment_recipe(target_lab, paint1['id'], paint2['id'])
                    if recipe and recipe['error'] < best_one_error:
                        best_two_pigment = recipe
                        break
                if best_two_pigment:
                    break
        
        # Choose best recipe
        if best_two_pigment and best_two_pigment['error'] < best_one_error:
            recipes.append({
                'palette_index': color['index'],
                'recipe': best_two_pigment,
                'type': 'two_pigment'
            })
        elif best_one_pigment:
            recipes.append({
                'palette_index': color['index'],
                'recipe': best_one_pigment,
                'type': 'one_pigment'
            })
        else:
            recipes.append({
                'palette_index': color['index'],
                'recipe': None,
                'error': 'No calibrated paints available'
            })
    
    return recipes

