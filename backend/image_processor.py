import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from PIL import Image
import uuid


def normalize_image(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Resize image to max_side while preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def quantize_lab(image: np.ndarray, n_colors: int, seed: int = 42, saturation_boost: float = 1.0) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Quantize image using Lab color space and k-means.
    
    Args:
        image: Input RGB image
        n_colors: Number of colors in palette
        seed: Random seed for reproducibility
        saturation_boost: Multiplier for saturation (1.0 = no change, >1.0 = more vibrant, <1.0 = less vibrant)
    """
    # Ensure image is valid (no NaN or inf values)
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        raise ValueError("Image contains invalid (NaN or Inf) values")
    
    # Apply saturation boost if requested
    if saturation_boost != 1.0:
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        # Boost saturation channel
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
        hsv = hsv.astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Convert to Lab
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    h, w = lab_image.shape[:2]
    pixels = lab_image.reshape(-1, 3).astype(np.float32)
    
    # Remove any NaN or Inf values from pixels
    valid_mask = ~(np.isnan(pixels).any(axis=1) | np.isinf(pixels).any(axis=1))
    if not np.all(valid_mask):
        # Replace invalid pixels with black
        pixels[~valid_mask] = [0, 128, 128]  # Black in Lab space
    
    # Ensure all values are finite and in valid range
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=255.0, neginf=0.0)
    pixels = np.clip(pixels, 0, 255)
    
    # Verify no NaN or Inf remain
    if np.any(np.isnan(pixels)) or np.any(np.isinf(pixels)):
        raise ValueError("Pixels still contain invalid values after cleaning")
    
    # K-means clustering - use random initialization to avoid NaN issues with k-means++
    # Random init is more robust when data has edge cases
    kmeans = KMeans(n_clusters=n_colors, random_state=seed, n_init=10, init='random')
    labels = kmeans.fit_predict(pixels)
    
    labels = labels.reshape(h, w)
    
    # Get palette centers in RGB
    # Lab centers from k-means are in float, need to convert properly
    lab_centers_float = kmeans.cluster_centers_
    rgb_centers = []
    for lab_center in lab_centers_float:
        # Ensure Lab values are in correct range for OpenCV
        # L: 0-100, a: -127 to 127, b: -127 to 127
        # But OpenCV stores as: L: 0-255, a: 0-255, b: 0-255
        lab_center_uint8 = np.clip(lab_center, 0, 255).astype(np.uint8)
        lab_3d = lab_center_uint8.reshape(1, 1, 3)
        rgb_3d = cv2.cvtColor(lab_3d, cv2.COLOR_LAB2RGB)
        # Clip RGB values to valid range and convert to int
        rgb_val = np.clip(rgb_3d[0, 0], 0, 255).astype(int)
        rgb_centers.append(rgb_val.tolist())
    
    # Calculate coverage
    total_pixels = h * w
    palette = []
    for idx in range(n_colors):
        coverage = np.sum(labels == idx) / total_pixels * 100
        palette.append({
            'index': idx,
            'rgb': rgb_centers[idx],
            'hex': '#{:02x}{:02x}{:02x}'.format(
                int(rgb_centers[idx][0]),
                int(rgb_centers[idx][1]),
                int(rgb_centers[idx][2])
            ),
            'coverage': round(coverage, 2)
        })
    
    # Create quantized preview
    quantized = np.zeros_like(image)
    for idx in range(n_colors):
        mask = labels == idx
        rgb_val = np.array(rgb_centers[idx], dtype=np.uint8)
        quantized[mask] = rgb_val
    
    return labels, quantized, palette


def clean_mask(mask: np.ndarray, min_area_ratio: float = 0.0002) -> np.ndarray:
    """Remove tiny components and apply morphological operations."""
    # Remove tiny components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    total_area = mask.shape[0] * mask.shape[1]
    min_area = int(total_area * min_area_ratio)
    
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    
    # Morphological close then open
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned


def order_layers(palette: List[Dict], order_mode: str) -> List[int]:
    """Order layers by coverage or return manual order."""
    if order_mode == 'largest':
        sorted_palette = sorted(palette, key=lambda x: x['coverage'], reverse=True)
        return [p['index'] for p in sorted_palette]
    elif order_mode == 'smallest':
        sorted_palette = sorted(palette, key=lambda x: x['coverage'])
        return [p['index'] for p in sorted_palette]
    else:  # manual
        return [p['index'] for p in palette]


def smart_overpaint_expansion(
    base_masks: Dict[int, np.ndarray],
    order: List[int],
    overpaint_mm: float,
    max_side: int,
    gamma: float = 1.5
) -> Dict[int, np.ndarray]:
    """Apply smart overpaint expansion with gamma scaling."""
    # Estimate px_per_mm (assume longest side ≈ 1000mm)
    px_per_mm = max_side / 1000.0
    r_px_base = max(1, round(overpaint_mm * px_per_mm))
    
    painted_union = np.zeros_like(list(base_masks.values())[0], dtype=np.uint8)
    expanded_masks = {}
    N = len(order)
    
    for idx, palette_idx in enumerate(order):
        base = base_masks[palette_idx].copy()
        
        # Gamma scaling: early layers expand more
        scale = (1 - idx / max(1, N - 1)) ** gamma
        r_px = max(1, round(r_px_base * scale))
        
        # Dilate
        kernel = np.ones((r_px * 2 + 1, r_px * 2 + 1), np.uint8)
        expanded = cv2.dilate(base, kernel, iterations=1)
        
        # Remove already painted areas
        paint_mask = cv2.bitwise_and(expanded, cv2.bitwise_not(painted_union))
        
        # Update painted union
        painted_union = cv2.bitwise_or(painted_union, paint_mask)
        
        expanded_masks[palette_idx] = paint_mask
    
    return expanded_masks


def generate_outline(mask: np.ndarray, style: str = 'thin') -> np.ndarray:
    """Generate outline overlay from mask."""
    if style == 'off':
        return np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    
    # Get edges
    if style == 'thin':
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        outline = cv2.subtract(dilated, mask)
    elif style == 'thick':
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        outline = cv2.subtract(dilated, mask)
    elif style == 'glow':
        # Soft glow effect
        blurred = cv2.GaussianBlur(mask, (15, 15), 0)
        outline = cv2.subtract(blurred, mask)
        outline = cv2.multiply(outline, 2)  # Brighten
        outline = np.clip(outline, 0, 255)
    else:
        outline = np.zeros_like(mask)
    
    # Convert to RGBA
    outline_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    outline_rgba[:, :, 0] = outline  # R
    outline_rgba[:, :, 1] = outline  # G
    outline_rgba[:, :, 2] = outline  # B
    outline_rgba[:, :, 3] = outline  # A
    
    return outline_rgba


def process_image(
    image_path: str,
    output_dir: Path,
    n_colors: int,
    overpaint_mm: float,
    order_mode: str,
    max_side: int,
    saturation_boost: float = 1.0
) -> Dict:
    """Main processing pipeline."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Check if file exists and is a valid image format.")
    
    if image.size == 0:
        raise ValueError("Loaded image is empty")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8 and in valid range
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        # Ensure values are in valid range
        image = np.clip(image, 0, 255)
    
    # Check for invalid values
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        raise ValueError("Image contains invalid (NaN or Inf) values after loading")
    
    # Step 1: Normalize
    normalized, scale = normalize_image(image, max_side)
    h, w = normalized.shape[:2]
    
    # Step 2: Quantize
    labels, quantized, palette = quantize_lab(normalized, n_colors, seed=42, saturation_boost=saturation_boost)
    
    # Save quantized preview
    preview_path = output_dir / 'preview.jpg'
    cv2.imwrite(str(preview_path), cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR))
    
    # Step 3: Clean masks
    base_masks = {}
    for idx in range(n_colors):
        mask = (labels == idx).astype(np.uint8) * 255
        cleaned = clean_mask(mask)
        base_masks[idx] = cleaned
    
    # Step 4: Order layers
    order = order_layers(palette, order_mode)
    
    # Step 5: Smart overpaint expansion
    expanded_masks = smart_overpaint_expansion(base_masks, order, overpaint_mm, max_side)
    
    # Step 6: Generate outlines and save
    layers = []
    for layer_idx, palette_idx in enumerate(order):
        mask = expanded_masks[palette_idx]
        mask_path = output_dir / f'layer_{layer_idx}_mask.png'
        cv2.imwrite(str(mask_path), mask)
        
        # Generate outlines
        for outline_style in ['thin', 'thick', 'glow']:
            outline = generate_outline(mask, outline_style)
            outline_path = output_dir / f'layer_{layer_idx}_outline_{outline_style}.png'
            cv2.imwrite(str(outline_path), cv2.cvtColor(outline, cv2.COLOR_RGBA2BGRA))
        
        layers.append({
            'layer_index': layer_idx,
            'palette_index': palette_idx,
            'mask_url': f'/api/sessions/{output_dir.name}/layer_{layer_idx}_mask.png',
            'outline_thin_url': f'/api/sessions/{output_dir.name}/layer_{layer_idx}_outline_thin.png',
            'outline_thick_url': f'/api/sessions/{output_dir.name}/layer_{layer_idx}_outline_thick.png',
            'outline_glow_url': f'/api/sessions/{output_dir.name}/layer_{layer_idx}_outline_glow.png'
        })
    
    return {
        'width': w,
        'height': h,
        'palette': palette,
        'order': order,
        'quantized_preview_url': f'/api/sessions/{output_dir.name}/preview.jpg',
        'layers': layers
    }

