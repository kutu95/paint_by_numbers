import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from PIL import Image
import uuid
import logging
import hashlib
import shutil
import json
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Mask cache directory
MASK_CACHE_DIR = Path(__file__).parent.parent / "data" / "mask_cache"
try:
    MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Mask cache directory initialized at: {MASK_CACHE_DIR}")
    logger.info(f"Cache directory exists: {MASK_CACHE_DIR.exists()}")
    logger.info(f"Cache directory is writable: {MASK_CACHE_DIR.is_dir()}")
except Exception as e:
    logger.error(f"Failed to create mask cache directory at {MASK_CACHE_DIR}: {e}")


def apply_exif_orientation(image: np.ndarray, image_path: str) -> np.ndarray:
    """Apply EXIF orientation to image if present.
    
    Many cameras and phones save images with EXIF orientation tags.
    OpenCV's imread ignores these tags, so images can appear rotated incorrectly.
    This function reads EXIF data and applies the correct orientation.
    
    Args:
        image: Image as numpy array (BGR format from cv2.imread)
        image_path: Path to the image file
    
    Returns:
        Image with correct orientation applied (still BGR format)
    """
    try:
        # Load image with PIL to read EXIF data
        pil_image = Image.open(image_path)
        
        # Check if image has EXIF data
        if hasattr(pil_image, '_getexif') and pil_image._getexif() is not None:
            exif = pil_image._getexif()
            orientation = exif.get(274)  # EXIF tag 274 is orientation
            
            # If no orientation tag or already correct (1), return as-is
            if orientation is None or orientation == 1:
                return image
            
            # Convert PIL image to RGB numpy array
            pil_image = pil_image.convert('RGB')
            rgb_image = np.array(pil_image)
            
            # Apply orientation transformations
            # OpenCV uses BGR, so we need to convert RGB->BGR at the end
            if orientation == 2:
                # Flip horizontal
                rgb_image = np.fliplr(rgb_image)
            elif orientation == 3:
                # Rotate 180
                rgb_image = np.rot90(rgb_image, 2)
            elif orientation == 4:
                # Flip vertical
                rgb_image = np.flipud(rgb_image)
            elif orientation == 5:
                # Rotate 90 CCW and flip horizontal
                rgb_image = np.rot90(rgb_image, -1)
                rgb_image = np.fliplr(rgb_image)
            elif orientation == 6:
                # Rotate 90 CW
                rgb_image = np.rot90(rgb_image, -1)
            elif orientation == 7:
                # Rotate 90 CW and flip horizontal
                rgb_image = np.rot90(rgb_image, -1)
                rgb_image = np.fliplr(rgb_image)
            elif orientation == 8:
                # Rotate 90 CCW
                rgb_image = np.rot90(rgb_image, 1)
            
            # Convert RGB back to BGR for OpenCV
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            return bgr_image
        
        # No EXIF orientation tag, return original
        return image
        
    except Exception as e:
        # If EXIF reading fails for any reason, return original image
        # (better to show rotated image than fail completely)
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to apply EXIF orientation to {image_path}: {e}")
        return image


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


def clean_mask(mask: np.ndarray, min_area_ratio: float = 0.0002, coverage: float = 0.0) -> np.ndarray:
    """Remove tiny components and apply morphological operations.
    
    Args:
        mask: Binary mask to clean
        min_area_ratio: Minimum area ratio threshold for keeping components
        coverage: Coverage percentage of this color (0-100), used to adjust threshold for sparse colors
    """
    # For colors with very low coverage, use a more lenient threshold
    # This prevents removing all content from sparse colors
    if coverage > 0 and coverage < 0.5:
        # For very sparse colors (<0.5% coverage), use 10x more lenient threshold
        min_area_ratio = min_area_ratio * 0.1
    
    # Remove tiny components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    total_area = mask.shape[0] * mask.shape[1]
    min_area = max(1, int(total_area * min_area_ratio))  # Ensure at least 1 pixel
    
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    
    # If cleaning removed everything, keep the largest component even if below threshold
    if np.sum(cleaned) == 0 and num_labels > 1:
        # Find the largest component
        largest_label = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]
        for label_id in range(2, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] > largest_area:
                largest_area = stats[label_id, cv2.CC_STAT_AREA]
                largest_label = label_id
        # Keep at least the largest component
        if largest_area > 0:
            cleaned[labels == largest_label] = 255
    
    # Morphological close then open (only if we have content)
    if np.sum(cleaned) > 0:
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned


def calculate_lightness(rgb: List[int]) -> float:
    """Calculate relative luminance (lightness) from RGB values."""
    # Use standard relative luminance formula
    # Convert to 0-1 range first
    r, g, b = [c / 255.0 for c in rgb]
    # Relative luminance formula
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@dataclass
class GradientRegion:
    """Represents a gradient region that needs special ramp-based quantization."""
    id: str
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    steps_n: int
    direction: str  # 'top-to-bottom' for now
    transition_mode: str  # 'off', 'dither', 'feather-preview'
    transition_width_px: int
    seed: int
    stops: List[Dict]  # List of {index, hex_color, rgb, mask_bitmap}


def detect_gradient_regions(
    original_image: np.ndarray,
    quantized_labels: np.ndarray,
    palette: List[Dict],
    edge_density_threshold: float = 0.15,  # Increased from 0.05 - allow more edge density
    lightness_variation_threshold: float = 0.15,  # Decreased from 0.3 - detect smaller variations
    min_region_area_ratio: float = 0.02  # Decreased from 0.05 - detect smaller regions
) -> List[GradientRegion]:
    """Detect gradient regions in the image.
    
    Args:
        original_image: Original RGB image (will be downscaled for analysis)
        quantized_labels: K-means labels from quantization
        palette: Palette colors
        edge_density_threshold: Maximum edge density to be considered gradient (0-1)
        lightness_variation_threshold: Minimum lightness variation to be gradient (0-1)
        min_region_area_ratio: Minimum area ratio for a region to be analyzed
    
    Returns:
        List of detected gradient regions
    """
    h, w = original_image.shape[:2]
    total_pixels = h * w
    min_region_area = int(total_pixels * min_region_area_ratio)
    
    # Downscale original image for analysis (max 512px on long edge)
    analysis_max_side = 512
    if max(h, w) > analysis_max_side:
        scale = analysis_max_side / max(h, w)
        analysis_h = int(h * scale)
        analysis_w = int(w * scale)
        analysis_image = cv2.resize(original_image, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA)
        analysis_labels = cv2.resize(quantized_labels.astype(np.uint8), (analysis_w, analysis_h), interpolation=cv2.INTER_NEAREST)
    else:
        analysis_image = original_image
        analysis_labels = quantized_labels
        analysis_h, analysis_w = h, w
        scale = 1.0
    
    # Convert to Lab for lightness analysis
    lab_image = cv2.cvtColor(analysis_image, cv2.COLOR_RGB2LAB)
    lightness = lab_image[:, :, 0].astype(np.float32)  # L channel
    
    # Compute edge density using Sobel
    gray = cv2.cvtColor(analysis_image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_density = gradient_magnitude / 255.0  # Normalize to 0-1
    
    gradient_regions = []
    region_id = 0
    
    # Analyze each quantized region
    logger.info(f"Analyzing {len(palette)} palette regions, min_area={min_region_area} ({min_region_area_ratio*100:.1f}% of image)")
    analyzed_count = 0
    for palette_idx in range(len(palette)):
        # Get mask for this palette color
        region_mask = (analysis_labels == palette_idx).astype(np.uint8)
        region_area = np.sum(region_mask)
        
        if region_area < min_region_area:
            continue
        
        analyzed_count += 1
        
        # Compute bounding box
        coords = np.argwhere(region_mask > 0)
        if len(coords) == 0:
            continue
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox_h = y_max - y_min + 1
        bbox_w = x_max - x_min + 1
        
        # Extract region for analysis
        region_lightness = lightness[y_min:y_max+1, x_min:x_max+1]
        region_edge_density = edge_density[y_min:y_max+1, x_min:x_max+1]
        region_mask_crop = region_mask[y_min:y_max+1, x_min:x_max+1]
        
        # Mask to only analyze pixels in this region
        region_lightness_masked = region_lightness[region_mask_crop > 0]
        region_edge_density_masked = region_edge_density[region_mask_crop > 0]
        
        if len(region_lightness_masked) == 0:
            continue
        
        # Compute edge density (mean of edge density in region)
        mean_edge_density = np.mean(region_edge_density_masked)
        
        # Compute lightness variation along Y direction (top-to-bottom)
        # Sample lightness values along vertical lines in the region
        lightness_samples = []
        for x in range(bbox_w):
            col_lightness = region_lightness[:, x]
            col_mask = region_mask_crop[:, x]
            if np.sum(col_mask) > 0:
                col_lightness_masked = col_lightness[col_mask > 0]
                if len(col_lightness_masked) > 1:
                    lightness_range = np.max(col_lightness_masked) - np.min(col_lightness_masked)
                    lightness_samples.append(lightness_range)
        
        if len(lightness_samples) == 0:
            continue
        
        mean_lightness_variation = np.mean(lightness_samples) / 255.0  # Normalize to 0-1
        
        # Classify as gradient region
        is_gradient = (
            mean_edge_density < edge_density_threshold and
            mean_lightness_variation > lightness_variation_threshold
        )
        
        # Log details for all analyzed regions
        logger.info(f"Region {palette_idx}: area={region_area} ({region_area/total_pixels*100:.1f}%), "
                   f"bbox={bbox_w}x{bbox_h}, edge_density={mean_edge_density:.3f} (threshold={edge_density_threshold}), "
                   f"lightness_variation={mean_lightness_variation:.3f} (threshold={lightness_variation_threshold}), "
                   f"is_gradient={is_gradient}")
        
        if is_gradient:
            # Scale bounding box back to full resolution
            full_x_min = int(x_min / scale)
            full_y_min = int(y_min / scale)
            full_x_max = int(x_max / scale)
            full_y_max = int(y_max / scale)
            full_bbox_w = full_x_max - full_x_min + 1
            full_bbox_h = full_y_max - full_y_min + 1
            
            gradient_regions.append(GradientRegion(
                id=f"gradient_{region_id}",
                bounding_box=(full_x_min, full_y_min, full_bbox_w, full_bbox_h),
                steps_n=9,  # Default, will be configurable
                direction='top-to-bottom',
                transition_mode='dither',
                transition_width_px=25,
                seed=42 + region_id,  # Deterministic seed
                stops=[]
            ))
            region_id += 1
            logger.info(f"Detected gradient region {region_id-1} (palette_idx={palette_idx}) at ({full_x_min}, {full_y_min}), "
                       f"size {full_bbox_w}x{full_bbox_h}, edge_density={mean_edge_density:.3f}, "
                       f"lightness_variation={mean_lightness_variation:.3f}")
    
    logger.info(f"Gradient detection complete: analyzed {analyzed_count} regions, detected {len(gradient_regions)} gradient regions")
    return gradient_regions


def generate_gradient_ramp(
    image: np.ndarray,
    gradient_region: GradientRegion,
    n_steps: int = 9
) -> List[Dict]:
    """Generate gradient ramp steps for a gradient region.
    
    Args:
        image: RGB image (normalized resolution)
        gradient_region: Gradient region to process
        n_steps: Number of ramp steps (5-15)
    
    Returns:
        List of stops with {index, hex_color, rgb, mask_bitmap}
    """
    x, y, w, h = gradient_region.bounding_box
    img_h, img_w = image.shape[:2]
    
    # Clamp bounding box to image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w <= 0 or h <= 0:
        return []
    
    # Extract region from image
    region_image = image[y:y+h, x:x+w].copy()
    region_h, region_w = region_image.shape[:2]
    
    if region_h == 0 or region_w == 0:
        return []
    
    # Convert to Lab for better color analysis
    region_lab = cv2.cvtColor(region_image, cv2.COLOR_RGB2LAB)
    
    # Create bins for each step (top-to-bottom)
    bins = [[] for _ in range(n_steps)]
    bin_mask = np.zeros((region_h, region_w), dtype=np.int32)
    
    # Assign each pixel to a bin based on Y position
    for py in range(region_h):
        # Normalize Y position (0 at top, 1 at bottom)
        t = py / max(1, region_h - 1)
        bin_index = int(np.clip(np.floor(t * n_steps), 0, n_steps - 1))
        
        for px in range(region_w):
            bins[bin_index].append((py, px))
            bin_mask[py, px] = bin_index
    
    # Compute representative color for each bin using median in Lab space
    stops = []
    for step_idx in range(n_steps):
        if len(bins[step_idx]) == 0:
            continue
        
        # Collect Lab values for pixels in this bin
        lab_values = []
        for py, px in bins[step_idx]:
            lab_val = region_lab[py, px]
            lab_values.append(lab_val)
        
        if len(lab_values) == 0:
            continue
        
        # Compute median in Lab space
        lab_array = np.array(lab_values)
        median_lab = np.median(lab_array, axis=0).astype(np.uint8)
        
        # Convert median Lab back to RGB
        lab_3d = median_lab.reshape(1, 1, 3)
        rgb_3d = cv2.cvtColor(lab_3d, cv2.COLOR_LAB2RGB)
        rgb = np.clip(rgb_3d[0, 0], 0, 255).astype(int).tolist()
        
        # Create binary mask for this step
        step_mask = (bin_mask == step_idx).astype(np.uint8) * 255
        
        stops.append({
            'index': step_idx,
            'hex_color': '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2]),
            'rgb': rgb,
            'mask_bitmap': step_mask
        })
    
    return stops


def apply_dithered_transitions(
    stops: List[Dict],
    transition_width_px: int,
    seed: int
) -> List[Dict]:
    """Apply dithered transitions between ramp steps.
    
    Args:
        stops: List of stops with mask_bitmap
        transition_width_px: Width of transition band in pixels
        seed: Random seed for deterministic dithering
    
    Returns:
        Updated stops with dithered masks
    """
    if len(stops) < 2 or transition_width_px <= 0:
        return stops
    
    updated_stops = []
    
    for i in range(len(stops)):
        current_mask = stops[i]['mask_bitmap'].copy()
        
        # Apply transition with next step (for top-to-bottom, next is below)
        if i < len(stops) - 1:
            next_mask = stops[i+1]['mask_bitmap']
            transition_mask = create_transition_band(
                current_mask, next_mask, transition_width_px, seed + i * 1000
            )
            # Add transition pixels to current mask
            current_mask = np.maximum(current_mask, transition_mask)
            # Remove transition pixels from next mask to avoid overlap
            stops[i+1]['mask_bitmap'] = np.minimum(
                stops[i+1]['mask_bitmap'],
                (255 - transition_mask).astype(np.uint8)
            )
        
        updated_stop = stops[i].copy()
        updated_stop['mask_bitmap'] = current_mask
        updated_stops.append(updated_stop)
    
    return updated_stops


def create_transition_band(
    step_mask: np.ndarray,
    next_step_mask: np.ndarray,
    width_px: int,
    seed: int
) -> np.ndarray:
    """Create a dithered transition band between two adjacent ramp steps.
    
    For top-to-bottom gradients, creates a horizontal transition band.
    
    Args:
        step_mask: Current step mask
        next_step_mask: Next step mask
        width_px: Width of transition band in pixels
        seed: Random seed for deterministic dithering
    
    Returns:
        Binary mask for transition pixels to add to current step
    """
    h, w = step_mask.shape
    transition_mask = np.zeros((h, w), dtype=np.uint8)
    
    if width_px <= 0:
        return transition_mask
    
    # For top-to-bottom gradient, find the boundary between steps
    # Find the bottom edge of current step and top edge of next step
    rng = random.Random(seed)
    
    # Find the bottommost row with pixels in current step
    current_bottom = -1
    for y in range(h - 1, -1, -1):
        if np.any(step_mask[y, :] > 0):
            current_bottom = y
            break
    
    # Find the topmost row with pixels in next step
    next_top = h
    for y in range(h):
        if np.any(next_step_mask[y, :] > 0):
            next_top = y
            break
    
    if current_bottom < 0 or next_top >= h:
        return transition_mask
    
    # Create transition band in the gap between steps
    transition_start = max(0, current_bottom - width_px // 2)
    transition_end = min(h, next_top + width_px // 2)
    
    for y in range(transition_start, transition_end):
        for x in range(w):
            # Only process pixels that are in next step
            if next_step_mask[y, x] == 0:
                continue
            
            # Calculate distance from current step bottom
            dist_from_current = y - current_bottom if y >= current_bottom else current_bottom - y
            # Calculate distance from next step top
            dist_from_next = next_top - y if y <= next_top else y - next_top
            
            # Use the closer distance
            min_dist = min(dist_from_current, dist_from_next)
            
            if min_dist <= width_px:
                # Probability decreases as distance increases
                prob = 1.0 - (min_dist / width_px)
                
                # Use deterministic RNG based on position
                rng.seed(seed + y * w + x)
                if rng.random() < prob:
                    transition_mask[y, x] = 255
    
    return transition_mask


def order_layers(palette: List[Dict], order_mode: str) -> List[int]:
    """Order layers by coverage, lightness, or return manual order."""
    if order_mode == 'largest':
        sorted_palette = sorted(palette, key=lambda x: x['coverage'], reverse=True)
        return [p['index'] for p in sorted_palette]
    elif order_mode == 'smallest':
        sorted_palette = sorted(palette, key=lambda x: x['coverage'])
        return [p['index'] for p in sorted_palette]
    elif order_mode == 'lightest':
        # Sort by lightness (L value) - lightest first
        sorted_palette = sorted(palette, key=lambda x: calculate_lightness(x['rgb']), reverse=True)
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
    """Apply smart overpaint expansion with gamma scaling.
    
    Improved algorithm: Final edges (won't be overpainted by later layers) are pixel-perfect.
    Only internal edges (will be overpainted by later layers) get expansion.
    """
    # Estimate px_per_mm (assume longest side ≈ 1000mm)
    px_per_mm = max_side / 1000.0
    r_px_base = max(1, round(overpaint_mm * px_per_mm))
    
    mask_shape = list(base_masks.values())[0].shape
    painted_union = np.zeros(mask_shape, dtype=np.uint8)
    expanded_masks = {}
    N = len(order)
    
    # First pass: Build future painted union (what will be painted by all later layers)
    # This tells us which areas will be covered by future layers (internal edges)
    logger.info(f"Overpaint expansion: First pass - building future painted unions for {N} layers...")
    future_painted_unions = {}
    for idx in range(N):
        layer_num = idx + 1
        future_union = np.zeros(mask_shape, dtype=np.uint8)
        # Union of all layers after this one
        for future_idx in range(idx + 1, N):
            future_palette_idx = order[future_idx]
            future_union = cv2.bitwise_or(future_union, base_masks[future_palette_idx])
        future_painted_unions[idx] = future_union
        logger.info(f"Layer {layer_num}: First pass complete")
    logger.info("First pass done - all future unions built")
    
    # Second pass: Expand each layer, but only keep expansion on internal edges
    logger.info(f"Overpaint expansion: Second pass - expanding {N} layers...")
    for idx, palette_idx in enumerate(order):
        layer_num = idx + 1
        logger.info(f"Layer {layer_num}: Second pass - expanding...")
        base = base_masks[palette_idx].copy()
        
        # If base mask is empty, skip (this shouldn't happen after our fix, but just in case)
        if np.sum(base) == 0:
            expanded_masks[palette_idx] = base
            continue
        
        # Gamma scaling: early layers expand more
        scale = (1 - idx / max(1, N - 1)) ** gamma
        r_px = max(1, round(r_px_base * scale))
        
        # Dilate to create expansion
        kernel = np.ones((r_px * 2 + 1, r_px * 2 + 1), np.uint8)
        expanded = cv2.dilate(base, kernel, iterations=1)
        
        # Find the expansion area (pixels added by dilation)
        expansion_area = cv2.bitwise_and(expanded, cv2.bitwise_not(base))
        
        # Get future painted union for this layer (what will be painted by later layers)
        future_union = future_painted_unions[idx]
        
        # Keep expansion only on internal edges (where expansion overlaps with future layers)
        # Final edges (no overlap with future layers) get no expansion (pixel-perfect)
        internal_expansion = cv2.bitwise_and(expansion_area, future_union)
        
        # Combine: base + internal expansion
        refined_expanded = cv2.bitwise_or(base, internal_expansion)
        
        # Remove already painted areas
        paint_mask = cv2.bitwise_and(refined_expanded, cv2.bitwise_not(painted_union))
        
        # If mask becomes empty after removing painted areas, use the base mask
        # This ensures every color that exists in the palette has at least something to paint
        if np.sum(paint_mask) == 0:
            # Use the base mask, but remove only the areas that would overlap
            paint_mask = cv2.bitwise_and(base, cv2.bitwise_not(painted_union))
            # If still empty, just use the base mask (this ensures the layer isn't completely empty)
            if np.sum(paint_mask) == 0:
                paint_mask = base
        
        # Update painted union
        painted_union = cv2.bitwise_or(painted_union, paint_mask)
        
        expanded_masks[palette_idx] = paint_mask
        logger.info(f"Layer {layer_num}: Second pass done")
    
    logger.info("Second pass done - all layers expanded")
    return expanded_masks


def ensure_complete_coverage(
    expanded_masks: Dict[int, np.ndarray],
    order: List[int],
    quantized_image: np.ndarray,
    labels: np.ndarray,
    palette: List[Dict]
) -> Dict[int, np.ndarray]:
    """Ensure all pixels are covered by at least one layer.
    
    After overpaint expansion, some pixels might be missed due to:
    - Mask cleaning removing small components
    - Overpaint logic removing pixels that were supposed to be painted by earlier layers
    
    This function assigns any unpainted pixels to the appropriate layer based on
    the quantized color at that location.
    
    Args:
        expanded_masks: Dictionary of palette index to mask
        order: List of palette indices in paint order
        quantized_image: The quantized RGB image
        labels: The K-means labels for each pixel (0 to n_colors-1)
        palette: List of palette dictionaries with 'rgb' key
    
    Returns:
        Updated expanded_masks with all pixels covered
    """
    # Create a union of all painted areas
    painted_union = np.zeros_like(list(expanded_masks.values())[0], dtype=np.uint8)
    for mask in expanded_masks.values():
        painted_union = cv2.bitwise_or(painted_union, mask)
    
    # Find unpainted pixels
    unpainted_mask = cv2.bitwise_not(painted_union)
    unpainted_pixels = np.sum(unpainted_mask > 0)
    
    if unpainted_pixels == 0:
        # Already fully covered
        return expanded_masks
    
    # For each unpainted pixel, assign it to the layer matching its quantized color
    # This ensures every pixel is covered by the layer that represents its color
    for palette_idx in order:
        if palette_idx >= len(palette):
            continue
            
        mask = expanded_masks[palette_idx]
        
        # Find pixels that should belong to this color (from K-means labels)
        color_mask = (labels == palette_idx).astype(np.uint8) * 255
        
        # Find unpainted pixels that match this color
        missing_pixels = cv2.bitwise_and(color_mask, unpainted_mask)
        
        # Add these pixels to the current layer
        if np.sum(missing_pixels) > 0:
            expanded_masks[palette_idx] = cv2.bitwise_or(mask, missing_pixels)
            # Update unpainted mask
            unpainted_mask = cv2.bitwise_and(unpainted_mask, cv2.bitwise_not(missing_pixels))
    
    # If there are still unpainted pixels (shouldn't happen, but just in case),
    # assign them to the nearest layer by color similarity
    remaining_unpainted = np.sum(unpainted_mask > 0)
    if remaining_unpainted > 0:
        # Get coordinates of unpainted pixels
        unpainted_coords = np.argwhere(unpainted_mask > 0)
        
        # For each unpainted pixel, find the palette color that best matches its quantized color
        for y, x in unpainted_coords:
            # Get the quantized color at this location (RGB)
            pixel_color_rgb = quantized_image[y, x]  # Shape: (3,)
            
            # Find the palette index that best matches this color
            best_idx = order[0] if order else 0  # Default to first layer
            min_dist = float('inf')
            
            for palette_idx in order:
                if palette_idx >= len(palette):
                    continue
                    
                palette_color_rgb = np.array(palette[palette_idx]['rgb'])
                
                # Calculate Euclidean distance in RGB space
                color_dist = np.linalg.norm(pixel_color_rgb - palette_color_rgb)
                if color_dist < min_dist:
                    min_dist = color_dist
                    best_idx = palette_idx
            
            # Add this pixel to the best matching layer
            expanded_masks[best_idx][y, x] = 255
    
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


def compute_cache_key(image_path: str, n_colors: int, overpaint_mm: float, order_mode: str, 
                     max_side: int, saturation_boost: float, detail_level: float,
                     enable_gradients: bool = True, gradient_steps_n: int = 9,
                     gradient_transition_mode: str = 'dither', gradient_transition_width: int = 25) -> str:
    """Compute cache key from image hash and processing parameters."""
    # Read image file and compute hash
    with open(image_path, 'rb') as f:
        image_hash = hashlib.sha256(f.read()).hexdigest()[:16]  # Use first 16 chars
    
    # Create parameter string (normalize floats to avoid precision issues)
    params = f"{n_colors}_{overpaint_mm:.2f}_{order_mode}_{max_side}_{saturation_boost:.2f}_{detail_level:.2f}_{enable_gradients}_{gradient_steps_n}_{gradient_transition_mode}_{gradient_transition_width}"
    
    # Combine image hash with parameters
    cache_key = f"{image_hash}_{params}"
    logger.info(f"Computed cache key: {cache_key}")
    return cache_key


def get_cache_dir(cache_key: str) -> Path:
    """Get cache directory for a cache key."""
    return MASK_CACHE_DIR / cache_key


def check_mask_cache(cache_key: str) -> Optional[Path]:
    """Check if masks are cached for the given key.
    
    Returns:
        Path to cache directory if found, None otherwise
    """
    cache_dir = get_cache_dir(cache_key)
    logger.info(f"Checking cache at: {cache_dir}")
    logger.info(f"Cache dir exists: {cache_dir.exists()}")
    
    if cache_dir.exists():
        metadata_path = cache_dir / "cache_metadata.json"
        logger.info(f"Metadata file exists: {metadata_path.exists()}")
        if metadata_path.exists():
            logger.info(f"Cache HIT for key: {cache_key}")
            return cache_dir
    
    logger.info(f"Cache MISS for key: {cache_key}")
    return None


def load_from_cache(cache_dir: Path, output_dir: Path, order_mode: str) -> Optional[Dict]:
    """Load masks from cache and copy to output directory.
    
    Args:
        cache_dir: Path to cache directory
        output_dir: Path to output session directory
        order_mode: Layer ordering mode (may differ from cached order)
    
    Returns:
        Processing result dict if successful, None if cache is invalid
    """
    try:
        logger.info(f"Loading from cache: {cache_dir}")
        
        # Load metadata
        metadata_path = cache_dir / "cache_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata: {len(metadata.get('palette', []))} colors")
        
        # Copy preview
        cached_preview = cache_dir / "preview.jpg"
        if cached_preview.exists():
            shutil.copy2(cached_preview, output_dir / "preview.jpg")
            logger.info("Copied preview from cache")
        else:
            logger.warning("Preview not found in cache")
        
        # Load palette and cached order
        palette = metadata.get('palette', [])
        cached_order = metadata.get('order', [])
        
        if not palette:
            logger.error("No palette in cache metadata")
            return None
        
        # Reorder layers based on current order_mode
        order = order_layers(palette, order_mode)
        logger.info(f"Reordered layers for mode '{order_mode}': {order}")
        
        # Copy masks and generate outlines for the new order
        layers = []
        missing_masks = []
        for layer_idx, palette_idx in enumerate(order):
            # Find the cached mask file (by palette index, not layer index)
            cached_mask = cache_dir / f"palette_{palette_idx}_mask.png"
            if not cached_mask.exists():
                logger.warning(f"Cached mask not found for palette index {palette_idx}")
                missing_masks.append(palette_idx)
                continue
            
            # Copy mask to new layer index
            mask_path = output_dir / f'layer_{layer_idx}_mask.png'
            shutil.copy2(cached_mask, mask_path)
            
            # Load mask to generate outlines
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.error(f"Failed to load mask from {mask_path}")
                continue
            
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
        
        if missing_masks:
            logger.error(f"Missing masks for palette indices: {missing_masks}")
            return None
        
        if not layers:
            logger.error("No layers loaded from cache")
            return None
        
        # Add finished layer
        finished_layer_index = len(layers)
        layers.append({
            'layer_index': finished_layer_index,
            'palette_index': -1,
            'is_finished': True,
            'finished_url': f'/api/sessions/{output_dir.name}/preview.jpg',
            'mask_url': f'/api/sessions/{output_dir.name}/preview.jpg',
            'outline_thin_url': f'/api/sessions/{output_dir.name}/preview.jpg',
            'outline_thick_url': f'/api/sessions/{output_dir.name}/preview.jpg',
            'outline_glow_url': f'/api/sessions/{output_dir.name}/preview.jpg'
        })
        
        logger.info(f"Successfully loaded {len(layers)-1} layers from cache")
        return {
            'width': metadata.get('width'),
            'height': metadata.get('height'),
            'palette': palette,
            'order': order,
            'quantized_preview_url': f'/api/sessions/{output_dir.name}/preview.jpg',
            'layers': layers
        }
    except Exception as e:
        logger.error(f"Failed to load from cache: {e}", exc_info=True)
        return None


def save_to_cache(cache_dir: Path, output_dir: Path, result: Dict):
    """Save processing results to cache.
    
    Args:
        cache_dir: Path to cache directory
        output_dir: Path to session output directory (source of files to cache)
        result: Processing result dictionary
    """
    try:
        logger.info(f"Saving to cache: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy preview
        preview_src = output_dir / "preview.jpg"
        if preview_src.exists():
            shutil.copy2(preview_src, cache_dir / "preview.jpg")
            logger.info("Cached preview image")
        else:
            logger.warning("Preview image not found to cache")
        
        # Save masks by palette index (not layer index) so they can be reordered
        cached_count = 0
        for layer in result['layers']:
            if layer.get('is_finished'):
                continue
            
            palette_idx = layer['palette_index']
            layer_idx = layer['layer_index']
            
            # Copy mask
            mask_src = output_dir / f"layer_{layer_idx}_mask.png"
            if mask_src.exists():
                mask_dst = cache_dir / f"palette_{palette_idx}_mask.png"
                shutil.copy2(mask_src, mask_dst)
                cached_count += 1
            else:
                logger.warning(f"Mask not found for layer {layer_idx}, palette {palette_idx}")
        
        # Save metadata
        metadata = {
            'width': result['width'],
            'height': result['height'],
            'palette': result['palette'],
            'order': result['order']  # Save the order used when caching
        }
        metadata_path = cache_dir / "cache_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully cached {cached_count} masks to: {cache_dir}")
    except Exception as e:
        logger.error(f"Failed to save to cache: {e}", exc_info=True)


def process_gradient_regions(
    normalized_image: np.ndarray,
    labels: np.ndarray,
    palette: List[Dict],
    gradient_steps_n: int = 9,
    gradient_transition_mode: str = 'dither',
    gradient_transition_width: int = 25,
    enable_gradient_detection: bool = True
) -> Tuple[Dict[int, np.ndarray], List[Dict], List[GradientRegion]]:
    """Process gradient regions and generate ramp masks.
    
    Args:
        original_image: Full resolution original RGB image
        normalized_image: Normalized RGB image (used for quantization)
        labels: K-means labels from quantization
        palette: Palette colors
        gradient_steps_n: Number of steps in gradient ramps (5-15)
        gradient_transition_mode: 'off', 'dither', or 'feather-preview'
        gradient_transition_width: Width of transition bands in pixels
        enable_gradient_detection: Whether to detect and process gradients
    
    Returns:
        Tuple of (gradient_masks_dict, gradient_layers_list, gradient_regions_list)
        gradient_masks_dict: Maps gradient layer index to mask array
        gradient_layers_list: List of layer dicts for gradient ramps
        gradient_regions_list: List of detected gradient regions
    """
    gradient_masks = {}
    gradient_layers = []
    gradient_regions = []
    
    if not enable_gradient_detection:
        return gradient_masks, gradient_layers, gradient_regions
    
    # Detect gradient regions
    logger.info("Detecting gradient regions...")
    logger.info(f"Image size: {normalized_image.shape[0]}x{normalized_image.shape[1]}, "
                f"Palette size: {len(palette)}, Labels range: {labels.min()}-{labels.max()}")
    gradient_regions = detect_gradient_regions(
        normalized_image,
        labels,
        palette
    )
    
    if len(gradient_regions) == 0:
        logger.info("No gradient regions detected - check debug logs above for region analysis")
        return gradient_masks, gradient_layers, gradient_regions
    
    logger.info(f"Detected {len(gradient_regions)} gradient regions")
    
    # Process each gradient region
    gradient_layer_index = 0
    for grad_region in gradient_regions:
        # Update region settings
        grad_region.steps_n = gradient_steps_n
        grad_region.transition_mode = gradient_transition_mode
        grad_region.transition_width_px = gradient_transition_width
        
        # Generate ramp steps
        stops = generate_gradient_ramp(
            normalized_image,
            grad_region,
            n_steps=gradient_steps_n
        )
        
        if len(stops) == 0:
            continue
        
        # Apply transitions if enabled
        if gradient_transition_mode == 'dither':
            stops = apply_dithered_transitions(
                stops,
                gradient_transition_width,
                grad_region.seed
            )
        
        grad_region.stops = stops
        
        # Create full-resolution masks for each stop
        x, y, w_region, h_region = grad_region.bounding_box
        full_h, full_w = normalized_image.shape[:2]
        
        for stop in stops:
            # Stop mask is already at region size
            stop_mask_region = stop['mask_bitmap']
            stop_h, stop_w = stop_mask_region.shape
            
            # Ensure mask matches region dimensions (should already match)
            if stop_h != h_region or stop_w != w_region:
                stop_mask_region = cv2.resize(
                    stop_mask_region,
                    (w_region, h_region),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Create full image mask
            full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
            # Clamp coordinates to image bounds
            x_end = min(x + w_region, full_w)
            y_end = min(y + h_region, full_h)
            x_start = max(0, x)
            y_start = max(0, y)
            mask_w = x_end - x_start
            mask_h = y_end - y_start
            
            # Extract the relevant portion of the region mask
            mask_x_offset = max(0, -x)
            mask_y_offset = max(0, -y)
            region_mask_crop = stop_mask_region[mask_y_offset:mask_y_offset+mask_h, mask_x_offset:mask_x_offset+mask_w]
            
            full_mask[y_start:y_end, x_start:x_end] = region_mask_crop
            
            # Store mask
            gradient_masks[gradient_layer_index] = full_mask
            
            # Create layer entry
            gradient_layers.append({
                'layer_index': gradient_layer_index,
                'palette_index': -2,  # Special marker for gradient ramp
                'gradient_region_id': grad_region.id,
                'gradient_step_index': stop['index'],
                'hex': stop['hex_color'],
                'rgb': stop['rgb'],
                'is_gradient': True
            })
            
            gradient_layer_index += 1
        
        # Optional: Add glaze pass (unifying translucent layer)
        # This can be enabled via parameter
    
    logger.info(f"Generated {len(gradient_layers)} gradient ramp layers")
    return gradient_masks, gradient_layers, gradient_regions


def process_image(
    image_path: str,
    output_dir: Path,
    n_colors: int,
    overpaint_mm: float,
    order_mode: str,
    max_side: int,
    saturation_boost: float = 1.0,
    detail_level: float = 0.5,
    enable_gradients: bool = True,
    gradient_steps_n: int = 9,
    gradient_transition_mode: str = 'dither',
    gradient_transition_width: int = 25
) -> Dict:
    """Main processing pipeline with caching support."""
    logger.info(f"process_image called: image_path={image_path}, n_colors={n_colors}, cache_dir={MASK_CACHE_DIR}")
    
    # Compute cache key
    cache_key = compute_cache_key(image_path, n_colors, overpaint_mm, order_mode, 
                                  max_side, saturation_boost, detail_level,
                                  enable_gradients, gradient_steps_n, 
                                  gradient_transition_mode, gradient_transition_width)
    
    # Check cache first
    cached_dir = check_mask_cache(cache_key)
    if cached_dir:
        logger.info(f"Using cached masks for key: {cache_key}")
        result = load_from_cache(cached_dir, output_dir, order_mode)
        if result:
            return result
        logger.warning("Cache load failed, regenerating masks")
    
    # Cache miss or load failed - process normally
    logger.info(f"Processing image (cache miss): {cache_key}")
    
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
    
    # Step 2.5: Process gradient regions (if enabled)
    gradient_masks, gradient_layers, gradient_regions = process_gradient_regions(
        normalized,  # Use normalized image for analysis and ramp generation
        labels,
        palette,
        gradient_steps_n=gradient_steps_n,
        gradient_transition_mode=gradient_transition_mode,
        gradient_transition_width=gradient_transition_width,
        enable_gradient_detection=enable_gradients
    )
    
    # Save quantized preview
    preview_path = output_dir / 'preview.jpg'
    cv2.imwrite(str(preview_path), cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR))
    
    # Step 3: Clean masks
    # detail_level: 0.0 = high detail (keep small components), 1.0 = low detail (remove more)
    # Map detail_level (0-1) to min_area_ratio (0.00005 - 0.002)
    # Low values = preserve more detail, high values = remove more small components
    min_area_ratio = 0.00005 + (detail_level * 0.00195)  # Range: 0.00005 to 0.002
    
    base_masks = {}
    # Create exclusion mask for gradient regions (pixels that will be handled by gradients)
    gradient_exclusion_mask = np.zeros((h, w), dtype=np.uint8)
    if len(gradient_regions) > 0:
        for grad_region in gradient_regions:
            x, y, w_region, h_region = grad_region.bounding_box
            # Clamp to image bounds
            x_end = min(x + w_region, w)
            y_end = min(y + h_region, h)
            x_start = max(0, x)
            y_start = max(0, y)
            gradient_exclusion_mask[y_start:y_end, x_start:x_end] = 255
    
    for idx in range(n_colors):
        mask = (labels == idx).astype(np.uint8) * 255
        
        # Remove pixels that are in gradient regions
        if np.sum(gradient_exclusion_mask) > 0:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(gradient_exclusion_mask))
        
        # Get coverage for this color to inform mask cleaning
        color_coverage = palette[idx]['coverage'] if idx < len(palette) else 0.0
        cleaned = clean_mask(mask, min_area_ratio=min_area_ratio, coverage=color_coverage)
        base_masks[idx] = cleaned
    
    # Step 4: Order layers
    order = order_layers(palette, order_mode)
    
    # Step 5: Smart overpaint expansion
    expanded_masks = smart_overpaint_expansion(base_masks, order, overpaint_mm, max_side)
    
    # Step 5.5: Ensure complete coverage - fill any unpainted areas
    expanded_masks = ensure_complete_coverage(expanded_masks, order, quantized, labels, palette)
    
    # Step 6: Generate outlines and save
    layers = []
    
    # Save regular quantized layers
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
    
    # Save gradient ramp layers
    next_layer_idx = len(layers)
    for grad_layer in gradient_layers:
        grad_layer_idx = grad_layer['layer_index']
        if grad_layer_idx in gradient_masks:
            mask = gradient_masks[grad_layer_idx]
            mask_path = output_dir / f'layer_{next_layer_idx}_mask.png'
            cv2.imwrite(str(mask_path), mask)
            
            # Generate outlines
            for outline_style in ['thin', 'thick', 'glow']:
                outline = generate_outline(mask, outline_style)
                outline_path = output_dir / f'layer_{next_layer_idx}_outline_{outline_style}.png'
                cv2.imwrite(str(outline_path), cv2.cvtColor(outline, cv2.COLOR_RGBA2BGRA))
            
            layers.append({
                'layer_index': next_layer_idx,
                'palette_index': grad_layer.get('palette_index', -2),
                'gradient_region_id': grad_layer.get('gradient_region_id'),
                'gradient_step_index': grad_layer.get('gradient_step_index'),
                'hex': grad_layer.get('hex'),
                'rgb': grad_layer.get('rgb'),
                'is_gradient': True,
                'mask_url': f'/api/sessions/{output_dir.name}/layer_{next_layer_idx}_mask.png',
                'outline_thin_url': f'/api/sessions/{output_dir.name}/layer_{next_layer_idx}_outline_thin.png',
                'outline_thick_url': f'/api/sessions/{output_dir.name}/layer_{next_layer_idx}_outline_thick.png',
                'outline_glow_url': f'/api/sessions/{output_dir.name}/layer_{next_layer_idx}_outline_glow.png'
            })
            next_layer_idx += 1
    
    # Add final "finished" layer showing the complete quantized image
    finished_layer_index = len(layers)
    layers.append({
        'layer_index': finished_layer_index,
        'palette_index': -1,  # Special marker for finished layer
        'is_finished': True,
        'finished_url': f'/api/sessions/{output_dir.name}/preview.jpg',
        'mask_url': f'/api/sessions/{output_dir.name}/preview.jpg',  # For backward compatibility
        'outline_thin_url': f'/api/sessions/{output_dir.name}/preview.jpg',  # No outline for finished
        'outline_thick_url': f'/api/sessions/{output_dir.name}/preview.jpg',
        'outline_glow_url': f'/api/sessions/{output_dir.name}/preview.jpg'
    })
    
    # Serialize gradient regions for response
    gradient_regions_data = []
    for grad_region in gradient_regions:
        gradient_regions_data.append({
            'id': grad_region.id,
            'bounding_box': grad_region.bounding_box,
            'steps_n': grad_region.steps_n,
            'direction': grad_region.direction,
            'transition_mode': grad_region.transition_mode,
            'transition_width_px': grad_region.transition_width_px,
            'stops': [
                {
                    'index': stop['index'],
                    'hex_color': stop['hex_color'],
                    'rgb': stop['rgb']
                }
                for stop in grad_region.stops
            ]
        })
    
    result = {
        'width': w,
        'height': h,
        'palette': palette,
        'order': order,
        'quantized_preview_url': f'/api/sessions/{output_dir.name}/preview.jpg',
        'layers': layers,
        'gradient_regions': gradient_regions_data
    }
    
    # Save to cache
    cache_dir = get_cache_dir(cache_key)
    save_to_cache(cache_dir, output_dir, result)
    
    return result

