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
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Bump this any time the processing pipeline changes in a way
# that should invalidate previously cached masks/preview images.
PIPELINE_VERSION = "v7"

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
    """Remove tiny components and apply light morphological cleanup.
    
    Uses a conservative min_area_ratio range so small but real paint regions are kept.
    """
    # For colors with very low coverage, use a more lenient threshold
    if coverage > 0 and coverage < 0.5:
        min_area_ratio = min_area_ratio * 0.1

    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    total_area = mask.shape[0] * mask.shape[1]
    min_area = max(1, int(total_area * min_area_ratio))

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels_cc == label_id] = 255

    if np.sum(cleaned) == 0 and num_labels > 1:
        largest_label = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]
        for label_id in range(2, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] > largest_area:
                largest_area = stats[label_id, cv2.CC_STAT_AREA]
                largest_label = label_id
        if largest_area > 0:
            cleaned[labels_cc == largest_label] = 255

    # Light close only (no open) to avoid removing thin connections
    if np.sum(cleaned) > 0:
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


def calculate_lightness(rgb: List[int]) -> float:
    """Calculate relative luminance (lightness) from RGB values."""
    # Use standard relative luminance formula
    # Convert to 0-1 range first
    r, g, b = [c / 255.0 for c in rgb]
    # Relative luminance formula
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


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
        
        # Gamma scaling: early layers expand more, but cap layer 1 so it doesn't light the whole canvas
        scale = (1 - idx / max(1, N - 1)) ** gamma
        if idx == 0:
            scale *= 0.45  # First layer gets 45% of full radius to avoid over-expansion
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


def ensure_base_masks_complete_coverage(
    base_masks: Dict[int, np.ndarray],
    labels: np.ndarray,
    n_colors: int
) -> None:
    """Ensure every pixel with a label appears in that label's base mask (no gaps in pure view).
    Modifies base_masks in place. Any pixel where labels[y,x]==idx is forced into base_masks[idx],
    so the pure mask is the full quantized region for that color.
    """
    for idx in range(n_colors):
        if idx not in base_masks:
            continue
        label_region = (labels == idx).astype(np.uint8) * 255
        added = cv2.bitwise_and(label_region, cv2.bitwise_not(base_masks[idx]))
        add_count = np.sum(added > 0)
        if add_count > 0:
            base_masks[idx] = cv2.bitwise_or(base_masks[idx], label_region)
            logger.info(f"Base mask {idx}: added {add_count} pixels so pure view has no holes")


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


def fill_holes_covered_by_later_layers(
    expanded_masks: Dict[int, np.ndarray],
    order: List[int]
) -> Dict[int, np.ndarray]:
    """Fill holes in each layer where a later layer will paint there.
    
    Earlier layers get painted first; anything painted on top will cover them.
    So if the current layer has a hole (no paint) and a later layer will cover
    that pixel, we can fill the hole in the current layer to make painting easier
    (user can paint through—it gets covered anyway).
    
    Args:
        expanded_masks: Dictionary of palette index to mask (modified in place)
        order: List of palette indices in paint order (first = painted first)
    
    Returns:
        expanded_masks (same dict, modified)
    """
    if len(order) < 2:
        return expanded_masks
    shape = list(expanded_masks.values())[0].shape
    n = len(order)
    # Build union of all "later" masks for each position (from the end)
    acc = np.zeros(shape, dtype=np.uint8)
    later_unions: List[np.ndarray] = [np.zeros(shape, dtype=np.uint8) for _ in range(n)]
    for idx in range(n - 1, 0, -1):
        acc = cv2.bitwise_or(acc, expanded_masks[order[idx]])
        later_unions[idx - 1] = acc.copy()
    for idx in range(n - 1):
        palette_idx = order[idx]
        mask = expanded_masks[palette_idx]
        later = later_unions[idx]
        holes = (mask == 0).astype(np.uint8) * 255
        fillable = cv2.bitwise_and(holes, later)
        if np.sum(fillable) > 0:
            expanded_masks[palette_idx] = cv2.bitwise_or(mask, fillable)
            logger.info(f"Filled {np.sum(fillable)} hole pixels in layer (palette {palette_idx}) with later-layer coverage")
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
                     max_side: int, saturation_boost: float, detail_level: float) -> str:
    """Compute cache key from image hash and processing parameters."""
    with open(image_path, 'rb') as f:
        image_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    params = f"{PIPELINE_VERSION}_{n_colors}_{overpaint_mm:.2f}_{order_mode}_{max_side}_{saturation_boost:.2f}_{detail_level:.2f}"
    cache_key = f"{image_hash}_{params}"
    logger.info(f"Computed cache key: {cache_key}")
    return cache_key


def get_cache_dir(cache_key: str) -> Path:
    """Get cache directory for a cache key."""
    return MASK_CACHE_DIR / cache_key


def regenerate_pure_mask_from_labels(session_dir: Path, layer_index: int) -> bool:
    """Write layer_{layer_index}_pure_mask.png from labels.npy and order.json (exact quantized region, no gaps)."""
    labels_path = session_dir / "labels.npy"
    order_path = session_dir / "order.json"
    if not labels_path.exists() or not order_path.exists():
        return False
    try:
        labels = np.load(str(labels_path))
        with open(order_path) as f:
            order = json.load(f)
        if layer_index < 0 or layer_index >= len(order):
            return False
        palette_idx = order[layer_index]
        pure_mask = ((labels == palette_idx).astype(np.uint8)) * 255
        out_path = session_dir / f"layer_{layer_index}_pure_mask.png"
        cv2.imwrite(str(out_path), pure_mask)
        return True
    except Exception as e:
        logger.warning(f"regenerate_pure_mask_from_labels failed: {e}")
        return False


def pure_masks_partition_stats(pure_masks: List[np.ndarray]) -> Tuple[int, int]:
    """Return (missing_pixels, overlap_pixels) for a set of pure masks."""
    if not pure_masks:
        return 0, 0
    stack = np.stack([(m > 0).astype(np.uint8) for m in pure_masks], axis=0)
    per_pixel_count = np.sum(stack, axis=0)
    missing_pixels = int(np.sum(per_pixel_count == 0))
    overlap_pixels = int(np.sum(per_pixel_count > 1))
    return missing_pixels, overlap_pixels


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
        labels_path = cache_dir / "labels.npy"
        order_path = cache_dir / "order.json"
        logger.info(f"Metadata file exists: {metadata_path.exists()}")
        logger.info(f"Labels file exists: {labels_path.exists()}")
        logger.info(f"Order file exists: {order_path.exists()}")
        if metadata_path.exists() and labels_path.exists() and order_path.exists():
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
        
        # Copy preview, labels and order (for pure-mask regeneration)
        cached_preview = cache_dir / "preview.jpg"
        if cached_preview.exists():
            shutil.copy2(cached_preview, output_dir / "preview.jpg")
            logger.info("Copied preview from cache")
        else:
            logger.warning("Preview not found in cache")
        for name in ("labels.npy", "order.json"):
            src = cache_dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)
        
        # Load palette and cached order
        palette = metadata.get('palette', [])
        cached_order = metadata.get('order', [])
        
        if not palette:
            logger.error("No palette in cache metadata")
            return None
        
        # Reorder layers based on current order_mode
        order = order_layers(palette, order_mode)
        logger.info(f"Reordered layers for mode '{order_mode}': {order}")
        with open(output_dir / "order.json", "w") as f:
            json.dump(order, f)
        
        # Pure masks must come from labels/order to guarantee complete pure coverage.
        labels_npy = output_dir / "labels.npy"
        if not labels_npy.exists():
            logger.error("Cache invalid: labels.npy missing, cannot regenerate pure masks reliably")
            return None
        try:
            labels = np.load(str(labels_npy))
        except Exception as e:
            logger.error(f"Cache invalid: failed loading labels.npy: {e}")
            return None

        layers = []
        missing_masks = []
        regular_start_idx = 0
        pure_masks_for_validation: List[np.ndarray] = []

        for layer_idx, palette_idx in enumerate(order):
            cached_mask = cache_dir / f"palette_{palette_idx}_mask.png"
            if not cached_mask.exists():
                logger.warning(f"Cached mask not found for palette index {palette_idx}")
                missing_masks.append(palette_idx)
                continue

            output_layer_idx = regular_start_idx + layer_idx
            mask_path = output_dir / f'layer_{output_layer_idx}_mask.png'
            shutil.copy2(cached_mask, mask_path)

            # Pure mask: always regenerate from labels/order so pure mode is deterministic.
            pure_mask_path = output_dir / f'layer_{output_layer_idx}_pure_mask.png'
            pure_mask = ((labels == palette_idx).astype(np.uint8)) * 255
            cv2.imwrite(str(pure_mask_path), pure_mask)
            pure_masks_for_validation.append(pure_mask)
            pure_url = f'/api/sessions/{output_dir.name}/layer_{output_layer_idx}_pure_mask.png'

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.error(f"Failed to load mask from {mask_path}")
                continue

            for outline_style in ['thin', 'thick', 'glow']:
                outline = generate_outline(mask, outline_style)
                outline_path = output_dir / f'layer_{output_layer_idx}_outline_{outline_style}.png'
                cv2.imwrite(str(outline_path), cv2.cvtColor(outline, cv2.COLOR_RGBA2BGRA))

            layers.append({
                'layer_index': output_layer_idx,
                'palette_index': palette_idx,
                'mask_url': f'/api/sessions/{output_dir.name}/layer_{output_layer_idx}_mask.png',
                'mask_pure_url': pure_url,
                'outline_thin_url': f'/api/sessions/{output_dir.name}/layer_{output_layer_idx}_outline_thin.png',
                'outline_thick_url': f'/api/sessions/{output_dir.name}/layer_{output_layer_idx}_outline_thick.png',
                'outline_glow_url': f'/api/sessions/{output_dir.name}/layer_{output_layer_idx}_outline_glow.png'
            })
        
        if missing_masks:
            logger.error(f"Missing masks for palette indices: {missing_masks}")
            return None
        
        if not layers:
            logger.error("No layers loaded from cache")
            return None
        
        missing_pixels, overlap_pixels = pure_masks_partition_stats(pure_masks_for_validation)
        if missing_pixels > 0 or overlap_pixels > 0:
            logger.error(
                f"Cache invalid: pure masks do not partition canvas (missing={missing_pixels}, overlap={overlap_pixels})"
            )
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
        
        logger.info(f"Successfully loaded {len(layers)} layers from cache")
        return {
            'width': metadata.get('width'),
            'height': metadata.get('height'),
            'palette': palette,
            'order': order,
            'quantized_preview_url': f'/api/sessions/{output_dir.name}/preview.jpg',
            'layers': layers,
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
        
        # Copy preview, labels and order for pure-mask regeneration
        preview_src = output_dir / "preview.jpg"
        if preview_src.exists():
            shutil.copy2(preview_src, cache_dir / "preview.jpg")
            logger.info("Cached preview image")
        else:
            logger.warning("Preview image not found to cache")
        for name in ("labels.npy", "order.json"):
            src = output_dir / name
            if src.exists():
                shutil.copy2(src, cache_dir / name)
        
        cached_count = 0
        for layer in result['layers']:
            if layer.get('is_finished'):
                continue
            palette_idx = layer['palette_index']
            layer_idx = layer['layer_index']
            mask_src = output_dir / f"layer_{layer_idx}_mask.png"
            if mask_src.exists():
                shutil.copy2(mask_src, cache_dir / f"palette_{palette_idx}_mask.png")
                cached_count += 1
            pure_src = output_dir / f"layer_{layer_idx}_pure_mask.png"
            if pure_src.exists():
                shutil.copy2(pure_src, cache_dir / f"palette_{palette_idx}_pure_mask.png")
            else:
                logger.warning(f"Pure mask not found for layer {layer_idx}, palette {palette_idx}")

        metadata = {
            'width': result['width'],
            'height': result['height'],
            'palette': result['palette'],
            'order': result['order'],
        }
        metadata_path = cache_dir / "cache_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully cached {cached_count} masks to: {cache_dir}")
    except Exception as e:
        logger.error(f"Failed to save to cache: {e}", exc_info=True)


def process_image(
    image_path: str,
    output_dir: Path,
    n_colors: int,
    overpaint_mm: float,
    order_mode: str,
    max_side: int,
    saturation_boost: float = 1.0,
    detail_level: float = 0.5,
) -> Dict:
    """Main processing pipeline with caching support."""
    logger.info(f"process_image called: image_path={image_path}, n_colors={n_colors}, cache_dir={MASK_CACHE_DIR}")

    # Compute cache key (gradients removed from pipeline)
    cache_key = compute_cache_key(image_path, n_colors, overpaint_mm, order_mode,
                                  max_side, saturation_boost, detail_level)
    
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
    
    # Load image (BGR) and apply EXIF orientation so we start from the
    # same upright view that cameras/phones intend.
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Check if file exists and is a valid image format.")
    
    if image.size == 0:
        raise ValueError("Loaded image is empty")
    
    # Apply EXIF orientation (cv2.imread ignores EXIF by default)
    image = apply_exif_orientation(image, image_path)
    
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
    
    # Auto-rotate portrait images 90 degrees counter-clockwise so they fit the canvas/screen better
    h_raw, w_raw = image.shape[:2]
    if h_raw > w_raw:
        logger.info(f"Input image is portrait ({w_raw}x{h_raw}), rotating 90 degrees CCW for processing")
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Save an oriented copy of the original image (for projection viewer "original" toggle)
    oriented_original_path = output_dir / "original_oriented.jpg"
    try:
        cv2.imwrite(str(oriented_original_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        logger.warning(f"Failed to save oriented original image to {oriented_original_path}: {e}")
    
    # Step 1: Normalize
    normalized, scale = normalize_image(image, max_side)
    h, w = normalized.shape[:2]
    
    # Step 2: Quantize
    labels, quantized, palette = quantize_lab(normalized, n_colors, seed=42, saturation_boost=saturation_boost)

    # Save quantized preview
    preview_path = output_dir / 'preview.jpg'
    cv2.imwrite(str(preview_path), cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR))
    
    # Step 3: Clean masks (conservative so pure view has fewer gaps)
    # detail_level: 0.0 = high detail, 1.0 = low detail; min_area_ratio range kept small
    min_area_ratio = 0.00002 + (detail_level * 0.00038)  # Range: 0.00002 to 0.0004
    
    base_masks = {}
    for idx in range(n_colors):
        mask = (labels == idx).astype(np.uint8) * 255
        # Get coverage for this color to inform mask cleaning
        color_coverage = palette[idx]['coverage'] if idx < len(palette) else 0.0
        cleaned = clean_mask(mask, min_area_ratio=min_area_ratio, coverage=color_coverage)
        base_masks[idx] = cleaned

    ensure_base_masks_complete_coverage(base_masks, labels, n_colors)

    # Step 4: Order layers
    order = order_layers(palette, order_mode)
    
    # Step 5: Smart overpaint expansion
    expanded_masks = smart_overpaint_expansion(base_masks, order, overpaint_mm, max_side)
    
    # Step 5.5: Ensure complete coverage - fill any unpainted areas
    expanded_masks = ensure_complete_coverage(expanded_masks, order, quantized, labels, palette)
    
    # Step 5.6: Fill holes in each layer where a later layer will paint (paint-through for easier painting)
    expanded_masks = fill_holes_covered_by_later_layers(expanded_masks, order)

    # Save labels and order so pure masks can be regenerated on demand (no gaps)
    np.save(str(output_dir / "labels.npy"), labels)
    with open(output_dir / "order.json", "w") as f:
        json.dump(order, f)

    # Step 6: Generate outlines and save
    layers = []
    regular_start_idx = 0
    pure_masks_for_validation: List[np.ndarray] = []
    for layer_idx, palette_idx in enumerate(order):
        mask = expanded_masks[palette_idx]
        mask_path = output_dir / f'layer_{regular_start_idx + layer_idx}_mask.png'
        cv2.imwrite(str(mask_path), mask)
        
        # Pure mask: exact quantized region (labels == palette_idx) so pure view has no gaps
        pure_mask = ((labels == palette_idx).astype(np.uint8)) * 255
        pure_masks_for_validation.append(pure_mask)
        pure_mask_path = output_dir / f'layer_{regular_start_idx + layer_idx}_pure_mask.png'
        cv2.imwrite(str(pure_mask_path), pure_mask)
        
        # Generate outlines
        for outline_style in ['thin', 'thick', 'glow']:
            outline = generate_outline(mask, outline_style)
            outline_path = output_dir / f'layer_{regular_start_idx + layer_idx}_outline_{outline_style}.png'
            cv2.imwrite(str(outline_path), cv2.cvtColor(outline, cv2.COLOR_RGBA2BGRA))
        
        layers.append({
            'layer_index': regular_start_idx + layer_idx,
            'palette_index': palette_idx,
            'mask_url': f'/api/sessions/{output_dir.name}/layer_{regular_start_idx + layer_idx}_mask.png',
            'mask_pure_url': f'/api/sessions/{output_dir.name}/layer_{regular_start_idx + layer_idx}_pure_mask.png',
            'outline_thin_url': f'/api/sessions/{output_dir.name}/layer_{regular_start_idx + layer_idx}_outline_thin.png',
            'outline_thick_url': f'/api/sessions/{output_dir.name}/layer_{regular_start_idx + layer_idx}_outline_thick.png',
            'outline_glow_url': f'/api/sessions/{output_dir.name}/layer_{regular_start_idx + layer_idx}_outline_glow.png'
        })
    
    missing_pixels, overlap_pixels = pure_masks_partition_stats(pure_masks_for_validation)
    if missing_pixels > 0 or overlap_pixels > 0:
        raise ValueError(
            f"Pure mask invariant failed (missing={missing_pixels}, overlap={overlap_pixels})."
        )
    
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
    
    result = {
        'width': w,
        'height': h,
        'palette': palette,
        'order': order,
        'quantized_preview_url': f'/api/sessions/{output_dir.name}/preview.jpg',
        'layers': layers,
        'original_url': f'/api/sessions/{output_dir.name}/original_oriented.jpg',
    }
    
    # Save to cache
    cache_dir = get_cache_dir(cache_key)
    save_to_cache(cache_dir, output_dir, result)
    
    return result
