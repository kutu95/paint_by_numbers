import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from PIL import Image
import uuid


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
    future_painted_unions = {}
    for idx in range(N):
        future_union = np.zeros(mask_shape, dtype=np.uint8)
        # Union of all layers after this one
        for future_idx in range(idx + 1, N):
            future_palette_idx = order[future_idx]
            future_union = cv2.bitwise_or(future_union, base_masks[future_palette_idx])
        future_painted_unions[idx] = future_union
    
    # Second pass: Expand each layer, but only keep expansion on internal edges
    for idx, palette_idx in enumerate(order):
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


def process_image(
    image_path: str,
    output_dir: Path,
    n_colors: int,
    overpaint_mm: float,
    order_mode: str,
    max_side: int,
    saturation_boost: float = 1.0,
    detail_level: float = 0.5
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
    # detail_level: 0.0 = high detail (keep small components), 1.0 = low detail (remove more)
    # Map detail_level (0-1) to min_area_ratio (0.00005 - 0.002)
    # Low values = preserve more detail, high values = remove more small components
    min_area_ratio = 0.00005 + (detail_level * 0.00195)  # Range: 0.00005 to 0.002
    
    base_masks = {}
    for idx in range(n_colors):
        mask = (labels == idx).astype(np.uint8) * 255
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
    
    return {
        'width': w,
        'height': h,
        'palette': palette,
        'order': order,
        'quantized_preview_url': f'/api/sessions/{output_dir.name}/preview.jpg',
        'layers': layers
    }

