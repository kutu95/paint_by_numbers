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