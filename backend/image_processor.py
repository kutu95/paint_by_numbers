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

from project_store import SOURCE_ORIENTED, artifact_url, source_url

logger = logging.getLogger(__name__)

# Bump this any time the processing pipeline changes in a way
# that should invalidate previously cached masks/preview images.
PIPELINE_VERSION = "v29"

VALID_STYLE_PRESETS = frozenset({
    "none",
    "natural",  # alias for none
    "easy_painting",
    "poster",
    "bold",
    "expressive",
    "graphic",
    "portrait",
    "stipple",
    "sketch",
    "harmony",
})

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


def load_rgb_image_oriented(image_path: str) -> np.ndarray:
    """Load RGB uint8 image with EXIF orientation applied; no aspect-ratio rotation."""
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise ValueError(f"Could not read image: {image_path}")
    image = apply_exif_orientation(image, image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255)
    return image


def load_rgb_image_normalized(image_path: str, max_side: int) -> np.ndarray:
    """Load image from disk, apply EXIF/orientation, return RGB uint8 normalized to max_side."""
    image = load_rgb_image_oriented(image_path)
    normalized, _scale = normalize_image(image, max_side)
    return normalized


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        logger.warning("OpenCV face cascade failed to load; face detail boost disabled")
        _FACE_CASCADE = None
        return None
    _FACE_CASCADE = cascade
    return cascade


_EYE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_eye_cascade() -> Optional[cv2.CascadeClassifier]:
    global _EYE_CASCADE
    if _EYE_CASCADE is not None:
        return _EYE_CASCADE
    for name in ("haarcascade_eye_tree_eyeglasses.xml", "haarcascade_eye.xml"):
        path = cv2.data.haarcascades + name
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            _EYE_CASCADE = cascade
            return cascade
    logger.warning("OpenCV eye cascade failed to load; eye detail boost disabled")
    _EYE_CASCADE = None
    return None


def _detect_faces_xywh(image_rgb: np.ndarray) -> np.ndarray:
    """Return Nx4 face boxes (x, y, w, h) or empty array."""
    cascade = _get_face_cascade()
    if cascade is None:
        return np.zeros((0, 4), dtype=np.int32)
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    min_dim = max(24, int(min(h, w) * 0.06))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_dim, min_dim),
    )
    if faces is None or len(faces) == 0:
        return np.zeros((0, 4), dtype=np.int32)
    return np.asarray(faces, dtype=np.int32)


def detect_face_detail_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Float mask in [0, 1]: higher values keep more detail (likely portrait/face areas)."""
    h, w = image_rgb.shape[:2]
    faces = _detect_faces_xywh(image_rgb)
    mask = np.zeros((h, w), dtype=np.float32)
    if faces.size == 0:
        return mask

    for x, y, fw, fh in faces:
        cx = int(x + fw * 0.5)
        cy = int(y + fh * 0.52)
        rx = max(8, int(fw * 0.72))
        ry = max(8, int(fh * 0.92))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, thickness=-1)

    k = max(15, int(min(h, w) * 0.035)) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)


def detect_eye_detail_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Float mask in [0, 1]: full detail preserved on detected eyes (within faces)."""
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    eye_cascade = _get_eye_cascade()
    faces = _detect_faces_xywh(image_rgb)
    if eye_cascade is None or faces.size == 0:
        return mask

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    for x, y, fw, fh in faces:
        x, y, fw, fh = int(x), int(y), int(fw), int(fh)
        if fw < 16 or fh < 16:
            continue
        x2 = min(w, x + fw)
        y2 = min(h, y + fh)
        roi_gray = gray[y:y2, x:x2]
        if roi_gray.size == 0:
            continue
        eye_top = int(fh * 0.62)
        eye_gray = roi_gray[0:eye_top, :]
        if eye_gray.shape[0] < 8 or eye_gray.shape[1] < 8:
            continue
        min_eye = max(6, min(fw, fh) // 10)
        eyes = eye_cascade.detectMultiScale(
            eye_gray,
            scaleFactor=1.06,
            minNeighbors=2,
            minSize=(min_eye, min_eye),
        )
        found = False
        if eyes is not None and len(eyes) > 0:
            for ex, ey, ew, eh in eyes:
                cx = x + int(ex + ew * 0.5)
                cy = y + int(ey + eh * 0.5)
                rx = max(4, int(ew * 0.72))
                ry = max(3, int(eh * 0.85))
                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, thickness=-1)
                found = True
        if not found:
            # Estimate eye positions from face box when cascade misses (glasses, angle, etc.)
            for frac_x in (0.33, 0.67):
                cx = int(x + fw * frac_x)
                cy = int(y + fh * 0.36)
                rx = max(5, int(fw * 0.12))
                ry = max(4, int(fh * 0.07))
                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 0.92, thickness=-1)

    k = max(3, int(min(h, w) * 0.008)) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    if float(mask.max()) > 1e-6:
        mask = mask / float(mask.max())
    return np.clip(mask, 0.0, 1.0)


def _blur_figure_mask(detail_mask: np.ndarray) -> np.ndarray:
    """Widen and soften the protection mask so RGB/label transitions are gradual."""
    h, w = detail_mask.shape[:2]
    k = max(11, int(min(h, w) * 0.032)) | 1
    blurred = cv2.GaussianBlur(detail_mask.astype(np.float32), (k, k), 0)
    return np.clip(blurred, 0.0, 1.0)


def _figure_blend_weight(detail_mask: np.ndarray) -> np.ndarray:
    """Per-pixel weight toward original image (strong in face core, smooth falloff at edges)."""
    m = _blur_figure_mask(detail_mask)
    return np.clip(np.power(m, 0.82) * 1.12, 0.0, 1.0)


def _figure_label_core_mask(detail_mask: np.ndarray) -> np.ndarray:
    """Interior face/eyes: keep original-driven labels."""
    m = _blur_figure_mask(detail_mask)
    return m > 0.52


def _figure_label_preserve_mask(detail_mask: np.ndarray) -> np.ndarray:
    """Figure region where label median-filter must not run (avoids grey halo at hair/face edge)."""
    m = _blur_figure_mask(detail_mask)
    return m > 0.18


_HOG_DETECTOR: Optional[cv2.HOGDescriptor] = None


def _get_hog_person_detector() -> Optional[cv2.HOGDescriptor]:
    global _HOG_DETECTOR
    if _HOG_DETECTOR is not None:
        return _HOG_DETECTOR
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _HOG_DETECTOR = hog
        return hog
    except Exception as e:
        logger.warning("HOG person detector unavailable: %s", e)
        _HOG_DETECTOR = None
        return None


def detect_person_body_fill_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Float mask in [0, 1]: filled region covering detected people (for silhouette)."""
    h, w = image_rgb.shape[:2]
    fill = np.zeros((h, w), dtype=np.float32)
    hog = _get_hog_person_detector()
    if hog is None:
        return fill

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    min_h = max(64, int(h * 0.12))
    try:
        rects, _weights = hog.detectMultiScale(
            gray,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.04,
            hitThreshold=0,
        )
    except Exception:
        return fill

    if rects is None or len(rects) == 0:
        return fill

    for x, y, bw, bh in rects:
        x1 = max(0, int(x - bw * 0.08))
        y1 = max(0, int(y - bh * 0.06))
        x2 = min(w, int(x + bw * 1.08))
        y2 = min(h, int(y + bh * 1.04))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx = max(8, (x2 - x1) // 2)
        ry = max(8, (y2 - y1) // 2)
        cv2.ellipse(fill, (cx, cy), (rx, ry), 0, 0, 360, 1.0, thickness=-1)

    k = max(9, int(min(h, w) * 0.02)) | 1
    fill = cv2.GaussianBlur(fill, (k, k), 0)
    return np.clip(fill, 0.0, 1.0)


def _body_fill_from_faces(image_rgb: np.ndarray) -> np.ndarray:
    """Rough body region when HOG misses a portrait (face → torso estimate)."""
    face = detect_face_detail_mask(image_rgb)
    if not np.any(face > 0.2):
        return np.zeros_like(face)
    h, w = face.shape
    body = np.zeros((h, w), dtype=np.float32)
    face_u8 = (face > 0.35).astype(np.uint8) * 255
    num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(face_u8, connectivity=8)
    for comp_id in range(1, num_labels):
        x = stats[comp_id, cv2.CC_STAT_LEFT]
        y = stats[comp_id, cv2.CC_STAT_TOP]
        fw = stats[comp_id, cv2.CC_STAT_WIDTH]
        fh = stats[comp_id, cv2.CC_STAT_HEIGHT]
        cx = int(x + fw * 0.5)
        top = int(y + fh * 0.35)
        rx = max(8, int(fw * 0.95))
        ry = max(12, int(fh * 2.8))
        cv2.ellipse(body, (cx, top + ry // 2), (rx, ry), 0, 0, 360, 1.0, thickness=-1)
    k = max(11, int(min(h, w) * 0.025)) | 1
    body = cv2.GaussianBlur(body, (k, k), 0)
    return np.clip(body, 0.0, 1.0)


def build_body_outline_detail_mask(
    body_fill: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Float mask peaked on the person silhouette edge (accurate body outline)."""
    h, w = image_shape
    if body_fill is None or not np.any(body_fill > 0.08):
        return np.zeros((h, w), dtype=np.float32)

    fill_u8 = (body_fill >= 0.35).astype(np.uint8) * 255
    band_px = max(5, int(min(h, w) * 0.012)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px, band_px))
    dilated = cv2.dilate(fill_u8, kernel, iterations=1)
    eroded = cv2.erode(fill_u8, kernel, iterations=1)
    ring = cv2.subtract(dilated, eroded)
    k = max(5, band_px - 2) | 1
    ring = cv2.GaussianBlur(ring.astype(np.float32), (k, k), 0)
    if float(ring.max()) > 1e-6:
        ring = ring / float(ring.max())
    return np.clip(ring * 0.98, 0.0, 0.98)


@dataclass
class FigureDetailOptions:
    """Auto-detected regions to keep sharp through stylization and quantization."""

    enabled: bool = True
    eyes: bool = True
    face: bool = True
    body_outline: bool = True


def build_figure_detail_mask(
    image_rgb: np.ndarray,
    *,
    eyes: bool = True,
    face: bool = True,
    body_outline: bool = True,
    include_face_interior: Optional[bool] = None,
) -> np.ndarray:
    """Combine body-outline, face interior, and eye preservation masks."""
    if include_face_interior is not None and not face:
        face = bool(include_face_interior)

    h, w = image_rgb.shape[:2]
    detail = np.zeros((h, w), dtype=np.float32)

    body_fill = None
    if body_outline:
        body_fill = detect_person_body_fill_mask(image_rgb)
        if not np.any(body_fill > 0.08):
            body_fill = _body_fill_from_faces(image_rgb)
        if np.any(body_fill > 0.08):
            # Low weight: outline guides RGB blend only, not hard label preservation (avoids speckle halo).
            detail = np.maximum(
                detail,
                build_body_outline_detail_mask(body_fill, (h, w)) * 0.35,
            )

    if face:
        face_mask = detect_face_detail_mask(image_rgb)
        if np.any(face_mask > 0.05):
            detail = np.maximum(detail, face_mask)
    if eyes:
        eye_mask = detect_eye_detail_mask(image_rgb)
        if np.any(eye_mask > 0.05):
            detail = np.maximum(detail, eye_mask)

    return np.clip(detail, 0.0, 1.0)


def build_figure_detail_mask_from_options(
    image_rgb: np.ndarray,
    options: Optional[FigureDetailOptions],
    *,
    include_body_outline: bool = True,
) -> Optional[np.ndarray]:
    if options is None or not options.enabled:
        return None
    use_outline = options.body_outline and include_body_outline
    if not (options.eyes or options.face or use_outline):
        return None
    mask = build_figure_detail_mask(
        image_rgb,
        eyes=options.eyes,
        face=options.face,
        body_outline=use_outline,
    )
    if not np.any(mask > 0.05):
        return None
    return mask


def normalize_style_preset(style_preset: Optional[str]) -> str:
    """Return a valid preset id (defaults to none / classic pipeline)."""
    p = (style_preset or "none").strip().lower().replace("-", "_")
    if p in ("easy", "easy_paint"):
        p = "easy_painting"
    if p in ("classic", "original", "off"):
        p = "none"
    if p == "natural":
        p = "none"
    if p not in VALID_STYLE_PRESETS:
        return "none"
    return p


@dataclass
class ResolvedImageStyle:
    preset: str
    background_simplify: float
    label_smooth: float
    figure_detail: FigureDetailOptions
    posterize_levels: int = 0
    hue_shift_deg: float = 0.0
    extra_saturation: float = 1.0
    stipple: bool = False
    sketch: bool = False
    harmony_ab: bool = False

    @property
    def easy_painting(self) -> bool:
        """Legacy flag: any background simplification is active."""
        return self.background_simplify > 1e-6 or self.label_smooth > 1e-6

    @property
    def easy_simplify(self) -> float:
        return self.background_simplify


def _figure_options(
    easy_face_detail: bool,
    detail_eyes: bool = True,
    detail_face: bool = True,
    detail_body_outline: bool = True,
) -> FigureDetailOptions:
    return FigureDetailOptions(
        enabled=easy_face_detail,
        eyes=detail_eyes,
        face=detail_face,
        body_outline=detail_body_outline,
    )


def resolve_image_style(
    style_preset: Optional[str],
    *,
    easy_painting: Optional[bool] = None,
    easy_simplify: Optional[float] = None,
    easy_face_detail: bool = True,
    detail_eyes: bool = True,
    detail_face: bool = True,
    detail_body_outline: bool = True,
) -> ResolvedImageStyle:
    """Map UI preset (+ optional overrides) to pipeline parameters."""
    preset = normalize_style_preset(style_preset)
    simp = _clip01(easy_simplify if easy_simplify is not None else 0.65)
    fig = _figure_options(easy_face_detail, detail_eyes, detail_face, detail_body_outline)

    if preset == "none":
        use_easy = bool(easy_painting) if easy_painting is not None else False
        if use_easy:
            easy_fig = fig if fig.enabled else _figure_options(True, detail_eyes, detail_face, detail_body_outline)
            return ResolvedImageStyle(
                preset="none",
                background_simplify=simp,
                label_smooth=simp,
                figure_detail=easy_fig,
            )
        return ResolvedImageStyle(
            preset="none",
            background_simplify=0.0,
            label_smooth=0.0,
            figure_detail=FigureDetailOptions(enabled=False),
        )

    if preset == "easy_painting":
        ep_fig = fig if fig.enabled else _figure_options(True, detail_eyes, detail_face, detail_body_outline)
        return ResolvedImageStyle(
            preset=preset,
            background_simplify=simp,
            label_smooth=simp,
            figure_detail=ep_fig,
        )
    if preset == "portrait":
        s = _clip01(simp if easy_simplify is not None else 0.5)
        return ResolvedImageStyle(
            preset=preset,
            background_simplify=s,
            label_smooth=s,
            figure_detail=_figure_options(True, detail_eyes, detail_face, detail_body_outline),
        )
    if preset == "poster":
        s = _clip01(simp if easy_simplify is not None else 0.2)
        return ResolvedImageStyle(
            preset=preset,
            background_simplify=s * 0.35,
            label_smooth=s * 0.4,
            figure_detail=fig,
            posterize_levels=6,
        )
    if preset == "bold":
        s = _clip01(simp if easy_simplify is not None else 0.25)
        return ResolvedImageStyle(
            preset=preset,
            background_simplify=s * 0.4,
            label_smooth=s * 0.45,
            figure_detail=fig,
            extra_saturation=1.28,
        )
    if preset == "expressive":
        return ResolvedImageStyle(
            preset=preset,
            background_simplify=0.0,
            label_smooth=0.0,
            figure_detail=fig,
            hue_shift_deg=18.0,
            extra_saturation=1.18,
        )
    if preset == "graphic":
        s = _clip01(simp if easy_simplify is not None else 0.85)
        return ResolvedImageStyle(
            preset=preset,
            background_simplify=s,
            label_smooth=s,
            figure_detail=fig,
        )
    if preset == "stipple":
        s = _clip01(simp if easy_simplify is not None else 0.15)
        return ResolvedImageStyle(
            preset=preset,
            background_simplify=s * 0.25,
            label_smooth=s * 0.2,
            figure_detail=fig,
            stipple=True,
        )
    if preset == "sketch":
        s = _clip01(simp if easy_simplify is not None else 0.12)
        return ResolvedImageStyle(
            preset=preset,
            background_simplify=0.0,
            label_smooth=s,
            figure_detail=fig,
            sketch=True,
        )
    if preset == "harmony":
        s = _clip01(simp if easy_simplify is not None else 0.22)
        return ResolvedImageStyle(
            preset=preset,
            background_simplify=s * 0.45,
            label_smooth=s * 0.5,
            figure_detail=fig,
            harmony_ab=True,
        )
    return ResolvedImageStyle(
        preset="none",
        background_simplify=0.0,
        label_smooth=0.0,
        figure_detail=FigureDetailOptions(enabled=False),
    )


def apply_posterize_preprocess(image_rgb: np.ndarray, levels: int = 5) -> np.ndarray:
    """Reduce tonal steps in L* for a poster-like look."""
    levels = int(max(3, min(12, levels)))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    step = 255.0 / float(levels)
    Lq = (np.floor(L / step) * step + step * 0.5).astype(np.uint8)
    lab[:, :, 0] = Lq
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_expressive_preprocess(
    image_rgb: np.ndarray,
    hue_shift_deg: float = 14.0,
    saturation_mult: float = 1.15,
) -> np.ndarray:
    """Shift hue and chroma while keeping structure."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    shift = float(hue_shift_deg) / 2.0
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation_mult), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_stipple_preprocess(image_rgb: np.ndarray, cell: int = 7) -> np.ndarray:
    """Halftone-like dots; darker areas get denser pattern (keeps highlights bright)."""
    h, w = image_rgb.shape[:2]
    cell = int(max(4, min(12, cell)))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (xx % cell) / float(cell) - 0.5
    cy = (yy % cell) / float(cell) - 0.5
    dist = np.sqrt(cx * cx + cy * cy)
    radius = np.clip(0.42 * (1.0 - gray), 0.05, 0.4)
    dots = (dist < radius).astype(np.float32)
    base = image_rgb.astype(np.float32)
    paper = np.clip(0.9 + gray * 0.1, 0.0, 1.0)
    ink = np.clip((1.0 - gray) * 0.38, 0.0, 0.38)
    factor = paper[:, :, None] * (1.0 - ink[:, :, None] * (1.0 - dots[:, :, None]))
    out = base * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_sketch_preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Pencil-sketch tone drawing with a hint of original colour."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    inv = 255 - cv2.GaussianBlur(gray, (0, 0), sigmaX=2.5, sigmaY=2.5)
    pencil = cv2.divide(gray, inv, scale=256.0)
    pencil = np.clip(pencil.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
    sketch = cv2.cvtColor(pencil, cv2.COLOR_GRAY2RGB).astype(np.float32)
    tint = cv2.bilateralFilter(image_rgb, 7, 45, 45).astype(np.float32)
    out = sketch * 0.84 + tint * 0.16
    return np.clip(out, 0, 255).astype(np.uint8)


def _restore_figure_detail(
    work_rgb: np.ndarray,
    original_rgb: np.ndarray,
    figure_detail_mask: np.ndarray,
) -> np.ndarray:
    """Blend protected regions from the unprocessed source."""
    weight = _figure_blend_weight(figure_detail_mask)[:, :, None]
    blended = work_rgb.astype(np.float32) * (1.0 - weight) + original_rgb.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_harmony_preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Snap colour into smoother hue families (painterly harmony)."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for ch in (1, 2):
        plane = lab[:, :, ch]
        lab[:, :, ch] = np.round(plane / 28.0) * 28.0
    lab[:, :, 1:3] = cv2.GaussianBlur(lab[:, :, 1:3], (11, 11), 0)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def apply_style_preprocess(
    image_rgb: np.ndarray,
    style: ResolvedImageStyle,
    figure_detail_mask: Optional[np.ndarray] = None,
    original_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run preset-specific preprocessing before quantization."""
    original = original_rgb if original_rgb is not None else image_rgb
    work = image_rgb
    if style.stipple:
        work = apply_stipple_preprocess(work)
    if style.sketch:
        work = apply_sketch_preprocess(work)
    if style.harmony_ab:
        work = apply_harmony_preprocess(work)
    if style.posterize_levels > 0:
        work = apply_posterize_preprocess(work, style.posterize_levels)
    if style.hue_shift_deg != 0.0 or style.extra_saturation != 1.0:
        work = apply_expressive_preprocess(
            work,
            hue_shift_deg=style.hue_shift_deg,
            saturation_mult=style.extra_saturation,
        )
    if style.background_simplify > 1e-6:
        work = apply_easy_painting_preprocess(
            work,
            simplify_strength=style.background_simplify,
            figure_detail_mask=figure_detail_mask,
            detail_source_rgb=original,
        )
    return work


def load_priority_region_mask(
    mask_path: str,
    target_h: int,
    target_w: int,
) -> Optional[np.ndarray]:
    """Load grayscale priority mask (0–1 float), resized to match the processing image."""
    raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    if raw.shape[0] != target_h or raw.shape[1] != target_w:
        raw = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(raw.astype(np.float32) / 255.0, 0.0, 1.0)


def load_priority_region_mask_bytes(
    content: bytes,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    raw = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError("Invalid priority region mask image")
    if raw.shape[0] != target_h or raw.shape[1] != target_w:
        raw = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(raw.astype(np.float32) / 255.0, 0.0, 1.0)


def apply_priority_region_preprocess(
    image_rgb: np.ndarray,
    priority_mask: np.ndarray,
    detail_strength: float = 0.7,
) -> np.ndarray:
    """Keep sharp source colour/detail inside a user-drawn region; simplify only outside."""
    strength = _clip01(detail_strength)
    if strength <= 0.01 or not np.any(priority_mask > 0.05):
        return image_rgb
    sigma = 28.0 + (1.0 - strength) * 22.0
    smooth = cv2.bilateralFilter(image_rgb, 9, sigma, sigma)
    core = _blur_figure_mask(priority_mask)
    weight = np.clip(core * (0.15 + 0.85 * strength), 0.0, 1.0)[:, :, None]
    blended = smooth.astype(np.float32) * (1.0 - weight) + image_rgb.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_easy_painting_preprocess(
    image_rgb: np.ndarray,
    simplify_strength: float = 0.65,
    figure_detail_mask: Optional[np.ndarray] = None,
    detail_source_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Smooth and blend the source before quantization (traditional paint-by-numbers look)."""
    strength = _clip01(simplify_strength)
    if strength <= 1e-6:
        return image_rgb

    detail_source = detail_source_rgb if detail_source_rgb is not None else image_rgb
    d = max(5, int(5 + strength * 10))
    if d % 2 == 0:
        d += 1
    sigma = 25.0 + strength * 45.0
    smooth = cv2.bilateralFilter(image_rgb, d, sigmaColor=sigma, sigmaSpace=sigma)
    k = max(3, int(3 + strength * 8))
    if k % 2 == 0:
        k += 1
    smooth = cv2.GaussianBlur(smooth, (k, k), 0)

    if figure_detail_mask is not None and np.any(figure_detail_mask > 0.05):
        weight = _figure_blend_weight(figure_detail_mask)[:, :, None]
        blended = detail_source.astype(np.float32) * weight + smooth.astype(np.float32) * (1.0 - weight)
        return np.clip(blended, 0, 255).astype(np.uint8)
    return smooth


def smooth_quantized_labels(
    labels: np.ndarray,
    n_colors: int,
    simplify_strength: float = 0.65,
    figure_detail_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Merge speckles and soften region boundaries on the label map."""
    strength = _clip01(simplify_strength)
    if strength <= 1e-6:
        return labels.astype(np.int32)

    labels_u8 = np.clip(labels, 0, max(0, n_colors - 1)).astype(np.uint8)
    strong_k = int(round(3 + strength * 8))
    if strong_k % 2 == 0:
        strong_k += 1

    smoothed_strong = cv2.medianBlur(labels_u8, strong_k)
    out_u8 = smoothed_strong.copy()
    if figure_detail_mask is not None and np.any(figure_detail_mask > 0.05):
        blurred = _blur_figure_mask(figure_detail_mask)
        preserve = blurred > 0.06
        out_u8[preserve] = labels_u8[preserve]

    h, w = out_u8.shape
    min_area = max(8, int(h * w * (0.00002 + strength * 0.00012)))
    out = out_u8.copy()
    figure_protect = None
    if figure_detail_mask is not None:
        figure_protect = (_blur_figure_mask(figure_detail_mask) > 0.06).astype(np.uint8)

    for idx in range(n_colors):
        mask = (out == idx).astype(np.uint8)
        if np.sum(mask) == 0:
            continue
        num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp_id in range(1, num_labels):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area >= min_area:
                continue
            if figure_protect is not None:
                comp_pixels = cc == comp_id
                if np.any(figure_protect[comp_pixels] > 0):
                    continue
            comp_mask = (cc == comp_id).astype(np.uint8) * 255
            dilated = cv2.dilate(comp_mask, np.ones((5, 5), np.uint8), iterations=2)
            border = cv2.bitwise_and(dilated, cv2.bitwise_not(comp_mask))
            neighbor_vals = out[border > 0]
            neighbor_vals = neighbor_vals[neighbor_vals != idx]
            if neighbor_vals.size == 0:
                continue
            replace_idx = int(np.bincount(neighbor_vals.astype(np.int32)).argmax())
            out[cc == comp_id] = replace_idx
    return out.astype(np.int32)


def _palette_lab_centers(palette: List[Dict]) -> np.ndarray:
    """Nx3 float Lab centers for palette entries."""
    centers = []
    for entry in palette:
        rgb = np.array(entry["rgb"], dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        centers.append(lab[0, 0])
    return np.stack(centers, axis=0)


def detect_skin_tone_mask(
    image_rgb: np.ndarray,
    figure_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Float mask for likely skin pixels (YCrCb rules, boosted inside face region)."""
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.int16)
    cr = ycrcb[:, :, 1].astype(np.int16)
    cb = ycrcb[:, :, 2].astype(np.int16)
    rule_a = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y >= 50)
    rule_b = (cr >= 120) & (cr <= 185) & (cb >= 68) & (cb <= 140) & (y >= 35) & (y <= 245)
    skin = np.logical_or(rule_a, rule_b).astype(np.float32)
    if figure_mask is not None:
        fm = _blur_figure_mask(figure_mask)
        skin = skin * np.clip(0.35 + fm * 0.65, 0.0, 1.0)
    h, w = skin.shape
    k = max(5, int(min(h, w) * 0.018)) | 1
    skin = cv2.GaussianBlur(skin, (k, k), 0)
    return np.clip(skin, 0.0, 1.0)


def _skin_palette_slot_count(n_colors: int, strength: float) -> int:
    """How many palette entries to reserve for skin (2 at low strength, up to ~⅓ of palette at max)."""
    strength = _clip01(strength)
    if strength <= 0.01 or n_colors < 4:
        return 0
    lo = 2
    hi = min(6, max(4, n_colors // 3))
    slots = int(round(lo + strength * (hi - lo)))
    return int(max(2, min(slots, n_colors - 2)))


def _priority_palette_slot_count(n_colors: int, strength: float) -> int:
    """Palette colours fitted only inside the user priority region (more = finer face/region)."""
    strength = _clip01(strength)
    if strength <= 0.01 or n_colors < 6:
        return 0
    lo = 4
    hi = min(10, max(6, n_colors // 2))
    slots = int(round(lo + strength * (hi - lo)))
    return int(max(3, min(slots, n_colors - 2)))


def _gather_priority_sample_pixels(
    original_lab_pixels: np.ndarray,
    priority_mask: np.ndarray,
) -> np.ndarray:
    """All Lab samples inside the drawn region (keeps shadows, lips, eyes — not just 'warm skin')."""
    sel = priority_mask.ravel() > 0.2
    pts = original_lab_pixels[sel]
    if pts.shape[0] >= 12:
        return pts
    sel = priority_mask.ravel() > 0.08
    return original_lab_pixels[sel]


def _fit_region_lab_centers(region_pixels: np.ndarray, n_slots: int, seed: int) -> np.ndarray:
    """Cluster centres for a priority region (full colour range, sorted light → dark)."""
    if n_slots <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if region_pixels.shape[0] == 0:
        defaults = np.array(
            [
                [205, 178, 182],
                [185, 168, 172],
                [165, 158, 162],
                [145, 148, 152],
                [125, 142, 148],
                [105, 138, 145],
            ],
            dtype=np.float32,
        )
        return defaults[:n_slots]
    if region_pixels.shape[0] < n_slots * 3:
        mean = np.mean(region_pixels, axis=0, keepdims=True)
        return np.repeat(mean, n_slots, axis=0)
    if n_slots == 1:
        return np.mean(region_pixels, axis=0, keepdims=True)
    kmeans = KMeans(n_clusters=n_slots, random_state=seed, n_init=12, init="random")
    kmeans.fit(region_pixels)
    centers = kmeans.cluster_centers_
    order = np.argsort(-centers[:, 0])
    return centers[order]


def _gather_skin_sample_pixels(
    image_rgb: np.ndarray,
    original_lab_pixels: np.ndarray,
    figure_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Lab pixels sampled from detected skin + face for palette fitting."""
    skin_fit = detect_skin_tone_mask(image_rgb, None).ravel() > 0.18
    sel = skin_fit.copy()
    if figure_mask is not None:
        fm = _blur_figure_mask(figure_mask).ravel()
        sel |= fm > 0.22
    face = detect_face_detail_mask(image_rgb).ravel()
    sel |= face > 0.28
    pts = original_lab_pixels[sel]
    if pts.shape[0] < 40:
        face_pts = original_lab_pixels[face > 0.2]
        if face_pts.shape[0] >= 12:
            pts = face_pts
    return pts


def _filter_warm_skin_lab_pixels(skin_pixels: np.ndarray) -> np.ndarray:
    """Drop grass/sky outliers so skin palette fitting stays on face tones."""
    if skin_pixels.shape[0] == 0:
        return skin_pixels
    a = skin_pixels[:, 1]
    b = skin_pixels[:, 2]
    l = skin_pixels[:, 0]
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
    warm = (l >= 95) & (l <= 245) & (a >= 132) & (b >= 132) & (chroma >= 6.0) & (b >= a - 6)
    filtered = skin_pixels[warm]
    if filtered.shape[0] >= 20:
        return filtered
    return skin_pixels


def _fit_skin_lab_centers(skin_pixels: np.ndarray, n_slots: int, seed: int) -> np.ndarray:
    """Warm skin cluster centres (light → dark), from photo samples or safe defaults."""
    if n_slots <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    skin_pixels = _filter_warm_skin_lab_pixels(skin_pixels)
    if skin_pixels.shape[0] == 0:
        defaults = np.array(
            [
                [205, 178, 182],
                [178, 170, 175],
                [148, 162, 170],
                [125, 155, 165],
                [100, 148, 158],
                [82, 142, 152],
            ],
            dtype=np.float32,
        )
        return defaults[:n_slots]
    if skin_pixels.shape[0] < n_slots * 3:
        mean = np.mean(skin_pixels, axis=0, keepdims=True)
        return np.repeat(mean, n_slots, axis=0)
    if n_slots == 1:
        return np.mean(skin_pixels, axis=0, keepdims=True)
    kmeans_skin = KMeans(n_clusters=n_slots, random_state=seed, n_init=10, init="random")
    kmeans_skin.fit(skin_pixels)
    centers = _refine_skin_lab_centers(kmeans_skin.cluster_centers_, skin_pixels)
    order = np.argsort(-centers[:, 0])
    return centers[order]


def _normalize_must_include_hex_list(hex_list: Optional[List[str]]) -> List[str]:
    """Deduplicated #RRGGBB colours from user picks."""
    if not hex_list:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for raw in hex_list:
        s = str(raw).strip().upper()
        if not s.startswith("#"):
            s = f"#{s}"
        if len(s) != 7:
            continue
        try:
            int(s[1:], 16)
        except ValueError:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _hex_to_rgb_uint8(hex_color: str) -> np.ndarray:
    h = hex_color.strip().lstrip("#").upper()
    return np.array(
        [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
        dtype=np.uint8,
    )


def _rgb_uint8_to_lab_float(rgb: np.ndarray, saturation_boost: float = 1.0) -> np.ndarray:
    arr = rgb.reshape(1, 1, 3)
    boosted = _apply_saturation_boost_rgb(arr, saturation_boost)
    lab = cv2.cvtColor(boosted, cv2.COLOR_RGB2LAB)
    return lab[0, 0].astype(np.float32)


def _must_include_lab_centers(hex_list: List[str], saturation_boost: float) -> np.ndarray:
    return np.vstack([_rgb_uint8_to_lab_float(_hex_to_rgb_uint8(h), saturation_boost) for h in hex_list])


def _sync_palette_must_include_swatches(
    labels: np.ndarray,
    palette: List[Dict],
    must_count: int,
    hex_list: List[str],
) -> List[Dict]:
    total = labels.size
    for idx in range(min(must_count, len(palette), len(hex_list))):
        hex_norm = hex_list[idx].strip().upper()
        if not hex_norm.startswith("#"):
            hex_norm = f"#{hex_norm}"
        rgb = _hex_to_rgb_uint8(hex_norm)
        mask = labels == idx
        palette[idx] = {
            "index": idx,
            "rgb": rgb.tolist(),
            "hex": hex_norm,
            "must_include": True,
            "coverage": round(float(np.sum(mask)) / total * 100, 2),
        }
    return palette


def _palette_has_must_include_hex(palette: List[Dict], hex_norm: str) -> bool:
    target = hex_norm.strip().upper()
    if not target.startswith("#"):
        target = f"#{target}"
    for entry in palette:
        if str(entry.get("hex", "")).strip().upper() == target:
            return True
    return False


def _inject_missing_must_include_swatches(
    palette: List[Dict],
    must_hex_list: List[str],
) -> List[Dict]:
    """Ensure every picked colour appears as an exact swatch (reserved prefix slots)."""
    for mi, raw_hex in enumerate(must_hex_list):
        hex_norm = raw_hex.strip().upper()
        if not hex_norm.startswith("#"):
            hex_norm = f"#{hex_norm}"
        if _palette_has_must_include_hex(palette, hex_norm):
            for entry in palette:
                if str(entry.get("hex", "")).strip().upper() == hex_norm:
                    entry["must_include"] = True
                    entry["hex"] = hex_norm
                    entry["rgb"] = _hex_to_rgb_uint8(hex_norm).tolist()
                    if "skin" in entry:
                        del entry["skin"]
            continue
        if mi >= len(palette):
            continue
        rgb = _hex_to_rgb_uint8(hex_norm)
        palette[mi] = {
            "index": mi,
            "rgb": rgb.tolist(),
            "hex": hex_norm,
            "must_include": True,
            "coverage": float(palette[mi].get("coverage", 0)),
        }
    return palette


def _nudge_must_include_labels(
    labels: np.ndarray,
    original_lab_pixels: np.ndarray,
    must_lab_centers: np.ndarray,
    must_count: int,
    max_dist_sq: float = 480.0,
) -> np.ndarray:
    """Pull pixels near a picked colour onto its reserved palette slot."""
    if must_count <= 0:
        return labels
    flat = labels.ravel().copy()
    for slot in range(must_count):
        center = must_lab_centers[slot]
        dists = np.sum((original_lab_pixels - center) ** 2, axis=1)
        flat[dists < max_dist_sq] = slot
    return flat.reshape(labels.shape)


def _finalize_must_include_in_pipeline(
    labels: np.ndarray,
    palette: List[Dict],
    image_rgb: np.ndarray,
    must_hex_list: List[str],
    saturation_boost: float,
) -> Tuple[np.ndarray, List[Dict]]:
    """Re-apply must-include slots after snap/smooth (those steps can steal label indices)."""
    must_count = len(must_hex_list)
    if must_count <= 0:
        return labels, palette
    boosted = _apply_saturation_boost_rgb(image_rgb, saturation_boost)
    orig_lab = _rgb_to_lab_pixels(boosted)
    must_lab = _must_include_lab_centers(must_hex_list, saturation_boost)
    labels = _nudge_must_include_labels(labels, orig_lab, must_lab, must_count)
    palette = _sync_palette_must_include_swatches(labels, palette, must_count, must_hex_list)
    palette = _inject_missing_must_include_swatches(palette, must_hex_list)
    return labels, palette


def _inject_skin_lab_centers(lab_centers: np.ndarray, skin_centers: np.ndarray) -> np.ndarray:
    out = lab_centers.copy()
    n = min(skin_centers.shape[0], out.shape[0])
    out[:n] = skin_centers[:n]
    return out


def _nudge_skin_region_to_skin_slots(
    labels: np.ndarray,
    skin_slots: int,
    skin_bias_mask: np.ndarray,
    lab_centers: np.ndarray,
    original_lab_pixels: np.ndarray,
) -> np.ndarray:
    """Assign strong skin-mask pixels to reserved skin palette indices (warm colours)."""
    if skin_slots <= 0:
        return labels
    h, w = labels.shape
    strong = skin_bias_mask.ravel() > 0.42
    wrong = strong & (labels.ravel() >= skin_slots)
    if not np.any(wrong):
        return labels
    skin_centers = lab_centers[:skin_slots]
    px = original_lab_pixels[wrong]
    dists = np.sum((px[:, None, :] - skin_centers[None, :, :]) ** 2, axis=2)
    labels_flat = labels.ravel()
    labels_flat[wrong] = dists.argmin(axis=1).astype(np.int32)
    return labels_flat.reshape(h, w)


def _lab_float_to_rgb_uint8(lab_center: np.ndarray) -> np.ndarray:
    lab_u8 = np.clip(lab_center, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    return np.clip(cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)[0, 0], 0, 255).astype(int)


def _sync_palette_skin_swatches(
    labels: np.ndarray,
    palette: List[Dict],
    skin_slots: int,
    source_rgb: np.ndarray,
    skin_lab_centers: np.ndarray,
) -> List[Dict]:
    """Reserved skin swatches come from photo-fitted Lab centres; blend medians when labels match."""
    if skin_slots <= 0:
        return palette
    total = labels.size
    for idx in range(min(skin_slots, len(palette))):
        mask = labels == idx
        base = _lab_float_to_rgb_uint8(skin_lab_centers[idx])
        if np.sum(mask) >= 25:
            med = np.median(source_rgb[mask], axis=0).astype(np.float32)
            rgb = np.clip(0.35 * base + 0.65 * med, 0, 255).astype(int)
        else:
            rgb = base
        palette[idx] = {
            **palette[idx],
            "index": idx,
            "rgb": rgb.tolist(),
            "hex": "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2])),
            "skin": True,
            "coverage": round(float(np.sum(mask)) / total * 100, 2),
        }
    return palette


def _apply_saturation_boost_rgb(image_rgb: np.ndarray, saturation_boost: float) -> np.ndarray:
    if abs(float(saturation_boost) - 1.0) < 1e-6:
        return image_rgb
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation_boost), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _rgb_to_lab_pixels(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    return lab.reshape(-1, 3).astype(np.float32)


def _refine_skin_lab_centers(centers: np.ndarray, skin_pixels: np.ndarray) -> np.ndarray:
    """Pull greyish skin cluster centers toward the warm mean of detected skin pixels."""
    if skin_pixels.shape[0] == 0:
        return centers
    out = centers.copy()
    chroma = np.sqrt((skin_pixels[:, 1] - 128.0) ** 2 + (skin_pixels[:, 2] - 128.0) ** 2)
    warm = skin_pixels[chroma >= 10.0]
    if warm.shape[0] == 0:
        warm = skin_pixels
    warm_mean = np.mean(warm, axis=0)
    for i, center in enumerate(out):
        c_chroma = float(np.sqrt((center[1] - 128.0) ** 2 + (center[2] - 128.0) ** 2))
        if c_chroma < 16.0:
            out[i] = 0.45 * center + 0.55 * warm_mean
    return out


def _labels_from_lab_centers(
    pixels: np.ndarray,
    centers: np.ndarray,
    h: int,
    w: int,
    *,
    skin_bias_mask: Optional[np.ndarray] = None,
    skin_slots: int = 0,
    original_lab_pixels: Optional[np.ndarray] = None,
    skin_bias_strength: float = 0.65,
    fixed_slot_offset: int = 0,
) -> np.ndarray:
    """Nearest palette entry; skin regions prefer warm slots via soft distance bias."""
    px = pixels
    if original_lab_pixels is not None and skin_bias_mask is not None:
        t = np.clip(skin_bias_mask.ravel(), 0.0, 1.0)[:, None]
        px = pixels * (1.0 - t) + original_lab_pixels * t
    dists = np.sum((px[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    if skin_slots > 0 and skin_bias_mask is not None:
        strength = _clip01(skin_bias_strength)
        skin_w = np.clip(skin_bias_mask.ravel(), 0.0, 1.0)
        penalty = 1.0 + 1.35 * strength
        free_from = fixed_slot_offset + skin_slots
        dists[:, free_from:] *= 1.0 + (penalty - 1.0) * skin_w[:, None]
    return dists.argmin(axis=1).reshape(h, w).astype(np.int32)


def _build_palette_and_quantized(
    image: np.ndarray,
    labels: np.ndarray,
    lab_centers_float: np.ndarray,
) -> Tuple[np.ndarray, List[Dict]]:
    h, w = labels.shape
    n_colors = lab_centers_float.shape[0]
    rgb_centers = []
    for lab_center in lab_centers_float:
        lab_center_uint8 = np.clip(lab_center, 0, 255).astype(np.uint8)
        lab_3d = lab_center_uint8.reshape(1, 1, 3)
        rgb_3d = cv2.cvtColor(lab_3d, cv2.COLOR_LAB2RGB)
        rgb_val = np.clip(rgb_3d[0, 0], 0, 255).astype(int)
        rgb_centers.append(rgb_val.tolist())

    total_pixels = h * w
    palette = []
    for idx in range(n_colors):
        coverage = np.sum(labels == idx) / total_pixels * 100
        palette.append({
            "index": idx,
            "rgb": rgb_centers[idx],
            "hex": "#{:02x}{:02x}{:02x}".format(
                int(rgb_centers[idx][0]),
                int(rgb_centers[idx][1]),
                int(rgb_centers[idx][2]),
            ),
            "coverage": round(coverage, 2),
        })

    quantized = np.zeros_like(image)
    for idx in range(n_colors):
        mask = labels == idx
        quantized[mask] = np.array(rgb_centers[idx], dtype=np.uint8)
    return quantized, palette


def defuse_figure_label_halo(
    labels: np.ndarray,
    figure_detail_mask: np.ndarray,
    n_colors: int,
) -> np.ndarray:
    """Median-filter label speckles in the soft transition band around the figure."""
    m = _blur_figure_mask(figure_detail_mask)
    ring = (m >= 0.05) & (m < 0.46)
    if not np.any(ring):
        return labels.astype(np.int32)

    u8 = np.clip(labels, 0, max(0, n_colors - 1)).astype(np.uint8)
    k = 5
    cleaned = cv2.medianBlur(u8, k)
    out = labels.copy()
    out[ring] = cleaned[ring]
    return out.astype(np.int32)


def snap_figure_labels_to_original(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    palette: List[Dict],
    figure_detail_mask: np.ndarray,
    min_weight: float = 0.72,
    skin_slot_count: int = 0,
) -> np.ndarray:
    """Re-map face-core pixels to nearest palette colour from the original photo."""
    weight = _figure_blend_weight(figure_detail_mask)
    keep = _figure_label_core_mask(figure_detail_mask) & (weight >= min_weight)
    if not np.any(keep):
        return labels

    lab_centers = _palette_lab_centers(palette)
    orig_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ys, xs = np.where(keep)
    pixels = orig_lab[ys, xs]
    dists = np.sum((pixels[:, None, :] - lab_centers[None, :, :]) ** 2, axis=2)
    if skin_slot_count > 0:
        skin_m = detect_skin_tone_mask(image_rgb, figure_detail_mask)
        skin_w = np.clip(skin_m[keep], 0.0, 1.0)
        dists[:, skin_slot_count:] *= 1.0 + 1.2 * skin_w[:, None]
    nearest = dists.argmin(axis=1).astype(np.int32)

    out = labels.copy()
    out[ys, xs] = nearest
    return out


def render_quantized_from_labels(labels: np.ndarray, palette: List[Dict]) -> np.ndarray:
    """Build RGB preview image from label map and palette entries."""
    h, w = labels.shape
    quantized = np.zeros((h, w, 3), dtype=np.uint8)
    for entry in palette:
        idx = int(entry["index"])
        rgb = entry["rgb"]
        quantized[labels == idx] = np.array(rgb, dtype=np.uint8)
    return quantized


def quantize_for_pipeline(
    image: np.ndarray,
    n_colors: int,
    *,
    saturation_boost: float = 1.0,
    easy_painting: bool = False,
    easy_simplify: float = 0.65,
    easy_face_detail: bool = False,
    detail_eyes: bool = True,
    detail_face: bool = True,
    detail_body_outline: bool = True,
    style_preset: Optional[str] = None,
    favor_skin_tones: bool = False,
    skin_tone_strength: float = 0.65,
    priority_region_mask: Optional[np.ndarray] = None,
    priority_region_strength: float = 0.7,
    mask_detail_level: float = 0.5,
    must_include_hex: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Quantize with optional style preset / figure-detail preservation."""
    must_hex_list = _normalize_must_include_hex_list(must_include_hex)
    preset = normalize_style_preset(style_preset)
    legacy_easy = easy_painting if preset == "none" else None
    style = resolve_image_style(
        style_preset,
        easy_painting=legacy_easy,
        easy_simplify=easy_simplify,
        easy_face_detail=easy_face_detail,
        detail_eyes=detail_eyes,
        detail_face=detail_face,
        detail_body_outline=detail_body_outline,
    )

    has_priority = (
        priority_region_mask is not None and np.any(priority_region_mask > 0.05)
    )
    figure_mask: Optional[np.ndarray] = None
    figure_mask_labels: Optional[np.ndarray] = None
    if has_priority:
        figure_mask = np.clip(priority_region_mask.astype(np.float32), 0.0, 1.0)
        figure_mask_labels = figure_mask
    elif style.figure_detail.enabled:
        figure_mask = build_figure_detail_mask_from_options(
            image, style.figure_detail, include_body_outline=True
        )
        figure_mask_labels = build_figure_detail_mask_from_options(
            image, style.figure_detail, include_body_outline=False
        )

    if style.preset == "none" and style.background_simplify <= 1e-6:
        work = image
        if has_priority:
            work = apply_priority_region_preprocess(
                image, figure_mask, detail_strength=priority_region_strength
            )
    else:
        work = apply_style_preprocess(
            image, style, figure_detail_mask=figure_mask, original_rgb=image
        )
        if has_priority and style.background_simplify <= 1e-6:
            work = apply_priority_region_preprocess(
                work, figure_mask, detail_strength=priority_region_strength
            )

    sample_w = None
    if figure_mask is not None and (has_priority or style.figure_detail.enabled):
        fw = _figure_blend_weight(figure_mask)
        sample_w = 0.35 + 2.5 * fw

    skin_ref = figure_mask_labels if figure_mask_labels is not None else figure_mask
    if favor_skin_tones and skin_tone_strength > 0.01:
        skin_m = detect_skin_tone_mask(image, skin_ref)
        if sample_w is None:
            sample_w = np.full(image.shape[:2], 0.4, dtype=np.float32)
        sample_w = sample_w + skin_tone_strength * 3.0 * skin_m

    skin_slots = (
        _skin_palette_slot_count(n_colors, skin_tone_strength)
        if favor_skin_tones and skin_tone_strength > 0.01
        else 0
    )
    priority_slots = (
        _priority_palette_slot_count(n_colors, priority_region_strength)
        if has_priority
        else 0
    )
    reserved_slots = max(skin_slots, priority_slots)

    labels, quantized, palette = quantize_lab(
        work,
        n_colors,
        seed=seed,
        saturation_boost=saturation_boost,
        sample_weight_2d=sample_w,
        favor_skin_tones=favor_skin_tones,
        skin_tone_strength=skin_tone_strength,
        figure_mask_for_skin=skin_ref,
        skin_source_rgb=image,
        reserved_palette_slots=0 if must_hex_list else reserved_slots,
        reserved_fit_mask=skin_ref if reserved_slots > 0 and not must_hex_list else None,
        reserved_from_priority=has_priority and priority_slots >= skin_slots and not must_hex_list,
        must_include_hex=must_hex_list or None,
    )
    label_mask = figure_mask_labels if figure_mask_labels is not None else figure_mask
    snap_skin_slots = 0 if must_hex_list else reserved_slots
    if label_mask is not None and (has_priority or style.figure_detail.enabled):
        labels = snap_figure_labels_to_original(
            image,
            labels,
            palette,
            label_mask,
            min_weight=0.42 if has_priority else 0.72,
            skin_slot_count=snap_skin_slots,
        )

    smooth_strength = style.label_smooth
    if smooth_strength > 1e-6:
        labels = smooth_quantized_labels(
            labels,
            n_colors,
            simplify_strength=smooth_strength,
            figure_detail_mask=label_mask,
        )
    elif has_priority and mask_detail_level < 0.92:
        labels = smooth_quantized_labels(
            labels,
            n_colors,
            simplify_strength=(1.0 - mask_detail_level) * 0.35,
            figure_detail_mask=label_mask,
        )

    if reserved_slots > 0 and not must_hex_list:
        if has_priority and priority_slots > 0:
            region_px = _gather_priority_sample_pixels(_rgb_to_lab_pixels(image), label_mask)
            region_lab = _fit_region_lab_centers(region_px, reserved_slots, seed)
        else:
            region_px = _gather_skin_sample_pixels(image, _rgb_to_lab_pixels(image), skin_ref)
            region_lab = _fit_skin_lab_centers(region_px, reserved_slots, seed)
        palette = _sync_palette_skin_swatches(labels, palette, reserved_slots, image, region_lab)
    if must_hex_list:
        labels, palette = _finalize_must_include_in_pipeline(
            labels, palette, image, must_hex_list, saturation_boost
        )
    quantized = render_quantized_from_labels(labels, palette)
    return labels, quantized, palette


def make_quantization_preview(
    image_path: str,
    n_colors: int,
    max_side: int,
    saturation_boost: float,
    easy_painting: bool = False,
    easy_simplify: float = 0.65,
    easy_face_detail: bool = False,
    detail_eyes: bool = True,
    detail_face: bool = True,
    detail_body_outline: bool = True,
    style_preset: Optional[str] = None,
    favor_skin_tones: bool = False,
    skin_tone_strength: float = 0.65,
    priority_region_mask: Optional[np.ndarray] = None,
    priority_region_strength: float = 0.7,
    mask_detail_level: float = 0.5,
    must_include_hex: Optional[List[str]] = None,
) -> Tuple[bytes, List[Dict]]:
    """Fast JPEG preview and palette swatches (no layer masks).

    Uses the same max_side as layer generation so the Image tab palette matches Layers.
    """
    normalized = load_rgb_image_normalized(image_path, int(max_side))
    _labels, quantized, palette = quantize_for_pipeline(
        normalized,
        n_colors,
        saturation_boost=saturation_boost,
        easy_painting=easy_painting,
        easy_simplify=easy_simplify,
        easy_face_detail=easy_face_detail,
        detail_eyes=detail_eyes,
        detail_face=detail_face,
        detail_body_outline=detail_body_outline,
        style_preset=style_preset,
        favor_skin_tones=favor_skin_tones,
        skin_tone_strength=skin_tone_strength,
        priority_region_mask=priority_region_mask,
        priority_region_strength=priority_region_strength,
        mask_detail_level=mask_detail_level,
        must_include_hex=must_include_hex,
    )
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88]
    )
    if not ok:
        raise ValueError("Failed to encode preview JPEG")
    return buf.tobytes(), palette


def make_quantization_preview_jpeg(
    image_path: str,
    n_colors: int,
    max_side: int,
    saturation_boost: float,
    easy_painting: bool = False,
    easy_simplify: float = 0.65,
    easy_face_detail: bool = False,
    detail_eyes: bool = True,
    detail_face: bool = True,
    detail_body_outline: bool = True,
    style_preset: Optional[str] = None,
    favor_skin_tones: bool = False,
    skin_tone_strength: float = 0.65,
    priority_region_mask: Optional[np.ndarray] = None,
    priority_region_strength: float = 0.7,
) -> bytes:
    """Fast JPEG preview of quantized colours (no layer masks)."""
    jpeg, _palette = make_quantization_preview(
        image_path,
        n_colors,
        max_side,
        saturation_boost,
        easy_painting=easy_painting,
        easy_simplify=easy_simplify,
        easy_face_detail=easy_face_detail,
        detail_eyes=detail_eyes,
        detail_face=detail_face,
        detail_body_outline=detail_body_outline,
        style_preset=style_preset,
        favor_skin_tones=favor_skin_tones,
        skin_tone_strength=skin_tone_strength,
        priority_region_mask=priority_region_mask,
        priority_region_strength=priority_region_strength,
    )
    return jpeg


def quantize_lab(
    image: np.ndarray,
    n_colors: int,
    seed: int = 42,
    saturation_boost: float = 1.0,
    sample_weight_2d: Optional[np.ndarray] = None,
    favor_skin_tones: bool = False,
    skin_tone_strength: float = 0.65,
    figure_mask_for_skin: Optional[np.ndarray] = None,
    skin_source_rgb: Optional[np.ndarray] = None,
    reserved_palette_slots: int = 0,
    reserved_fit_mask: Optional[np.ndarray] = None,
    reserved_from_priority: bool = False,
    must_include_hex: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Quantize image using Lab color space and k-means.
    
    Args:
        image: Input RGB image
        n_colors: Number of colors in palette
        seed: Random seed for reproducibility
        saturation_boost: Multiplier for saturation (1.0 = no change, >1.0 = more vibrant, <1.0 = less vibrant)
    """
    must_hex_list = _normalize_must_include_hex_list(must_include_hex)
    must_count = len(must_hex_list)
    if must_count >= n_colors:
        raise ValueError(
            f"Too many must-include colours ({must_count}); "
            f"max is {n_colors - 1} for palette size {n_colors}"
        )
    must_prefix_lab = (
        _must_include_lab_centers(must_hex_list, saturation_boost) if must_count > 0 else None
    )
    inner_colors = n_colors - must_count
    fixed_slot_offset = must_count

    # Must-include colours own the first N palette slots — do not reserve skin slots on top.
    if must_count > 0:
        reserved_palette_slots = 0
        reserved_fit_mask = None

    # Ensure image is valid (no NaN or inf values)
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        raise ValueError("Image contains invalid (NaN or Inf) values")
    
    image = _apply_saturation_boost_rgb(image, saturation_boost)
    skin_src = skin_source_rgb if skin_source_rgb is not None else image
    skin_src = _apply_saturation_boost_rgb(skin_src, saturation_boost)

    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    h, w = lab_image.shape[:2]
    pixels = lab_image.reshape(-1, 3).astype(np.float32)
    original_lab_pixels = _rgb_to_lab_pixels(skin_src)
    
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
    
    sample_weight = None
    if sample_weight_2d is not None:
        sample_weight = np.clip(sample_weight_2d.reshape(-1).astype(np.float64), 0.05, None)

    skin_slots = (
        0
        if must_count > 0
        else (_skin_palette_slot_count(inner_colors, skin_tone_strength) if favor_skin_tones else 0)
    )
    reserved_slots = 0 if must_count > 0 else max(skin_slots, reserved_palette_slots)
    lab_centers_float: Optional[np.ndarray] = None
    skin_bias_mask = None
    skin_centers_reserved: Optional[np.ndarray] = None

    if reserved_slots > 0 and reserved_fit_mask is not None:
        skin_bias_mask = detect_skin_tone_mask(skin_src, reserved_fit_mask)
        if reserved_from_priority:
            skin_pixels = _gather_priority_sample_pixels(original_lab_pixels, reserved_fit_mask)
            skin_centers_reserved = _fit_region_lab_centers(skin_pixels, reserved_slots, seed)
        else:
            skin_pixels = _gather_skin_sample_pixels(
                skin_src, original_lab_pixels, reserved_fit_mask
            )
            skin_centers_reserved = _fit_skin_lab_centers(skin_pixels, reserved_slots, seed)
        skin_sel = skin_bias_mask.ravel() > 0.22 if skin_bias_mask is not None else np.zeros(0, dtype=bool)
        if skin_sel.size == 0:
            skin_sel = reserved_fit_mask.ravel() > 0.2
        min_skin_px = max(reserved_slots * 2, 12) if reserved_from_priority else max(reserved_slots * 4, 24)
        if skin_pixels.shape[0] >= min_skin_px:
            bg_slots = inner_colors - reserved_slots
            bg_weight = (
                sample_weight.copy()
                if sample_weight is not None
                else np.ones(pixels.shape[0], dtype=np.float64)
            )
            bg_weight[skin_sel] *= 0.06
            kmeans_bg = KMeans(
                n_clusters=bg_slots, random_state=seed + 1, n_init=10, init="random"
            )
            kmeans_bg.fit(pixels, sample_weight=bg_weight)
            lab_centers_float = np.vstack([skin_centers_reserved, kmeans_bg.cluster_centers_])

    if lab_centers_float is None:
        kmeans = KMeans(n_clusters=inner_colors, random_state=seed, n_init=10, init="random")
        if sample_weight is not None:
            kmeans.fit(pixels, sample_weight=sample_weight)
            labels_flat = kmeans.predict(pixels)
        else:
            labels_flat = kmeans.fit_predict(pixels)
        lab_centers_float = kmeans.cluster_centers_
        if skin_centers_reserved is not None and reserved_slots > 0:
            lab_centers_float = _inject_skin_lab_centers(lab_centers_float, skin_centers_reserved)
        bias_strength = max(skin_tone_strength, 0.55 if reserved_from_priority else 0.0)
        if skin_bias_mask is not None and reserved_slots > 0:
            labels = _labels_from_lab_centers(
                pixels,
                lab_centers_float,
                h,
                w,
                skin_bias_mask=skin_bias_mask,
                skin_slots=reserved_slots,
                original_lab_pixels=original_lab_pixels,
                skin_bias_strength=bias_strength,
                fixed_slot_offset=fixed_slot_offset,
            )
            labels_flat = labels.ravel()
    else:
        bias_strength = max(skin_tone_strength, 0.55 if reserved_from_priority else 0.0)
        labels = _labels_from_lab_centers(
            pixels,
            lab_centers_float,
            h,
            w,
            skin_bias_mask=skin_bias_mask,
            skin_slots=reserved_slots,
            original_lab_pixels=original_lab_pixels,
            skin_bias_strength=bias_strength,
            fixed_slot_offset=fixed_slot_offset,
        )
        labels_flat = labels.ravel()

    labels = labels_flat.reshape(h, w)
    if reserved_slots > 0 and skin_bias_mask is not None:
        labels = _nudge_skin_region_to_skin_slots(
            labels,
            reserved_slots,
            skin_bias_mask,
            lab_centers_float,
            original_lab_pixels,
        )
    if must_count > 0 and must_prefix_lab is not None:
        full_centers = np.vstack([must_prefix_lab, lab_centers_float])
        bias_strength = max(skin_tone_strength, 0.55 if reserved_from_priority else 0.0)
        labels = _labels_from_lab_centers(
            pixels,
            full_centers,
            h,
            w,
            skin_bias_mask=skin_bias_mask,
            skin_slots=reserved_slots,
            original_lab_pixels=original_lab_pixels,
            skin_bias_strength=bias_strength,
            fixed_slot_offset=fixed_slot_offset,
        )
        labels = _nudge_must_include_labels(
            labels, original_lab_pixels, must_prefix_lab, must_count
        )
        lab_centers_float = full_centers
    quantized, palette = _build_palette_and_quantized(image, labels, lab_centers_float)
    if reserved_slots > 0 and skin_centers_reserved is not None:
        palette = _sync_palette_skin_swatches(
            labels, palette, reserved_slots, skin_src, skin_centers_reserved
        )
        quantized = render_quantized_from_labels(labels, palette)
    if must_count > 0:
        palette = _sync_palette_must_include_swatches(labels, palette, must_count, must_hex_list)
        palette = _inject_missing_must_include_swatches(palette, must_hex_list)
        quantized = render_quantized_from_labels(labels, palette)
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
    """Calculate Lab lightness L* normalized to 0..1."""
    rgb_arr = np.array([[rgb]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)[0, 0]
    # OpenCV uint8 Lab encoding: L in 0..255
    return float(lab[0]) / 255.0


def calculate_base_color_bonus(rgb: List[int]) -> float:
    """Estimate base-color bonus from chroma (saturated colors get a small boost)."""
    rgb_arr = np.array([[rgb]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    # Convert OpenCV Lab a,b from 0..255 to signed-ish range around 0.
    a = float(lab[1] - 128.0)
    b = float(lab[2] - 128.0)
    chroma = float(np.sqrt((a * a) + (b * b)))
    return max(0.0, min(1.0, chroma / 100.0))


def order_layers(palette: List[Dict], order_mode: str) -> List[int]:
    """Order layers by coverage, lightness, or return manual order."""
    if order_mode in ('auto', 'largest'):
        # Automatic priority: large/light/base-color layers first, dark detail later.
        # priority = 0.6 * area_score + 0.3 * lightness_score + 0.1 * base_color_bonus
        def layer_priority(p: Dict) -> float:
            area_score = max(0.0, min(1.0, float(p.get('coverage', 0.0)) / 100.0))
            lightness_score = calculate_lightness(p['rgb'])
            base_color_bonus = calculate_base_color_bonus(p['rgb'])
            return (0.6 * area_score) + (0.3 * lightness_score) + (0.1 * base_color_bonus)

        sorted_palette = sorted(palette, key=layer_priority, reverse=True)
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


def _build_later_label_unions(labels: np.ndarray, order: List[int]) -> List[np.ndarray]:
    """Per layer index: union of final label regions for colors painted later in order."""
    shape = labels.shape
    n = len(order)
    unions: List[np.ndarray] = [np.zeros(shape, dtype=np.uint8) for _ in range(n)]
    acc = np.zeros(shape, dtype=np.uint8)
    for idx in range(n - 1, -1, -1):
        unions[idx] = acc.copy()
        palette_idx = order[idx]
        acc = cv2.bitwise_or(acc, (labels == palette_idx).astype(np.uint8) * 255)
    return unions


def build_paint_masks_economical(
    base_masks: Dict[int, np.ndarray],
    labels: np.ndarray,
    order: List[int],
    overpaint_mm: float,
    max_side: int,
    gamma: float = 1.5,
) -> Dict[int, np.ndarray]:
    """Build paint masks that avoid redundant coats while keeping registration easy.

    Each layer paints:
      1. Its final color region (from quantization) — required for the finished image.
      2. An optional registration fringe on shared borders with later colors only
         (controlled by overpaint_mm, gamma-scaled for early layers).

    Unlike the legacy pipeline, we do not fill holes with paint where a later layer
  will cover, and we do not subtract earlier layers from later masks (each pixel has
  one owning color in the final image).
    """
    px_per_mm = max_side / 1000.0
    r_px_base = max(1, round(overpaint_mm * px_per_mm))
    N = len(order)
    later_unions = _build_later_label_unions(labels, order)
    paint_masks: Dict[int, np.ndarray] = {}

    for idx, palette_idx in enumerate(order):
        core = base_masks[palette_idx].copy()
        if np.sum(core) == 0:
            paint_masks[palette_idx] = core
            continue

        paint_mask = core
        if overpaint_mm > 0 and idx < N - 1:
            scale = (1 - idx / max(1, N - 1)) ** gamma
            if idx == 0:
                scale *= 0.45
            r_px = max(1, round(r_px_base * scale))
            kernel = np.ones((r_px * 2 + 1, r_px * 2 + 1), np.uint8)
            dilated = cv2.dilate(core, kernel, iterations=1)
            fringe = cv2.bitwise_and(dilated, cv2.bitwise_not(core))
            registration = cv2.bitwise_and(fringe, later_unions[idx])
            paint_mask = cv2.bitwise_or(core, registration)

        paint_masks[palette_idx] = paint_mask

    return paint_masks


def mask_paint_overlap_stats(
    paint_masks: Dict[int, np.ndarray],
    labels: np.ndarray,
    order: List[int],
) -> Dict[str, int]:
    """Summarize how much paint area is scheduled and how much overlaps between layers."""
    if not paint_masks or not order:
        return {
            "labeled_pixels": 0,
            "total_paint_pixels": 0,
            "overlap_pixels": 0,
            "redundant_pixels": 0,
        }
    shape = labels.shape
    labeled_pixels = int(labels.size)
    stack = np.zeros((len(order), shape[0], shape[1]), dtype=np.uint8)
    for i, palette_idx in enumerate(order):
        m = paint_masks.get(palette_idx)
        if m is not None:
            stack[i] = (m > 0).astype(np.uint8)
    per_pixel = np.sum(stack, axis=0)
    total_paint_pixels = int(np.sum(per_pixel))
    overlap_pixels = int(np.sum(per_pixel > 1))

    redundant_pixels = 0
    for palette_idx in order:
        mask = paint_masks.get(palette_idx)
        if mask is None:
            continue
        wrong_color = (mask > 0) & (labels != palette_idx)
        redundant_pixels += int(np.sum(wrong_color))

    return {
        "labeled_pixels": labeled_pixels,
        "total_paint_pixels": total_paint_pixels,
        "overlap_pixels": overlap_pixels,
        "redundant_pixels": redundant_pixels,
    }


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
                     mask_dilation_px: int = 0, easy_painting: bool = False,
                     easy_simplify: float = 0.65, easy_face_detail: bool = True,
                     style_preset: Optional[str] = None,
                     detail_eyes: bool = True,
                     detail_face: bool = True,
                     detail_body_outline: bool = True,
                     favor_skin_tones: bool = False,
                     skin_tone_strength: float = 0.65,
                     priority_region_hash: str = "",
                     priority_region_strength: float = 0.7,
                     must_include_hash: str = "") -> str:
    """Compute cache key from image hash and processing parameters."""
    with open(image_path, 'rb') as f:
        image_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    preset = normalize_style_preset(style_preset)
    params = (
        f"{PIPELINE_VERSION}_{n_colors}_{overpaint_mm:.2f}_{order_mode}_{max_side}_"
        f"{saturation_boost:.2f}_{detail_level:.2f}_d{int(mask_dilation_px)}"
        f"_st{preset}_es{easy_simplify:.2f}"
        f"_fd{1 if easy_face_detail else 0}"
        f"_de{1 if detail_eyes else 0}{1 if detail_face else 0}{1 if detail_body_outline else 0}"
        f"_sk{1 if favor_skin_tones else 0}{skin_tone_strength:.2f}"
        f"_pr{priority_region_hash or '0'}_prs{priority_region_strength:.2f}"
        f"_mi{must_include_hash or '0'}"
    )
    cache_key = f"{image_hash}_{params}"
    logger.info(f"Computed cache key: {cache_key}")
    return cache_key


def get_cache_dir(cache_key: str) -> Path:
    """Get cache directory for a cache key."""
    return MASK_CACHE_DIR / cache_key


def regenerate_pure_mask_from_labels(artifacts_path: Path, layer_index: int) -> bool:
    """Write layer_{layer_index}_pure_mask.png from labels.npy and order.json (exact quantized region, no gaps)."""
    labels_path = artifacts_path / "labels.npy"
    order_path = artifacts_path / "order.json"
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
        out_path = artifacts_path / f"layer_{layer_index}_pure_mask.png"
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


def load_from_cache(cache_dir: Path, artifacts_path: Path, project_id: str, order_mode: str) -> Optional[Dict]:
    """Load masks from cache and copy to project artifacts directory.
    
    Args:
        cache_dir: Path to cache directory
        artifacts_path: Path to project artifacts directory
        project_id: Project UUID for API URLs
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
            shutil.copy2(cached_preview, artifacts_path / "preview.jpg")
            logger.info("Copied preview from cache")
        else:
            logger.warning("Preview not found in cache")
        for name in ("labels.npy", "order.json"):
            src = cache_dir / name
            if src.exists():
                shutil.copy2(src, artifacts_path / name)
        
        # Load palette and cached order
        palette = metadata.get('palette', [])
        cached_order = metadata.get('order', [])
        
        if not palette:
            logger.error("No palette in cache metadata")
            return None
        
        # Reorder layers based on current order_mode
        order = order_layers(palette, order_mode)
        logger.info(f"Reordered layers for mode '{order_mode}': {order}")
        with open(artifacts_path / "order.json", "w") as f:
            json.dump(order, f)
        
        # Pure masks must come from labels/order to guarantee complete pure coverage.
        labels_npy = artifacts_path / "labels.npy"
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
            mask_path = artifacts_path / f'layer_{output_layer_idx}_mask.png'
            shutil.copy2(cached_mask, mask_path)

            # Pure mask: always regenerate from labels/order so pure mode is deterministic.
            pure_mask_path = artifacts_path / f'layer_{output_layer_idx}_pure_mask.png'
            pure_mask = ((labels == palette_idx).astype(np.uint8)) * 255
            cv2.imwrite(str(pure_mask_path), pure_mask)
            pure_masks_for_validation.append(pure_mask)
            pure_url = artifact_url(project_id, f'layer_{output_layer_idx}_pure_mask.png')

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.error(f"Failed to load mask from {mask_path}")
                continue

            for outline_style in ['thin', 'thick', 'glow']:
                outline = generate_outline(mask, outline_style)
                outline_path = artifacts_path / f'layer_{output_layer_idx}_outline_{outline_style}.png'
                cv2.imwrite(str(outline_path), cv2.cvtColor(outline, cv2.COLOR_RGBA2BGRA))

            layers.append({
                'layer_index': output_layer_idx,
                'palette_index': palette_idx,
                'mask_url': artifact_url(project_id, f'layer_{output_layer_idx}_mask.png'),
                'mask_pure_url': pure_url,
                'outline_thin_url': artifact_url(project_id, f'layer_{output_layer_idx}_outline_thin.png'),
                'outline_thick_url': artifact_url(project_id, f'layer_{output_layer_idx}_outline_thick.png'),
                'outline_glow_url': artifact_url(project_id, f'layer_{output_layer_idx}_outline_glow.png'),
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
            'finished_url': artifact_url(project_id, 'preview.jpg'),
            'mask_url': artifact_url(project_id, 'preview.jpg'),
            'outline_thin_url': artifact_url(project_id, 'preview.jpg'),
            'outline_thick_url': artifact_url(project_id, 'preview.jpg'),
            'outline_glow_url': artifact_url(project_id, 'preview.jpg'),
        })
        
        logger.info(f"Successfully loaded {len(layers)} layers from cache")
        return {
            'width': metadata.get('width'),
            'height': metadata.get('height'),
            'palette': palette,
            'order': order,
            'quantized_preview_url': artifact_url(project_id, 'preview.jpg'),
            'layers': layers,
        }
    except Exception as e:
        logger.error(f"Failed to load from cache: {e}", exc_info=True)
        return None


def write_oriented_source_jpeg(image_path: str, source_path: Path) -> bool:
    """Load input file, apply EXIF orientation, write oriented.jpg into project source/."""
    try:
        image = load_rgb_image_oriented(image_path)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(source_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return source_path.exists()
    except Exception as e:
        logger.warning(f"write_oriented_source_jpeg failed: {e}")
        return False


def save_to_cache(cache_dir: Path, artifacts_path: Path, result: Dict):
    """Save processing results to cache.
    
    Args:
        cache_dir: Path to cache directory
        artifacts_path: Path to project artifacts directory (source of files to cache)
        result: Processing result dictionary
    """
    try:
        logger.info(f"Saving to cache: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy preview, labels and order for pure-mask regeneration
        preview_src = artifacts_path / "preview.jpg"
        if preview_src.exists():
            shutil.copy2(preview_src, cache_dir / "preview.jpg")
            logger.info("Cached preview image")
        else:
            logger.warning("Preview image not found to cache")
        for name in ("labels.npy", "order.json"):
            src = artifacts_path / name
            if src.exists():
                shutil.copy2(src, cache_dir / name)
        
        cached_count = 0
        for layer in result['layers']:
            if layer.get('is_finished'):
                continue
            palette_idx = layer['palette_index']
            layer_idx = layer['layer_index']
            mask_src = artifacts_path / f"layer_{layer_idx}_mask.png"
            if mask_src.exists():
                shutil.copy2(mask_src, cache_dir / f"palette_{palette_idx}_mask.png")
                cached_count += 1
            pure_src = artifacts_path / f"layer_{layer_idx}_pure_mask.png"
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
    project_root: Path,
    project_id: str,
    n_colors: int,
    overpaint_mm: float,
    order_mode: str,
    max_side: int,
    saturation_boost: float = 1.0,
    detail_level: float = 0.5,
    mask_dilation_px: int = 0,
    easy_painting: bool = False,
    easy_simplify: float = 0.65,
    easy_face_detail: bool = False,
    detail_eyes: bool = True,
    detail_face: bool = True,
    detail_body_outline: bool = True,
    style_preset: Optional[str] = None,
    favor_skin_tones: bool = False,
    skin_tone_strength: float = 0.65,
    priority_region_path: Optional[str] = None,
    priority_region_strength: float = 0.7,
    must_include_hex: Optional[List[str]] = None,
) -> Dict:
    """Main processing pipeline with caching support."""
    must_hex_norm = _normalize_must_include_hex_list(must_include_hex)
    must_hash = hashlib.sha256(",".join(must_hex_norm).encode()).hexdigest()[:12] if must_hex_norm else ""
    artifacts_path = project_root / "artifacts"
    source_path = project_root / "source"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    source_path.mkdir(parents=True, exist_ok=True)
    oriented_file = source_path / SOURCE_ORIENTED

    logger.info(f"process_image called: image_path={image_path}, n_colors={n_colors}, cache_dir={MASK_CACHE_DIR}")

    pr_hash = ""
    if priority_region_path and Path(priority_region_path).is_file():
        with open(priority_region_path, "rb") as f:
            pr_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    # Compute cache key (gradients removed from pipeline)
    cache_key = compute_cache_key(
        image_path,
        n_colors,
        overpaint_mm,
        order_mode,
        max_side,
        saturation_boost,
        detail_level,
        mask_dilation_px,
        easy_painting=easy_painting,
        easy_simplify=easy_simplify,
        easy_face_detail=easy_face_detail,
        style_preset=style_preset,
        detail_eyes=detail_eyes,
        detail_face=detail_face,
        detail_body_outline=detail_body_outline,
        favor_skin_tones=favor_skin_tones,
        skin_tone_strength=skin_tone_strength,
        priority_region_hash=pr_hash,
        priority_region_strength=priority_region_strength,
        must_include_hash=must_hash,
    )
    preset_norm = normalize_style_preset(style_preset)
    style = resolve_image_style(
        style_preset,
        easy_painting=easy_painting if preset_norm == "none" else None,
        easy_simplify=easy_simplify,
        easy_face_detail=easy_face_detail,
        detail_eyes=detail_eyes,
        detail_face=detail_face,
        detail_body_outline=detail_body_outline,
    )

    # Check cache first
    cached_dir = check_mask_cache(cache_key)
    if cached_dir:
        logger.info(f"Using cached masks for key: {cache_key}")
        result = load_from_cache(cached_dir, artifacts_path, project_id, order_mode)
        if result:
            if write_oriented_source_jpeg(image_path, oriented_file):
                result["original_url"] = source_url(project_id, SOURCE_ORIENTED)
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
    
    # Save an oriented copy of the original image (for projection viewer "original" toggle)
    try:
        cv2.imwrite(str(oriented_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        logger.warning(f"Failed to save oriented source image to {oriented_file}: {e}")
    
    # Step 1: Normalize
    normalized, scale = normalize_image(image, max_side)
    h, w = normalized.shape[:2]
    
    # Step 2: Quantize
    priority_mask = None
    if priority_region_path:
        priority_mask = load_priority_region_mask(priority_region_path, h, w)

    labels, quantized, palette = quantize_for_pipeline(
        normalized,
        n_colors,
        saturation_boost=saturation_boost,
        easy_painting=easy_painting,
        easy_simplify=easy_simplify,
        easy_face_detail=easy_face_detail,
        detail_eyes=detail_eyes,
        detail_face=detail_face,
        detail_body_outline=detail_body_outline,
        style_preset=style_preset,
        favor_skin_tones=favor_skin_tones,
        skin_tone_strength=skin_tone_strength,
        priority_region_mask=priority_mask,
        priority_region_strength=priority_region_strength,
        mask_detail_level=float(detail_level),
        must_include_hex=must_hex_norm or None,
    )

    # Save quantized preview
    preview_path = artifacts_path / 'preview.jpg'
    cv2.imwrite(str(preview_path), cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR))
    
    # Step 3: Clean masks (conservative so pure view has fewer gaps)
    # detail_level: 0.0 = high detail, 1.0 = low detail; min_area_ratio range kept small
    effective_detail = float(detail_level)
    if style.background_simplify > 1e-6:
        s = _clip01(style.background_simplify)
        floor = detail_level + s * (0.82 - detail_level)
        effective_detail = max(detail_level, floor)
    min_area_ratio = 0.00002 + (effective_detail * 0.00038)  # Range: 0.00002 to 0.0004
    
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
    
    # Step 5: Economical paint masks (core region + border registration fringe only)
    expanded_masks = build_paint_masks_economical(
        base_masks, labels, order, overpaint_mm, max_side
    )
    stats = mask_paint_overlap_stats(expanded_masks, labels, order)
    logger.info(
        "Paint mask stats: total_paint=%s labeled=%s overlap=%s redundant=%s",
        stats["total_paint_pixels"],
        stats["labeled_pixels"],
        stats["overlap_pixels"],
        stats["redundant_pixels"],
    )

    # Step 5.5: Ensure every labeled pixel appears on its owning layer (no cross-color fill)
    expanded_masks = ensure_complete_coverage(expanded_masks, order, quantized, labels, palette)

    # Step 5.7: Optional uniform mask dilation to thicken paint regions for projection workflow.
    if mask_dilation_px > 0:
        r = int(max(1, mask_dilation_px))
        kernel = np.ones((r * 2 + 1, r * 2 + 1), np.uint8)
        for palette_idx in order:
            expanded_masks[palette_idx] = cv2.dilate(expanded_masks[palette_idx], kernel, iterations=1)

    # Save labels and order so pure masks can be regenerated on demand (no gaps)
    np.save(str(artifacts_path / "labels.npy"), labels)
    with open(artifacts_path / "order.json", "w") as f:
        json.dump(order, f)

    # Step 6: Generate outlines and save
    layers = []
    regular_start_idx = 0
    pure_masks_for_validation: List[np.ndarray] = []
    for layer_idx, palette_idx in enumerate(order):
        mask = expanded_masks[palette_idx]
        li = regular_start_idx + layer_idx
        mask_path = artifacts_path / f'layer_{li}_mask.png'
        cv2.imwrite(str(mask_path), mask)
        
        # Pure mask: exact quantized region (labels == palette_idx) so pure view has no gaps
        pure_mask = ((labels == palette_idx).astype(np.uint8)) * 255
        pure_masks_for_validation.append(pure_mask)
        pure_mask_path = artifacts_path / f'layer_{li}_pure_mask.png'
        cv2.imwrite(str(pure_mask_path), pure_mask)
        
        # Generate outlines
        for outline_style in ['thin', 'thick', 'glow']:
            outline = generate_outline(mask, outline_style)
            outline_path = artifacts_path / f'layer_{li}_outline_{outline_style}.png'
            cv2.imwrite(str(outline_path), cv2.cvtColor(outline, cv2.COLOR_RGBA2BGRA))
        
        layers.append({
            'layer_index': li,
            'palette_index': palette_idx,
            'mask_url': artifact_url(project_id, f'layer_{li}_mask.png'),
            'mask_pure_url': artifact_url(project_id, f'layer_{li}_pure_mask.png'),
            'outline_thin_url': artifact_url(project_id, f'layer_{li}_outline_thin.png'),
            'outline_thick_url': artifact_url(project_id, f'layer_{li}_outline_thick.png'),
            'outline_glow_url': artifact_url(project_id, f'layer_{li}_outline_glow.png'),
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
        'finished_url': artifact_url(project_id, 'preview.jpg'),
        'mask_url': artifact_url(project_id, 'preview.jpg'),
        'outline_thin_url': artifact_url(project_id, 'preview.jpg'),
        'outline_thick_url': artifact_url(project_id, 'preview.jpg'),
        'outline_glow_url': artifact_url(project_id, 'preview.jpg'),
    })
    
    result = {
        'width': w,
        'height': h,
        'palette': palette,
        'order': order,
        'quantized_preview_url': artifact_url(project_id, 'preview.jpg'),
        'layers': layers,
        'original_url': source_url(project_id, SOURCE_ORIENTED),
    }
    
    # Save to cache
    cache_dir = get_cache_dir(cache_key)
    save_to_cache(cache_dir, artifacts_path, result)
    
    return result
