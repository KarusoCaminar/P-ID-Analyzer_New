"""
Image processing utilities for P&ID analysis.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def resize_image_for_llm(input_path: str, max_size: int = 1024) -> str:
    """
    Resize an image so that the longest side is at most max_size pixels.
    
    Args:
        input_path: Path to input image
        max_size: Maximum size for longest side
        
    Returns:
        Path to resized temporary file
    """
    img = Image.open(input_path)
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    new_size = (int(w * scale), int(h * scale))
    
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    
    temp_dir = Path(input_path).parent / "temp_llm"
    temp_dir.mkdir(exist_ok=True)
    out_path = temp_dir / Path(input_path).name
    img_resized.save(out_path)
    
    return str(out_path)


def segment_image(
    input_path: str,
    drawing_area_bbox: Dict[str, float],
    output_folder: Path,
    segment_size: int = 1024,
    overlap_ratio: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Segment an image into overlapping segments.
    
    Args:
        input_path: Path to input image
        drawing_area_bbox: Bounding box for drawing area (normalized)
        output_folder: Folder for output segments
        segment_size: Size of each segment
        overlap_ratio: Overlap ratio between segments
        
    Returns:
        List of segment dictionaries
    """
    try:
        img = Image.open(input_path)
        w, h = img.size
        
        # Calculate absolute pixel area for segmentation
        area_x = int(drawing_area_bbox['x'] * w)
        area_y = int(drawing_area_bbox['y'] * h)
        area_w = int(drawing_area_bbox['width'] * w)
        area_h = int(drawing_area_bbox['height'] * h)
        
        overlap = int(segment_size * overlap_ratio)
        stride = segment_size - overlap
        base = Path(input_path).stem
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        segments = []
        for x in range(area_x, area_x + area_w, stride):
            for y in range(area_y, area_y + area_h, stride):
                box = (x, y, min(x + segment_size, w), min(y + segment_size, h))
                
                # Prevent empty segments at edges
                if box[2] - box[0] < overlap or box[3] - box[1] < overlap:
                    continue
                
                seg_path = output_folder / f"{base}_{x}_{y}.png"
                img.crop(box).save(seg_path)
                
                segments.append({
                    "path": str(seg_path),
                    "original_coords": {
                        "x_abs": box[0],
                        "y_abs": box[1],
                        "width_abs": box[2] - box[0],
                        "height_abs": box[3] - box[1]
                    }
                })
        return segments
    except Exception as e:
        logger.error(f"Error segmenting image {input_path}: {e}", exc_info=True)
        return []


def generate_raster_grid(
    image_path: str,
    tile_size: int = 1024,
    overlap: int = 128,
    excluded_zones: Optional[List[Dict[str, float]]] = None,
    output_folder: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Generate a raster grid of overlapping tiles from an image.
    
    Args:
        image_path: Path to input image
        tile_size: Size of each tile
        overlap: Overlap between tiles
        excluded_zones: Zones to exclude (normalized coordinates)
        output_folder: Folder for output tiles
        
    Returns:
        List of tile dictionaries
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        logger.error(f"Cannot generate raster: Image file not found at {image_path}")
        return []
    
    img_width, img_height = image.size
    logger.info(f"Generating raster grid for image of size {img_width}x{img_height}...")
    
    tiles = []
    temp_dir = output_folder if output_folder else Path(os.path.dirname(image_path)) / "temp_tiles"
    os.makedirs(temp_dir, exist_ok=True)
    
    stride = tile_size - overlap
    
    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            right = min(x + tile_size, img_width)
            bottom = min(y + tile_size, img_height)
            box = (x, y, right, bottom)
            
            tile_is_excluded = False
            if excluded_zones:
                for zone in excluded_zones:
                    ex_x1 = zone['x'] * img_width
                    ex_y1 = zone['y'] * img_height
                    ex_x2 = ex_x1 + zone['width'] * img_width
                    ex_y2 = ex_y1 + zone['height'] * img_height
                    # FIX: Nur ausschließen wenn Tile KOMPLETT in excluded zone liegt (100% Overlap)
                    # OLD: if not (right < ex_x1 or x > ex_x2 or bottom < ex_y1 or y > ex_y2):
                    # NEW: Nur ausschließen wenn Tile komplett innerhalb der Zone liegt
                    # Wenn Tile nur teilweise überlappt → NICHT ausschließen (analysieren)
                    # Dies verhindert dass gültige Diagramm-Bereiche ausgeschlossen werden
                    if (x >= ex_x1 and right <= ex_x2 and y >= ex_y1 and bottom <= ex_y2):
                        # Tile ist komplett in excluded zone → ausschließen
                        tile_is_excluded = True
                        break
            
            if tile_is_excluded:
                logger.debug(f"Skipping tile at ({x},{y}) due to excluded zone (100% overlap).")
                continue
            
            tile_image = image.crop(box)
            tile_path = os.path.join(temp_dir, f"tile_{x}_{y}.png")
            tile_image.save(tile_path)
            tiles.append({
                'path': tile_path,
                'coords': (x, y),
                'tile_width': tile_image.width,
                'tile_height': tile_image.height
            })
    
    logger.info(f"Generated {len(tiles)} tiles in '{temp_dir}'.")
    return tiles


def is_tile_complex(tile_path: str, canny_threshold1: int = 50, canny_threshold2: int = 150) -> bool:
    """
    Check if a tile is complex (has significant content).
    
    Args:
        tile_path: Path to tile image
        canny_threshold1: Canny edge detection threshold 1
        canny_threshold2: Canny edge detection threshold 2
        
    Returns:
        True if complex, False otherwise
    """
    try:
        img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        
        edges = cv2.Canny(img, canny_threshold1, canny_threshold2)
        edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Consider complex if edge ratio > 0.01
        return edge_ratio > 0.01
    except Exception as e:
        logger.error(f"Error checking tile complexity for {tile_path}: {e}")
        return True  # Default to complex if error


def has_legend_cv(image_path: str) -> bool:
    """
    Prüft ob das Diagramm eine Legende hat (CV-basiert).
    
    Args:
        image_path: Pfad zum Bild
        
    Returns:
        True wenn Legende erkannt wurde, False sonst
    """
    legend_bbox = find_legend_rectangle_cv(image_path, search_region='top_left')
    return legend_bbox is not None


def find_text_stamp_cv(image_path: str) -> Optional[Dict[str, float]]:
    """
    Findet Text-Stamp/Metadata-Feld (typischerweise unten rechts) mit OpenCV.
    
    Args:
        image_path: Pfad zum Bild
        
    Returns:
        Dictionary mit normalisierten Koordinaten {'x', 'y', 'width', 'height'} oder None
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"CV: Bild konnte nicht geladen werden: {image_path}")
            return None
        
        img_height, img_width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Fokus auf untere rechte Ecke (typischer Ort für Text-Stamp)
        # Suche in den letzten 30% der Breite und Höhe
        bottom_right_region = gray[int(img_height * 0.7):, int(img_width * 0.7):]
        
        # Schwellenwert für Text-Erkennung (Text ist meist dunkel)
        _, thresh = cv2.threshold(bottom_right_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Konturen finden
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_rect = None
        max_area = 0
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Text-Stamp sollte eine Mindestgröße haben und rechteckig sein
            if area > (img_width * img_height * 0.001) and w > h * 1.5 and area > max_area:
                max_area = area
                # Koordinaten zurück auf vollständiges Bild umrechnen
                best_rect = {
                    "x": (int(img_width * 0.7) + x) / img_width,
                    "y": (int(img_height * 0.7) + y) / img_height,
                    "width": w / img_width,
                    "height": h / img_height
                }
        
        if best_rect:
            logger.info(f"CV: Text-Stamp gefunden bei: {best_rect}")
            return best_rect
        else:
            logger.info("CV: Kein Text-Stamp gefunden.")
            return None
            
    except Exception as e:
        logger.error(f"Fehler bei CV-Text-Stamp-Erkennung: {e}", exc_info=True)
        return None


def find_legend_rectangle_cv(image_path: str, search_region: str = 'all') -> Optional[Dict[str, float]]:
    """
    Findet das größte, schwarze Rechteck (Legende) mit OpenCV.
    
    Gibt normalisierte Koordinaten zurück.
    
    Args:
        image_path: Pfad zum Bild
        search_region: 'all', 'top_left', 'top_right', 'bottom_left', 'bottom_right'
        
    Returns:
        Dictionary mit normalisierten Koordinaten {'x', 'y', 'width', 'height'} oder None
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"CV: Bild konnte nicht geladen werden: {image_path}")
            return None
        
        img_height, img_width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Region-basierte Suche (Legende ist typischerweise oben links)
        if search_region == 'top_left':
            search_area = gray[0:int(img_height * 0.5), 0:int(img_width * 0.5)]
            offset_x, offset_y = 0, 0
        elif search_region == 'top_right':
            search_area = gray[0:int(img_height * 0.5), int(img_width * 0.5):]
            offset_x, offset_y = int(img_width * 0.5), 0
        elif search_region == 'bottom_left':
            search_area = gray[int(img_height * 0.5):, 0:int(img_width * 0.5)]
            offset_x, offset_y = 0, int(img_height * 0.5)
        elif search_region == 'bottom_right':
            search_area = gray[int(img_height * 0.5):, int(img_width * 0.5):]
            offset_x, offset_y = int(img_width * 0.5), int(img_height * 0.5)
        else:
            search_area = gray
            offset_x, offset_y = 0, 0
        
        # Schwellenwert, um schwarze Kanten zu isolieren (sehr aggressiv)
        # Wir suchen nach Werten < 50 (fast schwarz)
        _, thresh = cv2.threshold(search_area, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Konturen finden
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_rect = None
        max_area = 0
        
        for cnt in contours:
            # Kontur approximieren, um Ecken zu bekommen
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Prüfen, ob es (ungefähr) ein Rechteck ist und eine Mindestgröße hat
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                
                # Es muss eine signifikante Größe haben (z.B. > 1% des Bildes)
                if area > (img_width * img_height * 0.01) and area > max_area:
                    max_area = area
                    # Koordinaten zurück auf vollständiges Bild umrechnen
                    best_rect = {
                        "x": (offset_x + x) / img_width,
                        "y": (offset_y + y) / img_height,
                        "width": w / img_width,
                        "height": h / img_height
                    }
        
        if best_rect:
            logger.info(f"CV: Legenden-Rechteck gefunden bei: {best_rect}")
            return best_rect
        else:
            logger.info("CV: Kein klares Legenden-Rechteck gefunden.")
            return None
            
    except Exception as e:
        logger.error(f"Fehler bei CV-Rechteckerkennung: {e}", exc_info=True)
        return None


def crop_image_for_correction(
    image_path: str,
    bbox: Dict[str, float],
    context_margin: float = 0.05
) -> Optional[str]:
    """
    Crop the original image to a specific bounding box plus a margin for context.
    Returns the path to the temporary cropped image file.
    
    Args:
        image_path: Path to original image
        bbox: Bounding box dictionary with 'x', 'y', 'width', 'height'
               (can be normalized 0-1 or pixel coordinates)
        context_margin: Margin ratio for context (default 0.05 = 5%)
        
    Returns:
        Path to cropped temporary file or None if error
    """
    import uuid
    
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Validate bbox structure
        if not all(k in bbox and isinstance(bbox[k], (int, float)) for k in ['x', 'y', 'width', 'height']):
            logger.error(f"Invalid bbox structure for crop_image_for_correction: {bbox}")
            return None
        if bbox['width'] <= 0 or bbox['height'] <= 0:
            logger.error(f"Invalid bbox dimensions for crop_image_for_correction: {bbox}")
            return None

        # Detect if bbox is normalized (0-1 range) or pixel coordinates
        # If max value is <= 1.0, assume normalized; otherwise assume pixel coordinates
        max_val = max(bbox['x'], bbox['y'], bbox['width'], bbox['height'])
        is_normalized = max_val <= 1.0 and bbox['x'] >= 0 and bbox['y'] >= 0
        
        if is_normalized:
            # Convert normalized coordinates to pixel coordinates
            x1 = int(bbox['x'] * img_width)
            y1 = int(bbox['y'] * img_height)
            x2 = int((bbox['x'] + bbox['width']) * img_width)
            y2 = int((bbox['y'] + bbox['height']) * img_height)
        else:
            # Already in pixel coordinates
            x1 = int(bbox['x'])
            y1 = int(bbox['y'])
            x2 = int(bbox['x'] + bbox['width'])
            y2 = int(bbox['y'] + bbox['height'])

        margin_x = int((x2 - x1) * context_margin)
        margin_y = int((y2 - y1) * context_margin)

        # Ensure minimum crop size (at least 10 pixels)
        min_crop_size = 10
        if x2 - x1 < min_crop_size:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - min_crop_size // 2)
            x2 = min(img_width, center_x + min_crop_size // 2)
        if y2 - y1 < min_crop_size:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - min_crop_size // 2)
            y2 = min(img_height, center_y + min_crop_size // 2)
            # Ensure we have at least min_crop_size
            if y2 - y1 < min_crop_size:
                y2 = min(img_height, y1 + min_crop_size)

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(img_width, x2 + margin_x)
        crop_y2 = min(img_height, y2 + margin_y)

        # Ensure valid crop coordinates (minimum size after margin)
        if crop_x2 - crop_x1 < min_crop_size:
            # Expand to minimum size
            center_x = (crop_x1 + crop_x2) // 2
            crop_x1 = max(0, center_x - min_crop_size // 2)
            crop_x2 = min(img_width, center_x + min_crop_size // 2)
            # Ensure we have at least min_crop_size
            if crop_x2 - crop_x1 < min_crop_size:
                crop_x2 = min(img_width, crop_x1 + min_crop_size)
        if crop_y2 - crop_y1 < min_crop_size:
            # Expand to minimum size
            center_y = (crop_y1 + crop_y2) // 2
            crop_y1 = max(0, center_y - min_crop_size // 2)
            crop_y2 = min(img_height, center_y + min_crop_size // 2)
            # Ensure we have at least min_crop_size
            if crop_y2 - crop_y1 < min_crop_size:
                crop_y2 = min(img_height, crop_y1 + min_crop_size)

        # Final validation - ensure we have valid dimensions
        if crop_x1 >= crop_x2:
            crop_x2 = min(img_width, crop_x1 + min_crop_size)
        if crop_y1 >= crop_y2:
            crop_y2 = min(img_height, crop_y1 + min_crop_size)
        
        if crop_x2 - crop_x1 < 1 or crop_y2 - crop_y1 < 1:
            logger.error(f"Invalid crop coordinates: ({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}). Original bbox: {bbox}")
            return None

        cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        temp_dir = Path(image_path).parent / "temp_tiles"
        temp_dir.mkdir(exist_ok=True)
        # Use UUID to avoid filename conflicts
        temp_path = temp_dir / f"correction_snippet_{uuid.uuid4().hex[:8]}.png"
        
        cropped_image.save(temp_path)
        logger.info(f"Created correction snippet: {temp_path}")
        return str(temp_path)

    except Exception as e:
        logger.error(f"Error creating cropped image for correction: {e}", exc_info=True)
        return None


def preprocess_image_for_line_detection(image_path: str, output_path: str) -> str:
    """
    Isolate black, red, and blue lines in an image to facilitate polyline extraction by LLM.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        
    Returns:
        Path to processed image (or original if error)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image at: {image_path}")
            return image_path  # Return original path to avoid crash

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Color ranges for line detection
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        combined_mask = cv2.bitwise_or(mask_black, cv2.bitwise_or(mask_red, mask_blue))

        result = np.full(img.shape, 255, dtype=np.uint8)
        result[combined_mask > 0] = img[combined_mask > 0]
        
        cv2.imwrite(output_path, result)
        return output_path
    except Exception as e:
        logger.error(f"Error preprocessing image for line detection {image_path}: {e}", exc_info=True)
        return image_path

