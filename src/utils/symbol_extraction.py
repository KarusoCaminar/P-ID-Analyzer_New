"""
Symbol Extraction Utilities - Advanced methods for extracting symbols from collections.

Uses combination of:
- Computer Vision (OpenCV): Contour detection, edge detection, morphological operations
- OCR-like methods: Text detection for labels
- LLM-based refinement: Precise bounding box and type detection

This module can be used both for:
1. Pretraining: Extracting symbols from collections
2. Pipeline analysis: Refining bounding boxes and detecting text regions
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Use optimized calculate_iou from graph_utils instead of local implementation
from src.utils.graph_utils import calculate_iou as _calculate_iou

logger = logging.getLogger(__name__)


def extract_symbols_with_cv(
    image_path: str,
    min_symbol_size: int = 50,
    max_symbol_size: int = 500,
    text_padding: int = 10
) -> List[Dict[str, Any]]:
    """
    Extract symbols from image using Computer Vision methods.
    
    Uses OpenCV for:
    - Contour detection
    - Edge detection
    - Morphological operations
    - Text region detection
    
    Args:
        image_path: Path to collection image
        min_symbol_size: Minimum symbol size in pixels
        max_symbol_size: Maximum symbol size in pixels
        text_padding: Padding around text regions
        
    Returns:
        List of symbol dictionaries with bbox and metadata
    """
    symbols = []
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return symbols
        
        img_height, img_width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Contour-based detection
        contours = _detect_symbol_contours(gray, min_symbol_size, max_symbol_size)
        
        # Method 2: Edge-based detection
        edges = cv2.Canny(gray, 50, 150)
        edge_contours = _detect_edge_regions(edges, min_symbol_size, max_symbol_size)
        
        # Method 3: Text region detection (for labels)
        text_regions = _detect_text_regions(gray, text_padding)
        
        # Combine and refine detections
        all_regions = []
        
        # Add contour-based detections
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            all_regions.append({
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'method': 'contour',
                'confidence': 0.8
            })
        
        # Add edge-based detections
        for region in edge_contours:
            all_regions.append({
                'bbox': region,
                'method': 'edge',
                'confidence': 0.7
            })
        
        # Merge overlapping regions
        merged_regions = _merge_overlapping_regions(all_regions, img_width, img_height)
        
        # Add text regions to symbols (for labels)
        for symbol_region in merged_regions:
            # Find nearby text regions
            symbol_bbox = symbol_region['bbox']
            nearby_text = _find_nearby_text(symbol_bbox, text_regions)
            
            symbol_region['text_regions'] = nearby_text
            symbol_region['normalized_bbox'] = {
                'x': symbol_bbox['x'] / img_width,
                'y': symbol_bbox['y'] / img_height,
                'width': symbol_bbox['width'] / img_width,
                'height': symbol_bbox['height'] / img_height
            }
            
            symbols.append(symbol_region)
        
        logger.info(f"CV extraction found {len(symbols)} symbol regions in {Path(image_path).name}")
        
    except Exception as e:
        logger.error(f"Error in CV symbol extraction: {e}", exc_info=True)
    
    return symbols


def _detect_symbol_contours(
    gray: np.ndarray,
    min_size: int,
    max_size: int
) -> List[np.ndarray]:
    """Detect symbol contours using OpenCV."""
    contours = []
    
    try:
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        detected_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in detected_contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter by size
            if min_size <= w <= max_size and min_size <= h <= max_size:
                # Filter by area ratio (symbols should be reasonably filled)
                bbox_area = w * h
                if bbox_area > 0 and area / bbox_area > 0.3:  # At least 30% filled
                    contours.append(contour)
        
    except Exception as e:
        logger.warning(f"Error in contour detection: {e}")
    
    return contours


def _detect_edge_regions(
    edges: np.ndarray,
    min_size: int,
    max_size: int
) -> List[Dict[str, int]]:
    """Detect symbol regions using edge detection."""
    regions = []
    
    try:
        # Find contours in edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if min_size <= w <= max_size and min_size <= h <= max_size:
                # Check edge density (symbols should have significant edges)
                roi = edges[y:y+h, x:x+w]
                edge_density = np.sum(roi > 0) / (w * h)
                
                if edge_density > 0.1:  # At least 10% edge pixels
                    regions.append({'x': x, 'y': y, 'width': w, 'height': h})
        
    except Exception as e:
        logger.warning(f"Error in edge region detection: {e}")
    
    return regions


def _detect_text_regions(
    gray: np.ndarray,
    padding: int = 10
) -> List[Dict[str, int]]:
    """
    Detect text regions using morphological operations.
    
    Text regions are typically:
    - Horizontal lines/strips
    - High density of edges
    - Connected components
    """
    text_regions = []
    
    try:
        # Use horizontal kernel for text detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        
        # Detect horizontal lines (text)
        detected = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, horizontal_kernel)
        _, binary = cv2.threshold(detected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find text regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Text regions are typically wider than tall
            if w > h * 2 and w > 20 and h > 5:
                # Add padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2 * padding)
                h = min(gray.shape[0] - y, h + 2 * padding)
                
                text_regions.append({'x': x, 'y': y, 'width': w, 'height': h})
        
    except Exception as e:
        logger.warning(f"Error in text region detection: {e}")
    
    return text_regions


def _merge_overlapping_regions(
    regions: List[Dict[str, Any]],
    img_width: int,
    img_height: int,
    iou_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """Merge overlapping regions to avoid duplicates."""
    if not regions:
        return []
    
    merged = []
    used = set()
    
    for i, region1 in enumerate(regions):
        if i in used:
            continue
        
        bbox1 = region1['bbox']
        best_region = region1
        best_confidence = region1.get('confidence', 0.5)
        
        # Find overlapping regions
        for j, region2 in enumerate(regions[i+1:], start=i+1):
            if j in used:
                continue
            
            bbox2 = region2['bbox']
            # Convert int bboxes to float format for calculate_iou (normalized format)
            bbox1_float = {k: float(v) for k, v in bbox1.items()}
            bbox2_float = {k: float(v) for k, v in bbox2.items()}
            iou = _calculate_iou(bbox1_float, bbox2_float)
            
            if iou > iou_threshold:
                # Merge regions (take larger bbox with higher confidence)
                conf2 = region2.get('confidence', 0.5)
                if conf2 > best_confidence:
                    best_region = region2
                    best_confidence = conf2
                used.add(j)
        
        # Expand bbox to include all merged regions
        merged_bbox = best_region['bbox'].copy()
        for j, region2 in enumerate(regions):
            if j != i and j not in used:
                bbox2 = region2['bbox']
                # Convert int bboxes to float format for calculate_iou (normalized format)
                merged_bbox_float = {k: float(v) for k, v in merged_bbox.items()}
                bbox2_float = {k: float(v) for k, v in bbox2.items()}
                iou = _calculate_iou(merged_bbox_float, bbox2_float)
                if iou > iou_threshold:
                    # Expand bbox
                    merged_bbox['x'] = min(merged_bbox['x'], bbox2['x'])
                    merged_bbox['y'] = min(merged_bbox['y'], bbox2['y'])
                    merged_bbox['width'] = max(
                        merged_bbox['x'] + merged_bbox['width'],
                        bbox2['x'] + bbox2['width']
                    ) - merged_bbox['x']
                    merged_bbox['height'] = max(
                        merged_bbox['y'] + merged_bbox['height'],
                        bbox2['y'] + bbox2['height']
                    ) - merged_bbox['y']
                    used.add(j)
        
        best_region['bbox'] = merged_bbox
        best_region['confidence'] = best_confidence
        merged.append(best_region)
        used.add(i)
    
    return merged


# NOTE: _calculate_iou is now imported from graph_utils (optimized version with early termination)
# The local implementation has been removed to avoid code duplication


def _find_nearby_text(
    symbol_bbox: Dict[str, int],
    text_regions: List[Dict[str, int]],
    max_distance: int = 50
) -> List[Dict[str, int]]:
    """Find text regions near a symbol."""
    nearby = []
    
    symbol_center_x = symbol_bbox['x'] + symbol_bbox['width'] // 2
    symbol_center_y = symbol_bbox['y'] + symbol_bbox['height'] // 2
    
    for text_region in text_regions:
        text_center_x = text_region['x'] + text_region['width'] // 2
        text_center_y = text_region['y'] + text_region['height'] // 2
        
        distance = np.sqrt(
            (symbol_center_x - text_center_x)**2 + 
            (symbol_center_y - text_center_y)**2
        )
        
        if distance <= max_distance:
            nearby.append(text_region)
    
    return nearby


def refine_symbol_bbox_with_cascade(
    symbol_image: Image.Image,
    initial_bbox: Optional[Dict[str, float]] = None,
    max_iterations: int = 3,
    iou_targets: List[float] = [0.7, 0.85, 0.95],
    min_improvement: float = 0.02
) -> Dict[str, float]:
    """
    Cascade BBox Regression: Iterative refinement with increasing precision.
    
    Iteration 1: Coarse refinement (IoU target: 0.7)
    Iteration 2: Medium refinement (IoU target: 0.85)
    Iteration 3: Fine refinement (IoU target: 0.95)
    Stop when IoU improvement < min_improvement per iteration
    
    Args:
        symbol_image: PIL Image of symbol
        initial_bbox: Optional initial bbox hint (normalized 0-1)
        max_iterations: Maximum number of refinement iterations
        iou_targets: List of IoU targets for each iteration
        min_improvement: Minimum IoU improvement per iteration to continue
        
    Returns:
        Refined bounding box (normalized 0-1)
    """
    try:
        current_bbox = initial_bbox or {'x': 0, 'y': 0, 'width': 1, 'height': 1}
        img_width, img_height = symbol_image.size
        
        for iteration in range(min(max_iterations, len(iou_targets))):
            iou_target = iou_targets[iteration]
            
            # Refine bbox for this iteration
            refined_bbox = refine_symbol_bbox_with_cv(
                symbol_image,
                initial_bbox=current_bbox,
                use_anchor_method=True
            )
            
            # Calculate IoU improvement
            if iteration > 0:
                # Calculate IoU between previous and current bbox
                prev_bbox_px = {
                    'x': int(current_bbox['x'] * img_width),
                    'y': int(current_bbox['y'] * img_height),
                    'width': int(current_bbox['width'] * img_width),
                    'height': int(current_bbox['height'] * img_height)
                }
                curr_bbox_px = {
                    'x': int(refined_bbox['x'] * img_width),
                    'y': int(refined_bbox['y'] * img_height),
                    'width': int(refined_bbox['width'] * img_width),
                    'height': int(refined_bbox['height'] * img_height)
                }
                
                from src.utils.graph_utils import calculate_iou
                iou = calculate_iou(prev_bbox_px, curr_bbox_px)
                
                # Check if we've reached the target IoU for this iteration
                if iou >= iou_target:
                    logger.debug(f"Cascade BBox regression: Reached target IoU {iou_target} at iteration {iteration+1}")
                    # Continue to next iteration to refine further
                else:
                    # Calculate improvement: how much better is current vs previous
                    # If IoU is very high (>0.9), we're close to perfect, so continue
                    # If IoU is low (<0.5), we need more improvement
                    if iou < 0.5:
                        # Low IoU means boxes are quite different - check if area improved
                        prev_area = prev_bbox_px['width'] * prev_bbox_px['height']
                        curr_area = curr_bbox_px['width'] * curr_bbox_px['height']
                        area_improvement = (prev_area - curr_area) / prev_area if prev_area > 0 else 0
                        
                        # If area didn't improve significantly, stop
                        if area_improvement < min_improvement:
                            logger.debug(f"Cascade BBox regression: Stopping at iteration {iteration+1} "
                                       f"(IoU: {iou:.3f}, area improvement: {area_improvement:.3f} < {min_improvement})")
                            break
            
            current_bbox = refined_bbox
        
        return current_bbox
    
    except Exception as e:
        logger.warning(f"Error in cascade BBox regression: {e}")
        return initial_bbox or {'x': 0, 'y': 0, 'width': 1, 'height': 1}


def refine_symbol_bbox_with_cv(
    symbol_image: Image.Image,
    initial_bbox: Optional[Dict[str, float]] = None,
    use_anchor_method: bool = True,
    text_regions: Optional[List[Dict[str, int]]] = None,
    include_text: bool = False
) -> Dict[str, float]:
    """
    Refine symbol bounding box using Computer Vision with anchor-based centering.
    
    Uses anchor method: Find contours, then center and crop symbol precisely.
    
    STRATEGIC TEXT-INTEGRATION:
    - For Viewshots: include_text=False (NUR Symbol, Text lenkt ab)
    - For Pretraining: include_text=True (Symbol + Text, OCR braucht Text)
    - For Legend/Diagramm: include_text=False (Text wird separat extrahiert)
    
    Args:
        symbol_image: PIL Image of symbol
        initial_bbox: Optional initial bbox hint
        use_anchor_method: Use anchor-based centering (finds contours, centers symbol)
        text_regions: Optional list of text regions (for Pretraining)
        include_text: If True, expand bbox to include text regions (ONLY for Pretraining)
        
    Returns:
        Refined bounding box (normalized 0-1)
    """
    try:
        # Convert to numpy
        img_array = np.array(symbol_image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_width, img_height = symbol_image.size
        
        if use_anchor_method:
            # ANCHOR METHOD: Find contours, then center symbol
            # Step 1: Multiple threshold methods for robust detection
            _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, binary2 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Combine both methods
            binary = cv2.bitwise_or(binary1, binary2)
            
            # Morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # Fallback: Use edge detection
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # Final fallback
                if initial_bbox:
                    return initial_bbox
                return {'x': 0, 'y': 0, 'width': 1, 'height': 1}
            
            # Find largest contour (main symbol)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # ANCHOR-BASED CENTERING: Find center of mass and adjust
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Center bbox on center of mass
                center_x = cx
                center_y = cy
                
                # Adjust bbox to be centered on center of mass
                x = max(0, center_x - w // 2)
                y = max(0, center_y - h // 2)
                
                # Ensure bbox doesn't exceed image bounds
                if x + w > img_width:
                    x = img_width - w
                if y + h > img_height:
                    y = img_height - h
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
            
            # Add small padding for context (5% on each side)
            padding = min(w, h) * 0.05
            x = max(0, int(x - padding))
            y = max(0, int(y - padding))
            w = min(img_width - x, int(w + 2 * padding))
            h = min(img_height - y, int(h + 2 * padding))
            
            # STRATEGIC TEXT-INTEGRATION: Only for Pretraining
            if include_text and text_regions and len(text_regions) > 0:
                # Expand bbox to include text regions (ONLY for Pretraining)
                min_x = x
                min_y = y
                max_x = x + w
                max_y = y + h
                
                for text_region in text_regions:
                    tx = text_region.get('x', 0)
                    ty = text_region.get('y', 0)
                    tw = text_region.get('width', 0)
                    th = text_region.get('height', 0)
                    
                    # Expand bbox to include text
                    min_x = min(min_x, tx)
                    min_y = min(min_y, ty)
                    max_x = max(max_x, tx + tw)
                    max_y = max(max_y, ty + th)
                
                # Update bbox (ensure within image bounds)
                x = max(0, min_x)
                y = max(0, min_y)
                w = min(img_width - x, max_x - x)
                h = min(img_height - y, max_y - y)
            
            # Normalize
            return {
                'x': x / img_width,
                'y': y / img_height,
                'width': w / img_width,
                'height': h / img_height
            }
        else:
            # Simple method: Binary threshold
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                if initial_bbox:
                    return initial_bbox
                return {'x': 0, 'y': 0, 'width': 1, 'height': 1}
            
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Normalize
            return {
                'x': x / img_width,
                'y': y / img_height,
                'width': w / img_width,
                'height': h / img_height
            }
        
    except Exception as e:
        logger.warning(f"Error refining bbox with CV: {e}")
        if initial_bbox:
            return initial_bbox
        return {'x': 0, 'y': 0, 'width': 1, 'height': 1}

