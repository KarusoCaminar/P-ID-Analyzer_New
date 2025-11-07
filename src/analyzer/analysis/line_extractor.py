"""
Line Extractor - Contour-based pipeline line extraction.

CRITICAL: Replaced Skeletonization with Contour Detection for robustness.
Skeletonization was unreliable (noise-sensitive, line-width dependent).
Contour Detection is more stable and handles varying line widths better.

Separates pipeline lines from symbol lines using contour detection.
This is critical to avoid confusion between symbol internal lines
and actual pipeline connections.
"""

import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image

# CRITICAL FIX: pytesseract import moved inside function to make dependency truly optional
# This prevents module-level import failures if pytesseract is not installed
TESSERACT_AVAILABLE = None  # Will be set inside _remove_text_labels() function

logger = logging.getLogger(__name__)


class LineExtractor:
    """
    Extracts pipeline lines and separates them from symbol lines using contour detection.
    
    CRITICAL: Uses Contour Detection instead of Skeletonization for robustness.
    Contour Detection is more stable against noise and varying line widths.
    
    This is critical to avoid confusion between:
    - Symbol internal lines (e.g., pump symbol lines)
    - Actual pipeline connections
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Line Extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logic_parameters = config.get('logic_parameters', {})
    
    def extract_pipeline_lines(
        self,
        image_path: str,
        elements: List[Dict[str, Any]],
        excluded_zones: List[Dict[str, float]],
        legend_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract pipeline lines with skeletonization.
        
        Args:
            image_path: Path to image
            elements: Detected elements with bboxes
            excluded_zones: Legend/metadata areas to exclude
            legend_data: Legend data with line_map (colors, styles)
            
        Returns:
            Dictionary with:
            - pipeline_lines: List of skeletonized line segments
            - junctions: List of junction points (splits/merges)
            - line_segments: List of line segments with endpoints
        """
            logger.info("=== Starting contour-based line extraction ===")
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return {'pipeline_lines': [], 'junctions': [], 'line_segments': []}
            
            img_height, img_width = img.shape[:2]
            
            # 1. Mask symbols (exclude bbox areas)
            masked_image = self._mask_symbols(img, elements, excluded_zones, img_width, img_height)
            
            # 1.5. Remove text labels (PRIORITÄT 2: Text-Removal)
            # This prevents text from breaking lines during skeletonization
            text_removed_image = self._remove_text_labels(masked_image)
            
            # 2. Isolate pipeline colors (from legend)
            pipeline_colors = self._extract_pipeline_colors(text_removed_image, legend_data)
            
            # 3. Extract contours (REPLACED: Skeletonization -> Contour Detection)
            # CRITICAL: Contour Detection is more robust than Skeletonization
            contours, polylines = self._extract_contours(pipeline_colors)
            
            # 4. Detect junctions (points where multiple contours meet)
            junctions = self._detect_junctions_from_contours(contours, polylines, img_width, img_height)
            
            # 5. Use polylines directly (already vectorized from contours)
            line_segments = polylines
            
            # 5.5. Bridge gaps (PRIORITÄT 3: Gap-Bridging)
            # This connects segments that were broken by text or noise
            line_segments = self._bridge_gaps(line_segments, img_width, img_height)
            
            # 6. Match to connections
            matched_lines = self._match_to_connections(line_segments, elements, img_width, img_height)
            
            logger.info(f"Line extraction complete: {len(matched_lines)} pipeline lines, "
                       f"{len(junctions)} junctions, {len(line_segments)} segments")
            
            return {
                'pipeline_lines': matched_lines,
                'junctions': junctions,
                'line_segments': line_segments
            }
            
        except Exception as e:
            logger.error(f"Error in line extraction: {e}", exc_info=True)
            return {'pipeline_lines': [], 'junctions': [], 'line_segments': []}
    
    def _mask_symbols(
        self,
        img: np.ndarray,
        elements: List[Dict[str, Any]],
        excluded_zones: List[Dict[str, float]],
        img_width: int,
        img_height: int
    ) -> np.ndarray:
        """
        Mask out symbol areas (bboxes) to avoid confusing symbol lines with pipeline lines.
        
        Args:
            img: Input image
            elements: Detected elements with bboxes
            excluded_zones: Legend/metadata areas to exclude
            img_width: Image width
            img_height: Image height
            
        Returns:
            Masked image with symbol areas removed
        """
        masked = img.copy()
        
        # Create mask for symbol areas
        mask = np.ones((img_height, img_width), dtype=np.uint8) * 255
        
        # Mask element bboxes
        for el in elements:
            bbox = el.get('bbox')
            if not bbox:
                continue
            
            # Convert normalized bbox to pixel coordinates
            x = int(bbox.get('x', 0) * img_width)
            y = int(bbox.get('y', 0) * img_height)
            w = int(bbox.get('width', 0) * img_width)
            h = int(bbox.get('height', 0) * img_height)
            
            # Expand bbox slightly to ensure symbol lines are fully masked
            margin = max(5, int(min(w, h) * 0.1))  # 10% margin or 5px minimum
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_width - x, w + 2 * margin)
            h = min(img_height - y, h + 2 * margin)
            
            # Fill bbox area with white (will be ignored in line detection)
            mask[y:y+h, x:x+w] = 0
        
        # Mask excluded zones (legend, metadata)
        for zone in excluded_zones:
            x = int(zone.get('x', 0) * img_width)
            y = int(zone.get('y', 0) * img_height)
            w = int(zone.get('width', 0) * img_width)
            h = int(zone.get('height', 0) * img_height)
            
            x = max(0, min(img_width, x))
            y = max(0, min(img_height, y))
            w = min(img_width - x, w)
            h = min(img_height - y, h)
            
            mask[y:y+h, x:x+w] = 0
        
        # Apply mask
        masked[mask == 0] = [255, 255, 255]  # White background
        
        return masked
    
    def _extract_pipeline_colors(
        self,
        img: np.ndarray,
        legend_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract pipeline colors based on legend or default colors.
        
        Args:
            img: Input image (masked)
            legend_data: Legend data with line_map
            
        Returns:
            Binary image with pipeline lines only
        """
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Default pipeline colors (black, red, blue)
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        
        # Red ranges
        mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Blue range
        mask_blue = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_black, cv2.bitwise_or(mask_red, mask_blue))
        
        # If legend has line_map, use it for additional colors
        if legend_data:
            line_map = legend_data.get('line_map', {})
            for line_type, line_info in line_map.items():
                if isinstance(line_info, dict):
                    color = line_info.get('color', '').lower()
                    if color == 'green':
                        mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
                        combined_mask = cv2.bitwise_or(combined_mask, mask_green)
        
        return combined_mask
    
    def _extract_contours(
        self,
        binary: np.ndarray
    ) -> Tuple[List[np.ndarray], List[List[List[float]]]]:
        """
        Extract contours from binary image and convert to polylines.
        
        CRITICAL: Replaced Skeletonization with Contour Detection for robustness.
        Contour Detection is more stable against noise and varying line widths.
        
        Args:
            binary: Binary image (0 = background, 255 = foreground)
            
        Returns:
            Tuple of (contours, polylines) where:
            - contours: List of OpenCV contour arrays
            - polylines: List of simplified polylines (list of [x, y] coordinates)
        """
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Convert to binary (0 or 255)
        binary = (binary > 128).astype(np.uint8) * 255
        
        # Find contours
        # RETR_CCOMP: Retrieves all contours and organizes them into a two-level hierarchy
        # CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polylines = []
        img_height, img_width = binary.shape
        
        # Convert contours to simplified polylines
        for contour in contours:
            if len(contour) < 2:
                continue
            
            # Approximate contour to polyline (Douglas-Peucker algorithm)
            # epsilon: Maximum distance from original curve to approximated curve
            # Set epsilon based on image size (smaller for high-res images)
            epsilon = max(1.0, min(img_width, img_height) * 0.001)  # 0.1% of image size
            
            # Approximate contour to polyline
            approx = cv2.approxPolyDP(contour, epsilon, closed=False)
            
            # Convert to list of [x, y] coordinates
            polyline = []
            for point in approx:
                x, y = point[0][0], point[0][1]
                polyline.append([float(x), float(y)])
            
            # Only add polylines with at least 2 points
            if len(polyline) >= 2:
                polylines.append(polyline)
        
        logger.info(f"Extracted {len(contours)} contours, converted to {len(polylines)} polylines")
        
        return contours, polylines
    
    def _detect_junctions_from_contours(
        self,
        contours: List[np.ndarray],
        polylines: List[List[List[float]]],
        img_width: int,
        img_height: int
    ) -> List[Dict[str, Any]]:
        """
        Detect junction points (splits/merges) from contours.
        
        Junction: point where multiple polylines meet (within tolerance).
        
        CRITICAL: Replaced pixel-based junction detection with polyline-based detection.
        This is more robust and handles varying line widths better.
        
        Args:
            contours: List of OpenCV contour arrays
            polylines: List of polylines (list of [x, y] coordinates)
            img_width: Image width
            img_height: Image height
            
        Returns:
            List of junction points with coordinates and degree
        """
        junctions = []
        junction_tolerance = max(5.0, min(img_width, img_height) * 0.01)  # 1% of image size
        
        # Create a map of all polyline endpoints
        endpoint_map = {}
        for polyline in polylines:
            if len(polyline) < 2:
                continue
            
            # Start point
            start = tuple(polyline[0])
            if start not in endpoint_map:
                endpoint_map[start] = []
            endpoint_map[start].append(polyline)
            
            # End point
            end = tuple(polyline[-1])
            if end != start:  # Avoid duplicate for single-point polylines
                if end not in endpoint_map:
                    endpoint_map[end] = []
                endpoint_map[end].append(polyline)
        
        # Find points where multiple polylines meet (junctions)
        visited_junctions = set()
        
        for point, connected_polylines in endpoint_map.items():
            if len(connected_polylines) > 1:
                # Multiple polylines meet at this point = junction
                # Check if we already have a junction nearby (within tolerance)
                is_new_junction = True
                for existing_junction in junctions:
                    dist = np.sqrt(
                        (point[0] - existing_junction['x'])**2 +
                        (point[1] - existing_junction['y'])**2
                    )
                    if dist <= junction_tolerance:
                        is_new_junction = False
                        # Update degree if higher
                        if len(connected_polylines) > existing_junction.get('degree', 0):
                            existing_junction['degree'] = len(connected_polylines)
                        break
                
                if is_new_junction:
                    junctions.append({
                        'x': float(point[0]),
                        'y': float(point[1]),
                        'degree': len(connected_polylines),
                        'id': f"junction_{len(junctions)}"
                    })
        
        logger.info(f"Detected {len(junctions)} junctions from contours")
        return junctions
    
    def _vectorize_segments(
        self,
        skeleton: np.ndarray,
        junctions: List[Dict[str, Any]]
    ) -> List[List[List[float]]]:
        """
        Vectorize skeleton into line segments.
        
        Args:
            skeleton: Skeletonized image
            junctions: List of junction points
            
        Returns:
            List of line segments (polylines)
        """
        segments = []
        
        # Create junction map for fast lookup
        junction_map = {}
        for j in junctions:
            jx, jy = int(j['x']), int(j['y'])
            junction_map[(jy, jx)] = j
        
        # Find all line pixels
        y_coords, x_coords = np.where(skeleton > 0)
        
        visited = set()
        img_height, img_width = skeleton.shape
        
        # Extract segments between junctions
        for y, x in zip(y_coords, x_coords):
            if (y, x) in visited:
                continue
            
            # If this is a junction, skip (handled separately)
            if (y, x) in junction_map:
                continue
            
            # Trace line segment
            segment = self._trace_line_segment(skeleton, x, y, junction_map, visited, img_width, img_height)
            
            if len(segment) > 1:
                segments.append(segment)
        
        logger.info(f"Vectorized {len(segments)} line segments")
        return segments
    
    def _trace_line_segment(
        self,
        skeleton: np.ndarray,
        start_x: int,
        start_y: int,
        junction_map: Dict[Tuple[int, int], Dict[str, Any]],
        visited: set,
        img_width: int,
        img_height: int
    ) -> List[List[float]]:
        """
        Trace a line segment from start point to junction or endpoint.
        
        Args:
            skeleton: Skeletonized image
            start_x: Start x coordinate
            start_y: Start y coordinate
            junction_map: Map of junction coordinates
            visited: Set of visited pixels
            img_width: Image width
            img_height: Image height
            
        Returns:
            List of [x, y] coordinates
        """
        segment = []
        current_x, current_y = start_x, start_y
        
        while True:
            # Add current point
            segment.append([float(current_x), float(current_y)])
            visited.add((current_y, current_x))
            
            # Check if we hit a junction
            if (current_y, current_x) in junction_map:
                break
            
            # Find next neighbor
            next_x, next_y = None, None
            neighbors_found = 0
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = current_y + dy, current_x + dx
                    if 0 <= ny < img_height and 0 <= nx < img_width:
                        if skeleton[ny, nx] > 0 and (ny, nx) not in visited:
                            neighbors_found += 1
                            if next_x is None:
                                next_x, next_y = nx, ny
            
            # End of segment (no more neighbors or multiple neighbors = junction)
            if neighbors_found == 0:
                break
            elif neighbors_found > 1:
                # Multiple neighbors = junction (should have been in map)
                break
            
            current_x, current_y = next_x, next_y
        
        return segment
    
    def _match_to_connections(
        self,
        line_segments: List[List[List[float]]],
        elements: List[Dict[str, Any]],
        img_width: int,
        img_height: int
    ) -> List[Dict[str, Any]]:
        """
        Match line segments to connections between elements.
        
        Args:
            line_segments: List of line segments (polylines)
            elements: Detected elements with bboxes
            img_width: Image width
            img_height: Image height
            
        Returns:
            List of matched pipeline lines with connection info
        """
        matched_lines = []
        
        # Create element map
        element_map = {el.get('id'): el for el in elements}
        
        # Normalize coordinates
        for segment in line_segments:
            if len(segment) < 2:
                continue
            
            # Normalize segment coordinates
            normalized_segment = []
            for point in segment:
                x_norm = point[0] / img_width
                y_norm = point[1] / img_height
                normalized_segment.append([x_norm, y_norm])
            
            # Find closest elements to start and end
            start_point = normalized_segment[0]
            end_point = normalized_segment[-1]
            
            closest_start = self._find_closest_element(start_point, element_map, img_width, img_height)
            closest_end = self._find_closest_element(end_point, element_map, img_width, img_height)
            
            matched_lines.append({
                'polyline': normalized_segment,
                'start_element': closest_start,
                'end_element': closest_end,
                'length': len(segment)
            })
        
        return matched_lines
    
    def _find_closest_element(
        self,
        point: List[float],
        element_map: Dict[str, Dict[str, Any]],
        img_width: int,
        img_height: int
    ) -> Optional[str]:
        """
        Find closest element to a point.
        
        Args:
            point: Normalized [x, y] coordinates
            element_map: Map of element_id -> element
            img_width: Image width
            img_height: Image height
            
        Returns:
            Element ID or None
        """
        px, py = point[0] * img_width, point[1] * img_height
        min_dist = float('inf')
        closest_id = None
        
        for el_id, el in element_map.items():
            bbox = el.get('bbox')
            if not bbox:
                continue
            
            # Calculate element center
            ex = (bbox.get('x', 0) + bbox.get('width', 0) / 2) * img_width
            ey = (bbox.get('y', 0) + bbox.get('height', 0) / 2) * img_height
            
            # Calculate distance
            dist = np.sqrt((px - ex)**2 + (py - ey)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_id = el_id
        
        # PRIORITÄT 1: Adaptive Thresholds
        # Calculate adaptive threshold based on image size
        adaptive_threshold = self._calculate_adaptive_thresholds(img_width, img_height)
        
        # Only return if within reasonable distance (adaptive threshold)
        if min_dist < adaptive_threshold:
            return closest_id
        
        return None
    
    def _remove_text_labels(self, image: np.ndarray) -> np.ndarray:
        """
        Remove text labels from image to prevent line breaks.
        
        PRIORITÄT 2: Text-Removal (Verbesserung 1)
        Uses pytesseract (if available) + CV fallback to find and remove text.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Image with text removed (via inpaint)
        """
        try:
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Use pytesseract if available (more accurate)
            # CRITICAL FIX: Import pytesseract inside function to make dependency truly optional
            # This prevents module-level import failures
            try:
                import pytesseract
                from pytesseract import Output
                tesseract_available = True
            except ImportError:
                tesseract_available = False
                logger.debug("pytesseract not available - using CV fallback for text removal")
            
            if tesseract_available:
                try:
                    # Get text bounding boxes
                    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
                    
                    # Create mask for text regions
                    text_mask = np.zeros(gray.shape, dtype=np.uint8)
                    
                    n_boxes = len(data['text'])
                    for i in range(n_boxes):
                        if int(data['conf'][i]) > 0:  # Confidence > 0
                            x = data['left'][i]
                            y = data['top'][i]
                            w = data['width'][i]
                            h = data['height'][i]
                            
                            # Expand box slightly to ensure complete removal
                            margin = 2
                            x1 = max(0, x - margin)
                            y1 = max(0, y - margin)
                            x2 = min(gray.shape[1], x + w + margin)
                            y2 = min(gray.shape[0], y + h + margin)
                            
                            text_mask[y1:y2, x1:x2] = 255
                    
                    if np.sum(text_mask) > 0:
                        # Use inpaint to remove text
                        result = cv2.inpaint(image, text_mask, 3, cv2.INPAINT_TELEA)
                        logger.info(f"Text removal: Removed {n_boxes} text regions using pytesseract")
                        return result
                    
                except Exception as e:
                    logger.warning(f"pytesseract text removal failed: {e}. Using CV fallback.")
            
            # Method 2: CV fallback - detect text-like regions
            # Use morphological operations to find rectangular regions (likely text)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            horizontal = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            vertical = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Combine horizontal and vertical text regions
            text_mask_cv = cv2.bitwise_or(horizontal, vertical)
            
            # Threshold to get binary mask
            _, text_mask_cv = cv2.threshold(text_mask_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours (likely text regions)
            contours, _ = cv2.findContours(text_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask for inpaint
            text_mask = np.zeros(gray.shape, dtype=np.uint8)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter small regions (likely not text)
                if w > 10 and h > 5 and w * h < (gray.shape[0] * gray.shape[1] * 0.01):  # < 1% of image
                    text_mask[y:y+h, x:x+w] = 255
            
            if np.sum(text_mask) > 0:
                result = cv2.inpaint(image, text_mask, 3, cv2.INPAINT_TELEA)
                logger.info(f"Text removal: Removed text regions using CV fallback")
                return result
            
            # No text found - return original
            return image
            
        except Exception as e:
            logger.error(f"Error in text removal: {e}", exc_info=True)
            return image  # Fallback: return original image
    
    def _calculate_adaptive_thresholds(self, img_width: int, img_height: int) -> float:
        """
        Calculate adaptive threshold based on image size.
        
        PRIORITÄT 1: Adaptive Thresholds (Verbesserung 4)
        Replaces fixed 50px threshold with adaptive value.
        
        Uses gap_bridging_threshold_px from config.yaml if available,
        otherwise calculates adaptive value based on image size.
        
        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Adaptive threshold in pixels
        """
        # Try to get threshold from config.yaml first (user-defined "Stellhebel")
        gap_threshold = self.logic_parameters.get('gap_bridging_threshold_px')
        if gap_threshold is not None:
            logger.debug(f"Using config gap_bridging_threshold_px: {gap_threshold}px")
            return float(gap_threshold)
        
        # Fallback: Calculate adaptive threshold based on image size
        # Calculate image diagonal (approximate scale)
        diagonal = np.sqrt(img_width ** 2 + img_height ** 2)
        
        # Base threshold: 0.5% of diagonal
        # This scales with image size:
        # - Small image (1000px): ~7px threshold
        # - Medium image (4000px): ~28px threshold
        # - Large image (8000px): ~56px threshold
        adaptive_threshold = diagonal * 0.005
        
        # Clamp to reasonable range (min 10px, max 100px)
        adaptive_threshold = max(10, min(100, adaptive_threshold))
        
        logger.debug(f"Adaptive threshold: {adaptive_threshold:.1f}px (image: {img_width}x{img_height})")
        return adaptive_threshold
    
    def _bridge_gaps(
        self,
        line_segments: List[List[List[float]]],
        img_width: int,
        img_height: int
    ) -> List[List[List[float]]]:
        """
        Bridge gaps between line segments.
        
        PRIORITÄT 3: Gap-Bridging (Verbesserung 3)
        Connects segments that were broken by text or noise.
        
        Args:
            line_segments: List of line segments (polylines)
            img_width: Image width
            img_height: Image height
            
        Returns:
            List of line segments with gaps bridged
        """
        if len(line_segments) < 2:
            return line_segments
        
        # Calculate adaptive gap threshold
        gap_threshold = self._calculate_adaptive_thresholds(img_width, img_height)
        
        # Try to merge segments
        merged_segments = []
        used_indices = set()
        
        for i, seg1 in enumerate(line_segments):
            if i in used_indices:
                continue
            
            if len(seg1) < 2:
                merged_segments.append(seg1)
                used_indices.add(i)
                continue
            
            # Try to find segments that connect to this one
            current_seg = seg1.copy()
            merged = True
            
            while merged:
                merged = False
                
                for j, seg2 in enumerate(line_segments):
                    if j in used_indices or j == i:
                        continue
                    
                    if len(seg2) < 2:
                        continue
                    
                    # Check if segments are close enough to bridge
                    seg1_end = current_seg[-1]
                    seg2_start = seg2[0]
                    seg2_end = seg2[-1]
                    
                    # Calculate pixel distances
                    dist_to_start = np.sqrt(
                        ((seg1_end[0] - seg2_start[0]) * img_width) ** 2 +
                        ((seg1_end[1] - seg2_start[1]) * img_height) ** 2
                    )
                    dist_to_end = np.sqrt(
                        ((seg1_end[0] - seg2_end[0]) * img_width) ** 2 +
                        ((seg1_end[1] - seg2_end[1]) * img_height) ** 2
                    )
                    
                    # Bridge gap if close enough
                    if dist_to_start < gap_threshold:
                        # Connect to start of seg2
                        current_seg.extend(seg2)
                        used_indices.add(j)
                        merged = True
                        break
                    elif dist_to_end < gap_threshold:
                        # Connect to end of seg2 (reverse)
                        seg2_reversed = seg2[::-1]
                        current_seg.extend(seg2_reversed)
                        used_indices.add(j)
                        merged = True
                        break
            
            merged_segments.append(current_seg)
            used_indices.add(i)
        
        # Add segments that couldn't be merged
        for i, seg in enumerate(line_segments):
            if i not in used_indices:
                merged_segments.append(seg)
        
        logger.info(f"Gap bridging: {len(line_segments)} -> {len(merged_segments)} segments")
        return merged_segments

