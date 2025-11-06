"""
Graph utilities for P&ID analysis.

Optimized algorithms for graph operations including:
- IoU calculation with early termination
- Efficient connection deduplication
- Optimized graph synthesis with spatial indexing
- Predictive graph completion
"""

import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from src.utils.type_utils import is_valid_bbox

logger = logging.getLogger(__name__)

# Try to import thefuzz for fuzzy string matching (Levenshtein distance)
try:
    from thefuzz import fuzz
    FUZZ_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        FUZZ_AVAILABLE = True
    except ImportError:
        FUZZ_AVAILABLE = False
        logger.warning("thefuzz/fuzzywuzzy not available. Install with: pip install thefuzz. Duplicate fusion will use basic string matching only.")


def calculate_iou(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Optimized version with early termination for non-overlapping boxes.
    Uses center-point distance check for fast rejection of distant boxes.
    
    Args:
        bbox1: First bounding box {x, y, width, height}
        bbox2: Second bounding box {x, y, width, height}
        
    Returns:
        IoU value (0.0 to 1.0)
    """
    if not is_valid_bbox(bbox1) or not is_valid_bbox(bbox2):
        return 0.0
    
    # CRITICAL FIX: Convert all bbox values to float to handle string values from JSON
    bbox1_x = float(bbox1.get('x', 0))
    bbox1_y = float(bbox1.get('y', 0))
    bbox1_width = float(bbox1.get('width', 0))
    bbox1_height = float(bbox1.get('height', 0))
    bbox2_x = float(bbox2.get('x', 0))
    bbox2_y = float(bbox2.get('y', 0))
    bbox2_width = float(bbox2.get('width', 0))
    bbox2_height = float(bbox2.get('height', 0))
    
    # Early termination: Quick distance check for non-overlapping boxes
    center1_x = bbox1_x + bbox1_width / 2
    center1_y = bbox1_y + bbox1_height / 2
    center2_x = bbox2_x + bbox2_width / 2
    center2_y = bbox2_y + bbox2_height / 2
    
    max_distance = max(
        (bbox1_width + bbox2_width) / 2,
        (bbox1_height + bbox2_height) / 2
    )
    
    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    
    # If centers are too far apart, boxes don't overlap (optimization)
    if distance > max_distance * 1.5:
        return 0.0
    
    # Standard IoU calculation for potentially overlapping boxes
    x1 = max(bbox1_x, bbox2_x)
    y1 = max(bbox1_y, bbox2_y)
    x2 = min(bbox1_x + bbox1_width, bbox2_x + bbox2_width)
    y2 = min(bbox1_y + bbox1_height, bbox2_y + bbox2_height)
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    
    if inter == 0:
        return 0.0
    
    # CRITICAL FIX: Use already converted float values
    area1 = bbox1_width * bbox1_height
    area2 = bbox2_width * bbox2_height
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def dedupe_connections(conns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate connections (same from/to in any order).
    
    Args:
        conns: List of connection dictionaries
        
    Returns:
        Deduplicated list of connections
    """
    seen: Set[Tuple[str, str]] = set()
    kept: List[Dict[str, Any]] = []
    
    for c in conns:
        a = c.get("from_id")
        b = c.get("to_id")
        if not a or not b:
            continue
        
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        
        seen.add(key)
        kept.append(c)
    
    return kept


def propagate_connection_confidence(
    connections: List[Dict[str, Any]],
    elements: List[Dict[str, Any]],
    method: str = "min"
) -> List[Dict[str, Any]]:
    """
    Propagate confidence from elements to connections.
    
    Connection confidence is calculated based on the confidence of the
    elements it connects. This ensures consistency between element and connection confidences.
    
    Args:
        connections: List of connection dictionaries with 'from_id' and 'to_id'
        elements: List of element dictionaries with 'id' and 'confidence'
        method: Method to calculate connection confidence from element confidences
               - "min": Use minimum of from/to element confidences (conservative)
               - "weighted_avg": Use weighted average (balanced)
    
    Returns:
        Updated connections with propagated confidence scores
    """
    # Build element map for fast lookup
    element_map = {el.get('id'): el for el in elements if el.get('id')}
    
    updated_connections = []
    for conn in connections:
        updated_conn = conn.copy()
        
        from_id = conn.get('from_id')
        to_id = conn.get('to_id')
        
        from_el = element_map.get(from_id)
        to_el = element_map.get(to_id)
        
        if not from_el or not to_el:
            # If elements not found, keep original confidence or set default
            if 'confidence' not in updated_conn:
                updated_conn['confidence'] = 0.3
            updated_connections.append(updated_conn)
            continue
        
        # Get element confidences
        from_conf = from_el.get('confidence', 0.5)
        to_conf = to_el.get('confidence', 0.5)
        
        # Calculate connection confidence based on method
        if method == "min":
            conn_confidence = min(from_conf, to_conf)
        elif method == "weighted_avg":
            # Weighted average (favor higher confidence)
            conn_confidence = (from_conf * 0.5) + (to_conf * 0.5)
        else:
            # Default: use minimum
            conn_confidence = min(from_conf, to_conf)
        
        # Update connection confidence
        updated_conn['confidence'] = max(0.0, min(1.0, conn_confidence))
        updated_connections.append(updated_conn)
    
    return updated_connections


@dataclass
class SynthesizerConfig:
    """Configuration for GraphSynthesizer."""
    iou_match_threshold: float = 0.5
    border_coords_tolerance_factor_x: float = 0.015
    border_coords_tolerance_factor_y: float = 0.015


class GraphSynthesizer:
    """
    Synthesizes a graph from multiple tile results.
    
    This is a simplified version - full implementation will be added incrementally.
    """
    
    def __init__(
        self,
        raw_results: List[Any],  # List of TileResult-like objects
        image_width: int,
        image_height: int,
        config: Optional[SynthesizerConfig] = None
    ):
        if not image_width > 0 or not image_height > 0:
            raise ValueError("Image dimensions must be positive.")
        
        self.raw_results = raw_results
        self.image_width = image_width
        self.image_height = image_height
        self.config = config or SynthesizerConfig()
        
        self.border_tolerance_x_px = self.config.border_coords_tolerance_factor_x * self.image_width
        self.border_tolerance_y_px = self.config.border_coords_tolerance_factor_y * self.image_height
        
        self._canonical_id_map: Dict[str, str] = {}
        self.final_elements: Dict[str, Dict[str, Any]] = {}
        self.final_connections: List[Dict[str, Any]] = []
    
    def synthesize(self) -> Dict[str, Any]:
        """
        Synthesize graph from tile results.
        
        Returns:
            Dictionary with 'elements' and 'connections' keys
        """
        logger.info("Starting graph synthesis...")
        
        # Collect all elements and connections
        all_elements: List[Dict[str, Any]] = []
        all_connections: List[Dict[str, Any]] = []
        
        # CV-based BBox refinement enabled?
        use_cv_refinement = getattr(self.config, 'use_cv_bbox_refinement', True)
        
        for tile_result in self.raw_results:
            if hasattr(tile_result, 'data'):
                tile_data = tile_result.data
            elif isinstance(tile_result, dict) and 'data' in tile_result:
                tile_data = tile_result['data']
            else:
                tile_data = tile_result
            
            if isinstance(tile_data, dict):
                elements = tile_data.get('elements', [])
                connections = tile_data.get('connections', [])
                
                # Adjust coordinates if tile_result has tile_coords
                tile_coords = None
                if hasattr(tile_result, 'tile_coords'):
                    tile_coords = tile_result.tile_coords
                elif isinstance(tile_result, dict):
                    tile_coords = tile_result.get('tile_coords')
                
                if tile_coords:
                    # tile_coords can be (offset_x, offset_y) or (x, y, width, height) tuple
                    if len(tile_coords) == 2:
                        offset_x, offset_y = tile_coords
                    else:
                        # If tile_coords is (x, y, width, height), extract offset
                        offset_x = tile_coords[0] if len(tile_coords) > 0 else 0
                        offset_y = tile_coords[1] if len(tile_coords) > 1 else 0
                    
                    # Normalize offsets if they're in pixel coordinates
                    # Check if offsets are > 1.0 (likely pixel coords) and normalize
                    # CRITICAL FIX: Convert to float before comparison
                    offset_x = float(offset_x) if offset_x is not None else 0.0
                    offset_y = float(offset_y) if offset_y is not None else 0.0
                    if offset_x > 1.0 or offset_y > 1.0:
                        # Pixel coordinates - normalize them
                        offset_x_norm = offset_x / self.image_width
                        offset_y_norm = offset_y / self.image_height
                        offset_x = offset_x_norm
                        offset_y = offset_y_norm
                    
                    for elem in elements:
                        if 'bbox' in elem and isinstance(elem['bbox'], dict):
                            bbox = elem['bbox']
                            # Check if bbox is in pixel coordinates (> 1.0) and normalize
                            # CRITICAL FIX: Convert to float to handle string values from JSON
                            x_val = float(bbox.get('x', 0)) if bbox.get('x') is not None else 0.0
                            y_val = float(bbox.get('y', 0)) if bbox.get('y') is not None else 0.0
                            if x_val > 1.0 or y_val > 1.0:
                                # Pixel coordinates - normalize first
                                # CRITICAL FIX: Convert to float before division
                                bbox['x'] = float(bbox.get('x', 0)) / self.image_width
                                bbox['y'] = float(bbox.get('y', 0)) / self.image_height
                                bbox['width'] = float(bbox.get('width', 0)) / self.image_width
                                bbox['height'] = float(bbox.get('height', 0)) / self.image_height
                                logger.debug(f"Normalized pixel bbox to normalized coords for element {elem.get('id', 'unknown')}")
                            
                            # Add tile offset (now both in normalized coordinates)
                            # CRITICAL FIX: Convert to float before addition
                            bbox['x'] = float(bbox.get('x', 0)) + offset_x
                            bbox['y'] = float(bbox.get('y', 0)) + offset_y
                            
                            # Ensure normalized values stay in [0, 1] range
                            # CRITICAL FIX: Convert to float before comparison
                            bbox['x'] = max(0.0, min(1.0, float(bbox.get('x', 0))))
                            bbox['y'] = max(0.0, min(1.0, float(bbox.get('y', 0))))
                            # Ensure width/height don't exceed bounds
                            # CRITICAL FIX: Convert to float before calculation
                            max_width = 1.0 - float(bbox.get('x', 0))
                            max_height = 1.0 - float(bbox.get('y', 0))
                            bbox['width'] = max(0.0, min(max_width, float(bbox.get('width', 0))))
                            bbox['height'] = max(0.0, min(max_height, float(bbox.get('height', 0))))
                            
                            # Adjust connection coordinates if they reference this element
                            element_id = elem.get('id')
                            if element_id:
                                for conn in connections:
                                    # Adjust connections that reference this element
                                    if conn.get('from_id') == element_id or conn.get('to_id') == element_id:
                                        # If connection has polyline, adjust it to match new bbox
                                        if 'polyline' in conn and conn['polyline']:
                                            polyline = conn['polyline']
                                            # Adjust first/last point to element center
                                            if conn.get('from_id') == element_id and len(polyline) > 0:
                                                # Adjust start point to element center
                                                # CRITICAL FIX: Convert to float before arithmetic
                                                el_center_x = float(bbox.get('x', 0)) + float(bbox.get('width', 0)) / 2
                                                el_center_y = float(bbox.get('y', 0)) + float(bbox.get('height', 0)) / 2
                                                polyline[0] = [el_center_x, el_center_y]
                                            
                                            if conn.get('to_id') == element_id and len(polyline) > 0:
                                                # Adjust end point to element center
                                                # CRITICAL FIX: Convert to float before arithmetic
                                                el_center_x = float(bbox.get('x', 0)) + float(bbox.get('width', 0)) / 2
                                                el_center_y = float(bbox.get('y', 0)) + float(bbox.get('height', 0)) / 2
                                                polyline[-1] = [el_center_x, el_center_y]
                                            
                                            conn['polyline'] = polyline
                
                all_elements.extend(elements)
                all_connections.extend(connections)
        
        # Deduplicate elements by IoU
        self._deduplicate_elements(all_elements)
        
        # Deduplicate connections
        self.final_connections = dedupe_connections(all_connections)
        
        logger.info(f"Synthesis complete: {len(self.final_elements)} elements and {len(self.final_connections)} connections.")
        
        return {
            'elements': list(self.final_elements.values()),
            'connections': self.final_connections
        }
    
    def _deduplicate_elements(self, elements: List[Dict[str, Any]]) -> None:
        """
        Deduplicate elements by ID + Label + IoU with smart deduplication.
        
        Uses:
        1. ID-based deduplication (same ID = duplicate)
        2. Label-based deduplication (similar labels + IoU)
        3. IoU-based deduplication (spatial clustering + class-aware matching)
        """
        # FIX 1: ID-based deduplication (same ID = duplicate)
        elements_by_id = {}
        for elem in elements:
            elem_id = elem.get('id', '')
            if elem_id:
                if elem_id not in elements_by_id:
                    elements_by_id[elem_id] = []
                elements_by_id[elem_id].append(elem)
        
        # For each ID set: keep best version (highest confidence + smallest BBox)
        id_deduplicated = []
        for elem_id, elem_list in elements_by_id.items():
            if len(elem_list) > 1:
                # Choose best version: highest confidence * (1 / area)
                best_elem = max(elem_list, key=lambda e: (
                    e.get('confidence', 0.5) * 
                    (1.0 / max(e.get('bbox', {}).get('width', 1) * e.get('bbox', {}).get('height', 1), 0.001))
                ))
                id_deduplicated.append(best_elem)
                logger.debug(f"ID deduplication: {elem_id} - kept best version out of {len(elem_list)} duplicates")
            else:
                id_deduplicated.extend(elem_list)
        
        # FIX 2: Label-based deduplication with Levenshtein distance (similar labels + IoU)
        # Normalize labels for matching
        def normalize_label(label: str) -> str:
            if not label:
                return ""
            return label.strip().replace(' ', '-').replace('_', '-').lower()
        
        elements_by_normalized_label = {}
        for elem in id_deduplicated:
            label = elem.get('label', '')
            normalized_label = normalize_label(label)
            if normalized_label:
                if normalized_label not in elements_by_normalized_label:
                    elements_by_normalized_label[normalized_label] = []
                elements_by_normalized_label[normalized_label].append(elem)
        
        # Deduplicate similar labels with IoU check
        label_deduplicated = []
        for normalized_label, label_elements in elements_by_normalized_label.items():
            if len(label_elements) > 1:
                # Check IoU between elements with same label
                unique_elements = []
                for elem in label_elements:
                    is_duplicate = False
                    for unique_elem in unique_elements:
                        if is_valid_bbox(elem.get('bbox', {})) and is_valid_bbox(unique_elem.get('bbox', {})):
                            iou = calculate_iou(elem['bbox'], unique_elem['bbox'])
                            if iou >= 0.3:  # Same label + high IoU = duplicate
                                # Keep better version
                                # CRITICAL FIX: Convert to float before multiplication
                                elem_precision = elem.get('confidence', 0.5) / max(float(elem['bbox'].get('width', 1)) * float(elem['bbox'].get('height', 1)), 0.001)
                                unique_precision = unique_elem.get('confidence', 0.5) / max(float(unique_elem['bbox'].get('width', 1)) * float(unique_elem['bbox'].get('height', 1)), 0.001)
                                if elem_precision > unique_precision:
                                    unique_elements.remove(unique_elem)
                                    unique_elements.append(elem)
                                is_duplicate = True
                                break
                    if not is_duplicate:
                        unique_elements.append(elem)
                label_deduplicated.extend(unique_elements)
                if len(label_elements) != len(unique_elements):
                    logger.debug(f"Label deduplication: {normalized_label} - {len(label_elements)} → {len(unique_elements)}")
            else:
                label_deduplicated.extend(label_elements)
        
        # FIX 2.5: Aggressive duplicate fusion with Levenshtein distance (for similar but not identical labels)
        # This catches cases like "ISA" vs "(Instrument Air Supply)" or "ISA-Supply" vs "Instrument Air Supply"
        if FUZZ_AVAILABLE:
            fuzz_deduplicated = []
            used_indices = set()
            
            for i, elem_a in enumerate(label_deduplicated):
                if i in used_indices:
                    continue
                
                label_a = elem_a.get('label', '')
                bbox_a = elem_a.get('bbox', {})
                
                if not label_a or not is_valid_bbox(bbox_a):
                    fuzz_deduplicated.append(elem_a)
                    continue
                
                # Check against all remaining elements
                merged_group = [elem_a]
                for j, elem_b in enumerate(label_deduplicated[i+1:], start=i+1):
                    if j in used_indices:
                        continue
                    
                    label_b = elem_b.get('label', '')
                    bbox_b = elem_b.get('bbox', {})
                    
                    if not label_b or not is_valid_bbox(bbox_b):
                        continue
                    
                    # Calculate fuzzy string similarity (token set ratio handles word order differences)
                    similarity = fuzz.token_set_ratio(label_a, label_b)
                    
                    # Calculate IoU
                    iou = calculate_iou(bbox_a, bbox_b)
                    
                    # If similarity > 80% AND IoU > 0.05, treat as duplicate
                    if similarity > 80 and iou > 0.05:
                        merged_group.append(elem_b)
                        used_indices.add(j)
                        logger.debug(f"Fuzzy duplicate fusion: '{label_a}' <-> '{label_b}' (similarity={similarity}%, IoU={iou:.3f})")
                
                # Keep element with highest confidence from merged group
                if len(merged_group) > 1:
                    best_elem = max(merged_group, key=lambda e: e.get('confidence', 0.5))
                    fuzz_deduplicated.append(best_elem)
                    logger.info(f"Fuzzy duplicate fusion: Merged {len(merged_group)} elements (kept: '{best_elem.get('label', '')}')")
                else:
                    fuzz_deduplicated.append(elem_a)
            
            label_deduplicated = fuzz_deduplicated
        
        # FIX 3: IoU-based deduplication (spatial clustering + class-aware matching) - as before
        # Smart Deduplication: Group by class first (class-aware matching)
        elements_by_class = {}
        for elem in label_deduplicated:
            elem_type = elem.get('type', 'Unknown')
            if elem_type not in elements_by_class:
                elements_by_class[elem_type] = []
            elements_by_class[elem_type].append(elem)
        
        # Deduplicate within each class first (more efficient)
        for elem_type, class_elements in elements_by_class.items():
            # Sort by precision (confidence * IoU / area) - higher precision first
            sorted_elements = sorted(
                class_elements,
                key=lambda e: (
                    e.get('confidence', 0.5) * 
                    (float(e.get('bbox', {}).get('width', 0)) * float(e.get('bbox', {}).get('height', 0)))
                ),
                reverse=True
            )
            
            for elem in sorted_elements:
                if 'bbox' not in elem or not is_valid_bbox(elem['bbox']):
                    continue
                
                # Check for matches with existing elements using optimized search
                matched = False
                # CRITICAL FIX: Convert to float before arithmetic operations
                elem_bbox_x = float(elem['bbox'].get('x', 0))
                elem_bbox_y = float(elem['bbox'].get('y', 0))
                elem_bbox_width = float(elem['bbox'].get('width', 0))
                elem_bbox_height = float(elem['bbox'].get('height', 0))
                elem_center_x = elem_bbox_x + elem_bbox_width / 2
                elem_center_y = elem_bbox_y + elem_bbox_height / 2
                
                for existing_id, existing_elem in self.final_elements.items():
                    if 'bbox' not in existing_elem or not is_valid_bbox(existing_elem['bbox']):
                        continue
                    
                    # Quick distance check before IoU calculation (optimization)
                    # CRITICAL FIX: Convert to float before arithmetic operations
                    existing_bbox_x = float(existing_elem['bbox'].get('x', 0))
                    existing_bbox_y = float(existing_elem['bbox'].get('y', 0))
                    existing_bbox_width = float(existing_elem['bbox'].get('width', 0))
                    existing_bbox_height = float(existing_elem['bbox'].get('height', 0))
                    existing_center_x = existing_bbox_x + existing_bbox_width / 2
                    existing_center_y = existing_bbox_y + existing_bbox_height / 2
                    
                    distance = ((elem_center_x - existing_center_x) ** 2 + (elem_center_y - existing_center_y) ** 2) ** 0.5
                    max_bbox_size = max(
                        elem_bbox_width + elem_bbox_height,
                        existing_bbox_width + existing_bbox_height
                    )
                    
                    # Skip if too far apart (optimization)
                    if distance > max_bbox_size * 2:
                        continue
                    
                    iou = calculate_iou(elem['bbox'], existing_elem['bbox'])
                    if iou >= self.config.iou_match_threshold:
                        # Calculate precision scores for both elements
                        # Precision = (Confidence * IoU) / BBox_Area
                        # Higher precision = better (smaller area, higher confidence)
                        
                        # CRITICAL FIX: Convert to float before multiplication
                        elem_bbox_area = float(elem['bbox'].get('width', 0)) * float(elem['bbox'].get('height', 0))
                        existing_bbox_area = float(existing_elem['bbox'].get('width', 0)) * float(existing_elem['bbox'].get('height', 0))
                        
                        elem_confidence = elem.get('confidence', 0.5)
                        existing_confidence = existing_elem.get('confidence', 0.5)
                        
                        # Calculate precision scores
                        elem_precision = (elem_confidence * iou) / max(elem_bbox_area, 0.001)  # Avoid division by zero
                        existing_precision = (existing_confidence * iou) / max(existing_bbox_area, 0.001)
                        
                        # Prefer element with higher precision (smaller area, higher confidence)
                        if elem_precision > existing_precision:
                            # Replace existing with more precise element
                            self.final_elements[existing_id] = elem
                            logger.debug(f"Replaced element {existing_id} with more precise version "
                                       f"(precision: {elem_precision:.4f} > {existing_precision:.4f})")
                        else:
                            # Keep existing element (it's more precise)
                            logger.debug(f"Kept existing element {existing_id} (precision: {existing_precision:.4f} >= {elem_precision:.4f})")
                        
                        matched = True
                        break
                
                if not matched:
                    elem_id = elem.get('id', f"elem_{len(self.final_elements)}")
                    self.final_elements[elem_id] = elem


def match_polylines_to_connections(
    elements: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    polylines: List[List[List[float]]]
) -> List[Dict[str, Any]]:
    """
    Match a list of raw polylines to the best matching connections.
    
    Args:
        elements: List of element dictionaries
        connections: List of connection dictionaries
        polylines: List of polyline coordinate lists
        
    Returns:
        Updated connections with matched polylines
    """
    if not polylines:
        return connections

    id_to_el = {el.get('id'): el for el in elements}

    for conn in connections:
        from_el = id_to_el.get(conn.get('from_id'))
        to_el = id_to_el.get(conn.get('to_id'))
        if not from_el or not to_el or not from_el.get('bbox') or not to_el.get('bbox'):
            continue

        # Find port bboxes or use element bbox
        start_port_bbox = next(
            (p['bbox'] for p in from_el.get('ports', []) if p.get('id') == conn.get('from_port_id')),
            from_el['bbox']
        )
        end_port_bbox = next(
            (p['bbox'] for p in to_el.get('ports', []) if p.get('id') == conn.get('to_port_id')),
            to_el['bbox']
        )

        # CRITICAL FIX: Convert to float before arithmetic operations
        start_point = (
            float(start_port_bbox.get('x', 0)) + float(start_port_bbox.get('width', 0)) / 2,
            float(start_port_bbox.get('y', 0)) + float(start_port_bbox.get('height', 0)) / 2
        )
        end_point = (
            float(end_port_bbox.get('x', 0)) + float(end_port_bbox.get('width', 0)) / 2,
            float(end_port_bbox.get('y', 0)) + float(end_port_bbox.get('height', 0)) / 2
        )

        best_polyline = None
        min_dist = float('inf')

        for poly in polylines:
            if not poly or len(poly) == 0:
                continue  # Skip empty polylines
            poly_start = poly[0]
            poly_end = poly[-1]
            
            # Calculate distance for both directions
            dist1 = np.linalg.norm(np.array(start_point) - np.array(poly_start)) + \
                   np.linalg.norm(np.array(end_point) - np.array(poly_end))
            dist2 = np.linalg.norm(np.array(start_point) - np.array(poly_end)) + \
                   np.linalg.norm(np.array(end_point) - np.array(poly_start))
            
            dist = min(dist1, dist2)
            if dist < min_dist:
                min_dist = dist
                best_polyline = poly if dist1 <= dist2 else poly[::-1]

        if best_polyline:
            conn['polyline'] = best_polyline
    
    return connections


def predict_and_complete_graph(
    elements: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    logger_instance: Optional[logging.Logger] = None,
    distance_threshold: float = 50.0
) -> List[Dict[str, Any]]:
    """
    Predict and complete missing connections in the graph using geometric heuristics.
    
    Uses geometric heuristics to add probable, missing connections between
    nearby, unconnected elements.
    
    Args:
        elements: List of elements with 'id' and 'bbox'
        connections: List of existing connections
        logger_instance: Optional logger instance
        distance_threshold: Distance threshold for predicting connections
        
    Returns:
        Complete list of connections (including predicted ones)
    """
    import networkx as nx
    import uuid
    
    if logger_instance is None:
        logger_instance = logger
    
    logger_instance.info("Starting predictive graph completion to close gaps...")
    
    # Build directed graph
    G = nx.DiGraph()
    node_positions = {}
    
    for el in elements:
        if el.get('id') and el.get('bbox'):
            G.add_node(el['id'])
            pos = el['bbox']
            node_positions[el['id']] = (pos['x'] + pos['width'] / 2, pos['y'] + pos['height'] / 2)
    
    for conn in connections:
        from_id = conn.get('from_id')
        to_id = conn.get('to_id')
        if from_id in G and to_id in G:
            G.add_edge(from_id, to_id)
    
    # Find isolated nodes
    G_undirected = G.to_undirected()
    isolated_nodes = list(nx.isolates(G_undirected))
    
    if not isolated_nodes:
        logger_instance.info("No isolated nodes found. Predictive completion not necessary.")
        return connections
    
    logger_instance.info(f"Executing heuristic for {len(isolated_nodes)} isolated nodes...")
    newly_predicted_connections = []
    
    # Iterate through all unique pairs of isolated nodes
    for i, node1_id in enumerate(isolated_nodes):
        if node1_id not in node_positions:
            continue
        
        pos1 = node_positions[node1_id]
        
        for node2_id in isolated_nodes[i + 1:]:
            if node2_id not in node_positions:
                continue
            
            pos2 = node_positions[node2_id]
            
            # Calculate Euclidean distance
            distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
            
            if distance < distance_threshold:
                logger_instance.info(
                    f"PREDICTION: Adding probable connection between close, isolated nodes: "
                    f"{node1_id} <-> {node2_id} (Distance: {distance:.3f})"
                )
                
                new_conn: Dict[str, Any] = {
                    "id": str(uuid.uuid4()),
                    "from_id": node1_id,
                    "to_id": node2_id,
                    "from_port_id": "",
                    "to_port_id": "",
                    "color": None,
                    "style": None,
                    "predicted": True,
                    "status": "predicted_by_heuristic",
                    "original_border_coords": None
                }
                newly_predicted_connections.append(new_conn)
                G.add_edge(node1_id, node2_id)
    
    # Return original connections plus newly predicted ones
    result = connections + newly_predicted_connections
    if newly_predicted_connections:
        logger_instance.info(f"{len(newly_predicted_connections)} connection gaps predicted.")
    
    return result

