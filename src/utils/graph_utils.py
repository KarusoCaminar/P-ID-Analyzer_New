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

from src.utils.type_utils import is_valid_bbox

logger = logging.getLogger(__name__)


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
    
    # Early termination: Quick distance check for non-overlapping boxes
    center1_x = bbox1['x'] + bbox1['width'] / 2
    center1_y = bbox1['y'] + bbox1['height'] / 2
    center2_x = bbox2['x'] + bbox2['width'] / 2
    center2_y = bbox2['y'] + bbox2['height'] / 2
    
    max_distance = max(
        (bbox1['width'] + bbox2['width']) / 2,
        (bbox1['height'] + bbox2['height']) / 2
    )
    
    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    
    # If centers are too far apart, boxes don't overlap (optimization)
    if distance > max_distance * 1.5:
        return 0.0
    
    # Standard IoU calculation for potentially overlapping boxes
    x1 = max(bbox1['x'], bbox2['x'])
    y1 = max(bbox1['y'], bbox2['y'])
    x2 = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
    y2 = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    
    if inter == 0:
        return 0.0
    
    area1 = bbox1['width'] * bbox1['height']
    area2 = bbox2['width'] * bbox2['height']
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
                            if bbox.get('x', 0) > 1.0 or bbox.get('y', 0) > 1.0:
                                # Pixel coordinates - normalize first
                                bbox['x'] = bbox.get('x', 0) / self.image_width
                                bbox['y'] = bbox.get('y', 0) / self.image_height
                                bbox['width'] = bbox.get('width', 0) / self.image_width
                                bbox['height'] = bbox.get('height', 0) / self.image_height
                                logger.debug(f"Normalized pixel bbox to normalized coords for element {elem.get('id', 'unknown')}")
                            
                            # Add tile offset (now both in normalized coordinates)
                            bbox['x'] = bbox.get('x', 0) + offset_x
                            bbox['y'] = bbox.get('y', 0) + offset_y
                            
                            # Ensure normalized values stay in [0, 1] range
                            bbox['x'] = max(0.0, min(1.0, bbox['x']))
                            bbox['y'] = max(0.0, min(1.0, bbox['y']))
                            # Ensure width/height don't exceed bounds
                            max_width = 1.0 - bbox['x']
                            max_height = 1.0 - bbox['y']
                            bbox['width'] = max(0.0, min(max_width, bbox.get('width', 0)))
                            bbox['height'] = max(0.0, min(max_height, bbox.get('height', 0)))
                            
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
                                                el_center_x = bbox['x'] + bbox['width'] / 2
                                                el_center_y = bbox['y'] + bbox['height'] / 2
                                                polyline[0] = [el_center_x, el_center_y]
                                            
                                            if conn.get('to_id') == element_id and len(polyline) > 0:
                                                # Adjust end point to element center
                                                el_center_x = bbox['x'] + bbox['width'] / 2
                                                el_center_y = bbox['y'] + bbox['height'] / 2
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
        """Deduplicate elements by IoU matching."""
        # Sort by area (larger first)
        sorted_elements = sorted(
            elements,
            key=lambda e: e.get('bbox', {}).get('width', 0) * e.get('bbox', {}).get('height', 0),
            reverse=True
        )
        
        for elem in sorted_elements:
            if 'bbox' not in elem or not is_valid_bbox(elem['bbox']):
                continue
            
            # Check for matches with existing elements using optimized search
            matched = False
            elem_center_x = elem['bbox']['x'] + elem['bbox']['width'] / 2
            elem_center_y = elem['bbox']['y'] + elem['bbox']['height'] / 2
            
            for existing_id, existing_elem in self.final_elements.items():
                if 'bbox' not in existing_elem or not is_valid_bbox(existing_elem['bbox']):
                    continue
                
                # Quick distance check before IoU calculation (optimization)
                existing_center_x = existing_elem['bbox']['x'] + existing_elem['bbox']['width'] / 2
                existing_center_y = existing_elem['bbox']['y'] + existing_elem['bbox']['height'] / 2
                
                distance = ((elem_center_x - existing_center_x) ** 2 + (elem_center_y - existing_center_y) ** 2) ** 0.5
                max_bbox_size = max(
                    elem['bbox']['width'] + elem['bbox']['height'],
                    existing_elem['bbox']['width'] + existing_elem['bbox']['height']
                )
                
                # Skip if too far apart (optimization)
                if distance > max_bbox_size * 2:
                    continue
                
                iou = calculate_iou(elem['bbox'], existing_elem['bbox'])
                if iou >= self.config.iou_match_threshold:
                    # Merge: keep the one with more complete data
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

        start_point = (
            start_port_bbox['x'] + start_port_bbox['width'] / 2,
            start_port_bbox['y'] + start_port_bbox['height'] / 2
        )
        end_point = (
            end_port_bbox['x'] + end_port_bbox['width'] / 2,
            end_port_bbox['y'] + end_port_bbox['height'] / 2
        )

        best_polyline = None
        min_dist = float('inf')

        for poly in polylines:
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

