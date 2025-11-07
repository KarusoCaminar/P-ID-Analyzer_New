"""
Topology Critic - Graph-based topology validation.

Validates the consistency of the pipeline network topology:
- Node degrees (connections per element)
- Connectivity (all nodes reachable)
- Path consistency (polylines match connections)
- Missing splits/merges
"""

import logging
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class TopologyCritic:
    """
    Validates topology and finds inconsistencies in the pipeline network.
    
    Checks:
    - Disconnected nodes (elements without connections)
    - Invalid node degrees (impossible connection counts)
    - Broken paths (polylines don't match connections)
    - Missing splits/merges (detected in graph but not as elements)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Topology Critic.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logic_parameters = config.get('logic_parameters', {})
    
    def validate_topology(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        polylines: Optional[List[List[List[float]]]] = None,
        pipeline_lines: Optional[List[Dict[str, Any]]] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate topology and find inconsistencies.
        
        CRITICAL: Now includes CV-based line verification (Pattern 4 from Audit).
        Verifies that LLM-detected connections have actual physical line paths
        in the CV-extracted pipeline lines.
        
        Args:
            elements: Detected elements
            connections: Detected connections
            polylines: Optional list of polylines for each connection
            pipeline_lines: Optional list of CV-extracted pipeline lines from line_extractor
                          Format: [{'polyline': [[x, y], ...], 'start_element': id, 'end_element': id}, ...]
            image_width: Optional image width (required if pipeline_lines provided)
            image_height: Optional image height (required if pipeline_lines provided)
            
        Returns:
            Dictionary with validation results:
            - disconnected_nodes: Nodes without connections
            - invalid_degrees: Nodes with impossible degrees
            - broken_paths: Paths that don't connect properly
            - missing_splits: Detected splits without Line_Split elements
            - missing_merges: Detected merges without Line_Merge elements
            - unverified_connections: Connections without physical CV line verification
            - validation_score: Overall validation score (0-100)
        """
        logger.info("=== Starting topology validation ===")
        
        try:
            # Build graph
            graph = self._build_graph(elements, connections)
            
            # Check node degrees
            invalid_degrees = self._check_node_degrees(graph, elements)
            
            # Check connectivity
            disconnected = self._find_disconnected_nodes(graph, elements)
            
            # Check polyline consistency
            broken_paths = []
            if polylines:
                broken_paths = self._validate_polylines(connections, polylines)
            
            # CRITICAL: CV-based line verification (Pattern 4)
            unverified_connections = []
            if pipeline_lines and image_width and image_height:
                unverified_connections = self._verify_connections_with_cv_lines(
                    elements, connections, pipeline_lines, image_width, image_height
                )
                logger.info(f"CV line verification: {len(unverified_connections)} unverified connections found")
            
            # Check for missing splits/merges
            missing_splits, missing_merges = self._check_splits_merges(graph, elements)
            
            # Calculate validation score (include unverified connections as issues)
            total_issues = (
                len(disconnected) +
                len(invalid_degrees) +
                len(broken_paths) +
                len(unverified_connections) +
                len(missing_splits) +
                len(missing_merges)
            )
            
            total_elements = len(elements)
            validation_score = max(0, 100 - (total_issues / max(1, total_elements) * 100))
            
            logger.info(f"Topology validation complete: {total_issues} issues found, "
                       f"score: {validation_score:.2f}")
            
            return {
                'disconnected_nodes': disconnected,
                'invalid_degrees': invalid_degrees,
                'broken_paths': broken_paths,
                'unverified_connections': unverified_connections,
                'missing_splits': missing_splits,
                'missing_merges': missing_merges,
                'validation_score': validation_score,
                'total_issues': total_issues
            }
            
        except Exception as e:
            logger.error(f"Error in topology validation: {e}", exc_info=True)
            return {
                'disconnected_nodes': [],
                'invalid_degrees': [],
                'broken_paths': [],
                'unverified_connections': [],
                'missing_splits': [],
                'missing_merges': [],
                'validation_score': 0.0,
                'total_issues': 0
            }
    
    def _build_graph(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        """
        Build directed graph from elements and connections.
        
        Args:
            elements: Detected elements
            connections: Detected connections
            
        Returns:
            NetworkX directed graph
        """
        graph = nx.DiGraph()
        
        # Add nodes (elements)
        for el in elements:
            el_id = el.get('id')
            if el_id:
                graph.add_node(el_id, **el)
        
        # Add edges (connections)
        for conn in connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            
            if from_id and to_id and from_id in graph and to_id in graph:
                graph.add_edge(from_id, to_id, **conn)
        
        return graph
    
    def _check_node_degrees(
        self,
        graph: nx.DiGraph,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check for nodes with invalid degrees (impossible connection counts).
        
        Args:
            graph: Network graph
            elements: Detected elements
            
        Returns:
            List of nodes with invalid degrees
        """
        invalid = []
        
        element_types_to_check = {
            'Pump': {'min_in': 1, 'max_in': 1, 'min_out': 1, 'max_out': 1},
            'Valve': {'min_in': 1, 'max_in': 1, 'min_out': 1, 'max_out': 1},
            'Tank': {'min_in': 0, 'max_in': 10, 'min_out': 0, 'max_out': 10},
            'Heat Exchanger': {'min_in': 2, 'max_in': 2, 'min_out': 2, 'max_out': 2},
        }
        
        for node_id in graph.nodes():
            in_degree = graph.in_degree(node_id)
            out_degree = graph.out_degree(node_id)
            
            # Find element type
            element = next((el for el in elements if el.get('id') == node_id), None)
            if not element:
                continue
            
            el_type = element.get('type', '')
            
            # Check against expected degree ranges
            if el_type in element_types_to_check:
                constraints = element_types_to_check[el_type]
                
                if in_degree < constraints['min_in'] or in_degree > constraints['max_in']:
                    invalid.append({
                        'element_id': node_id,
                        'element_type': el_type,
                        'issue': f"Invalid in-degree: {in_degree} (expected {constraints['min_in']}-{constraints['max_in']})",
                        'in_degree': in_degree,
                        'out_degree': out_degree
                    })
                
                if out_degree < constraints['min_out'] or out_degree > constraints['max_out']:
                    invalid.append({
                        'element_id': node_id,
                        'element_type': el_type,
                        'issue': f"Invalid out-degree: {out_degree} (expected {constraints['min_out']}-{constraints['max_out']})",
                        'in_degree': in_degree,
                        'out_degree': out_degree
                    })
        
        return invalid
    
    def _find_disconnected_nodes(
        self,
        graph: nx.DiGraph,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find disconnected nodes (elements without connections).
        
        Args:
            graph: Network graph
            elements: Detected elements
            
        Returns:
            List of disconnected nodes
        """
        disconnected = []
        
        for el in elements:
            el_id = el.get('id')
            if not el_id or el_id not in graph:
                continue
            
            # Check if node has any connections
            if graph.degree(el_id) == 0:
                disconnected.append({
                    'element_id': el_id,
                    'element_type': el.get('type', 'Unknown'),
                    'issue': 'No connections (isolated node)'
                })
        
        return disconnected
    
    def _validate_polylines(
        self,
        connections: List[Dict[str, Any]],
        polylines: List[List[List[float]]]
    ) -> List[Dict[str, Any]]:
        """
        Validate that polylines match connections.
        
        Args:
            connections: Detected connections
            polylines: List of polylines (one per connection)
            
        Returns:
            List of broken paths
        """
        broken = []
        
        if len(polylines) != len(connections):
            broken.append({
                'issue': f"Polyline count mismatch: {len(polylines)} polylines vs {len(connections)} connections"
            })
            return broken
        
        for i, (conn, poly) in enumerate(zip(connections, polylines)):
            if not poly or len(poly) < 2:
                broken.append({
                    'connection_id': conn.get('id', f'conn_{i}'),
                    'from_id': conn.get('from_id'),
                    'to_id': conn.get('to_id'),
                    'issue': 'Empty or invalid polyline'
                })
        
        return broken
    
    def _check_splits_merges(
        self,
        graph: nx.DiGraph,
        elements: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Check for missing split/merge elements.
        
        If graph has nodes with out_degree > 1 but no Line_Split element,
        or in_degree > 1 but no Line_Merge element, flag as missing.
        
        Args:
            graph: Network graph
            elements: Detected elements
            
        Returns:
            Tuple of (missing_splits, missing_merges)
        """
        missing_splits = []
        missing_merges = []
        
        # Check for splits (multiple outputs)
        for node_id in graph.nodes():
            out_degree = graph.out_degree(node_id)
            
            if out_degree > 1:
                # Check if there's a Line_Split element nearby
                has_split = any(
                    el.get('type') == 'Line_Split' and
                    el.get('id') == f"Split_at_{node_id}"
                    for el in elements
                )
                
                if not has_split:
                    missing_splits.append({
                        'element_id': node_id,
                        'out_degree': out_degree,
                        'issue': f'Node has {out_degree} outputs but no Line_Split element'
                    })
        
        # Check for merges (multiple inputs)
        for node_id in graph.nodes():
            in_degree = graph.in_degree(node_id)
            
            if in_degree > 1:
                # Check if there's a Line_Merge element nearby
                has_merge = any(
                    el.get('type') == 'Line_Merge' and
                    el.get('id') == f"Merge_at_{node_id}"
                    for el in elements
                )
                
                if not has_merge:
                    missing_merges.append({
                        'element_id': node_id,
                        'in_degree': in_degree,
                        'issue': f'Node has {in_degree} inputs but no Line_Merge element'
                    })
        
        return missing_splits, missing_merges
    
    def _verify_connections_with_cv_lines(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        pipeline_lines: List[Dict[str, Any]],
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Verify LLM-detected connections against CV-extracted pipeline lines.
        
        CRITICAL: This is Pattern 4 from the Audit - CV-based connection verification.
        For each LLM-detected connection, checks if there's an actual physical line
        path in the CV-extracted pipeline lines that connects the two elements.
        
        Args:
            elements: Detected elements with bboxes
            connections: LLM-detected connections
            pipeline_lines: CV-extracted pipeline lines from line_extractor
                          Format: [{'polyline': [[x, y], ...], 'start_element': id, 'end_element': id}, ...]
            image_width: Image width
            image_height: Image height
            
        Returns:
            List of unverified connections (connections without physical CV line paths)
        """
        unverified = []
        
        # Create element map for quick lookup
        element_map = {el.get('id'): el for el in elements}
        
        # Create CV line map: (from_id, to_id) -> pipeline_line
        cv_line_map = {}
        for cv_line in pipeline_lines:
            start_id = cv_line.get('start_element')
            end_id = cv_line.get('end_element')
            if start_id and end_id:
                cv_line_map[(start_id, end_id)] = cv_line
        
        # Verify each LLM connection
        for conn in connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            
            if not from_id or not to_id:
                continue
            
            # Check if both elements exist
            from_el = element_map.get(from_id)
            to_el = element_map.get(to_id)
            
            if not from_el or not to_el:
                # Element missing - this is a different issue, skip CV verification
                continue
            
            # Check if CV line exists for this connection
            if (from_id, to_id) in cv_line_map:
                # CV line found - connection is verified
                continue
            
            # Check reverse direction (CV might have detected line in opposite direction)
            if (to_id, from_id) in cv_line_map:
                # CV line found in reverse - connection is verified (but might be bidirectional)
                continue
            
            # Check if CV line connects to/from bboxes (more flexible matching)
            # Sometimes CV detects line segments that are close to element bboxes
            bbox_match = self._check_bbox_connection(
                from_el, to_el, pipeline_lines, image_width, image_height
            )
            
            if not bbox_match:
                # No CV line found - connection is unverified (likely hallucination)
                unverified.append({
                    'connection_id': conn.get('id', f"{from_id}->{to_id}"),
                    'from_id': from_id,
                    'to_id': to_id,
                    'from_type': from_el.get('type', 'Unknown'),
                    'to_type': to_el.get('type', 'Unknown'),
                    'issue': 'No physical CV line path found for this connection (likely LLM hallucination)',
                    'confidence': conn.get('confidence', 0.0)
                })
        
        return unverified
    
    def _check_bbox_connection(
        self,
        from_el: Dict[str, Any],
        to_el: Dict[str, Any],
        pipeline_lines: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        tolerance: float = 0.05  # 5% tolerance for bbox matching
    ) -> bool:
        """
        Check if any CV pipeline line connects the bounding boxes of two elements.
        
        This is a more flexible check that doesn't require exact element ID matching.
        It checks if a CV line segment starts near from_el bbox and ends near to_el bbox.
        
        Args:
            from_el: Source element
            to_el: Target element
            pipeline_lines: CV-extracted pipeline lines
            image_width: Image width
            image_height: Image height
            tolerance: Normalized tolerance for bbox matching (default 0.05 = 5%)
            
        Returns:
            True if a connecting line is found, False otherwise
        """
        from_bbox = from_el.get('bbox')
        to_bbox = to_el.get('bbox')
        
        if not from_bbox or not to_bbox:
            return False
        
        # Get bbox centers
        from_center_x = float(from_bbox.get('x', 0)) + float(from_bbox.get('width', 0)) / 2
        from_center_y = float(from_bbox.get('y', 0)) + float(from_bbox.get('height', 0)) / 2
        to_center_x = float(to_bbox.get('x', 0)) + float(to_bbox.get('width', 0)) / 2
        to_center_y = float(to_bbox.get('y', 0)) + float(to_bbox.get('height', 0)) / 2
        
        # Check each CV line
        for cv_line in pipeline_lines:
            polyline = cv_line.get('polyline', [])
            if len(polyline) < 2:
                continue
            
            # Get start and end points of CV line
            start_point = polyline[0]  # Normalized [x, y]
            end_point = polyline[-1]   # Normalized [x, y]
            
            # Calculate distances from element centers to line endpoints
            dist_from_start = np.sqrt(
                (start_point[0] - from_center_x)**2 + 
                (start_point[1] - from_center_y)**2
            )
            dist_from_end = np.sqrt(
                (end_point[0] - to_center_x)**2 + 
                (end_point[1] - to_center_y)**2
            )
            
            # Check if line connects from_el to to_el (within tolerance)
            if dist_from_start <= tolerance and dist_from_end <= tolerance:
                return True
            
            # Check reverse direction (line might go from to_el to from_el)
            dist_reverse_start = np.sqrt(
                (start_point[0] - to_center_x)**2 + 
                (start_point[1] - to_center_y)**2
            )
            dist_reverse_end = np.sqrt(
                (end_point[0] - from_center_x)**2 + 
                (end_point[1] - from_center_y)**2
            )
            
            if dist_reverse_start <= tolerance and dist_reverse_end <= tolerance:
                return True
        
        return False

