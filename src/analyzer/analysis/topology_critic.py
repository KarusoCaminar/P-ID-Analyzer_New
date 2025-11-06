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
        polylines: Optional[List[List[List[float]]]] = None
    ) -> Dict[str, Any]:
        """
        Validate topology and find inconsistencies.
        
        Args:
            elements: Detected elements
            connections: Detected connections
            polylines: Optional list of polylines for each connection
            
        Returns:
            Dictionary with validation results:
            - disconnected_nodes: Nodes without connections
            - invalid_degrees: Nodes with impossible degrees
            - broken_paths: Paths that don't connect properly
            - missing_splits: Detected splits without Line_Split elements
            - missing_merges: Detected merges without Line_Merge elements
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
            
            # Check for missing splits/merges
            missing_splits, missing_merges = self._check_splits_merges(graph, elements)
            
            # Calculate validation score
            total_issues = (
                len(disconnected) +
                len(invalid_degrees) +
                len(broken_paths) +
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

