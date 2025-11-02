"""
Advanced Graph Theory Utilities for P&ID Analysis.

Provides comprehensive graph theory operations using NetworkX:
- Split/Merge Detection with positions
- Pipeline Flow Analysis
- Graph Metrics (Centrality, Density, Cycles)
- Network Graph Visualization
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class GraphTheoryAnalyzer:
    """
    Advanced graph theory analyzer for P&ID diagrams.
    
    Uses NetworkX for professional graph analysis.
    """
    
    def __init__(self, elements: List[Dict[str, Any]], connections: List[Dict[str, Any]]):
        """
        Initialize graph theory analyzer.
        
        Args:
            elements: List of elements with id, bbox
            connections: List of connections with from_id, to_id
        """
        self.elements = elements
        self.connections = connections
        self.id_to_element = {el.get('id'): el for el in elements if el.get('id')}
        
        # Build graph
        self.graph = self._build_graph()
        self.positions = self._calculate_positions()
    
    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph from elements and connections."""
        G = nx.DiGraph()
        
        # Add nodes
        for el in self.elements:
            el_id = el.get('id')
            if el_id:
                G.add_node(el_id, **{
                    'type': el.get('type'),
                    'label': el.get('label'),
                    'bbox': el.get('bbox'),
                    'confidence': el.get('confidence', 0.0)
                })
        
        # Add edges
        for conn in self.connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            if from_id and to_id and from_id in G and to_id in G:
                G.add_edge(from_id, to_id, **{
                    'id': conn.get('id'),
                    'confidence': conn.get('confidence', 0.0),
                    'kind': conn.get('kind', 'process'),
                    'polyline': conn.get('polyline', [])
                })
        
        return G
    
    def _calculate_positions(self) -> Dict[str, Tuple[float, float]]:
        """Calculate positions for all elements from BBox coordinates."""
        positions = {}
        
        for el in self.elements:
            el_id = el.get('id')
            bbox = el.get('bbox')
            if el_id and bbox:
                x = bbox.get('x', 0.0) + bbox.get('width', 0.0) / 2
                y = bbox.get('y', 0.0) + bbox.get('height', 0.0) / 2
                positions[el_id] = (x, y)
        
        return positions
    
    def detect_splits_and_merges(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect split and merge points with positions using graph theory.
        
        Returns:
            Dictionary with 'splits' and 'merges' lists
        """
        splits = []
        merges = []
        
        # Calculate in-degree and out-degree
        for node_id in self.graph.nodes():
            in_degree = self.graph.in_degree(node_id)
            out_degree = self.graph.out_degree(node_id)
            
            node_pos = self.positions.get(node_id, (0.0, 0.0))
            
            # Merge point: multiple inputs
            if in_degree > 1:
                predecessor_ids = list(self.graph.predecessors(node_id))
                
                # Calculate merge position (average of predecessors and node)
                connected_positions = [node_pos]
                for pred_id in predecessor_ids:
                    pred_pos = self.positions.get(pred_id)
                    if pred_pos:
                        connected_positions.append(pred_pos)
                
                if len(connected_positions) > 1:
                    avg_x = np.mean([p[0] for p in connected_positions])
                    avg_y = np.mean([p[1] for p in connected_positions])
                    
                    merges.append({
                        'id': f"Merge_at_{node_id}",
                        'node_id': node_id,
                        'position': {'x': avg_x, 'y': avg_y},
                        'connected_from': predecessor_ids,
                        'connected_to': [node_id],
                        'degree': in_degree
                    })
            
            # Split point: multiple outputs
            if out_degree > 1:
                successor_ids = list(self.graph.successors(node_id))
                
                # Calculate split position (average of node and successors)
                connected_positions = [node_pos]
                for succ_id in successor_ids:
                    succ_pos = self.positions.get(succ_id)
                    if succ_pos:
                        connected_positions.append(succ_pos)
                
                if len(connected_positions) > 1:
                    avg_x = np.mean([p[0] for p in connected_positions])
                    avg_y = np.mean([p[1] for p in connected_positions])
                    
                    splits.append({
                        'id': f"Split_at_{node_id}",
                        'node_id': node_id,
                        'position': {'x': avg_x, 'y': avg_y},
                        'connected_from': [node_id],
                        'connected_to': successor_ids,
                        'degree': out_degree
                    })
        
        return {'splits': splits, 'merges': merges}
    
    def analyze_pipeline_flows(self) -> List[Dict[str, Any]]:
        """
        Analyze pipeline flows (complete flow paths) using graph theory.
        
        Returns:
            List of flow dictionaries with path, positions, and components
        """
        flows = []
        processed_edges = set()
        
        # Find all flow paths
        for edge in self.graph.edges():
            edge_key = (edge[0], edge[1])
            if edge_key in processed_edges:
                continue
            
            # Build flow path starting from this edge
            flow_path = [edge[0], edge[1]]
            current_id = edge[1]
            processed_edges.add(edge_key)
            
            # Extend path forward
            while True:
                successors = list(self.graph.successors(current_id))
                if not successors:
                    break
                
                # Find next unprocessed edge
                next_edge = next((
                    (current_id, succ_id) for succ_id in successors
                    if (current_id, succ_id) not in processed_edges
                ), None)
                
                if next_edge:
                    flow_path.append(next_edge[1])
                    processed_edges.add(next_edge)
                    current_id = next_edge[1]
                else:
                    break
            
            # Get positions and components along flow
            if len(flow_path) > 1:
                flow_positions = []
                flow_components = []
                
                for node_id in flow_path:
                    pos = self.positions.get(node_id)
                    node_data = self.graph.nodes[node_id]
                    
                    if pos:
                        el = self.id_to_element.get(node_id, {})
                        bbox = el.get('bbox', {})
                        
                        flow_positions.append({
                            'x': pos[0],
                            'y': pos[1],
                            'bbox': bbox
                        })
                        
                        flow_components.append({
                            'id': node_id,
                            'type': node_data.get('type', 'Unknown'),
                            'label': node_data.get('label', node_id)
                        })
                
                flows.append({
                    'id': f"Flow_{len(flows)+1}",
                    'path': flow_path,
                    'positions': flow_positions,
                    'components': flow_components,
                    'length': len(flow_path),
                    'start_id': flow_path[0],
                    'end_id': flow_path[-1]
                })
        
        return flows
    
    def calculate_graph_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive graph metrics using NetworkX.
        
        Returns:
            Dictionary with graph metrics (density, centrality, cycles, etc.)
        """
        if self.graph.number_of_nodes() == 0:
            return {
                'density': 0.0,
                'num_cycles': 0,
                'max_centrality': 0.0,
                'avg_centrality': 0.0,
                'num_components': 0
            }
        
        metrics = {}
        
        # Graph density
        if self.graph.number_of_nodes() > 1:
            metrics['density'] = round(nx.density(self.graph), 4)
        else:
            metrics['density'] = 0.0
        
        # Cycle detection
        try:
            cycles = list(nx.simple_cycles(self.graph))
            metrics['num_cycles'] = len(cycles)
            metrics['cycles'] = cycles
        except Exception as e:
            logger.warning(f"Cycle detection failed: {e}")
            metrics['num_cycles'] = 0
            metrics['cycles'] = []
        
        # Centrality (betweenness)
        if self.graph.number_of_nodes() > 2:
            try:
                centrality = nx.betweenness_centrality(self.graph, normalized=True)
                if centrality:
                    metrics['max_centrality'] = round(max(centrality.values()), 4)
                    metrics['avg_centrality'] = round(np.mean(list(centrality.values())), 4)
                    metrics['centrality'] = {k: round(v, 4) for k, v in centrality.items()}
            except Exception as e:
                logger.warning(f"Centrality calculation failed: {e}")
                metrics['max_centrality'] = 0.0
                metrics['avg_centrality'] = 0.0
        
        # Connected components (undirected)
        G_undirected = self.graph.to_undirected()
        components = list(nx.connected_components(G_undirected))
        metrics['num_components'] = len(components)
        metrics['component_sizes'] = [len(comp) for comp in components]
        
        # Degree statistics
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        metrics['avg_in_degree'] = round(np.mean(list(in_degrees.values())), 2) if in_degrees else 0.0
        metrics['avg_out_degree'] = round(np.mean(list(out_degrees.values())), 2) if out_degrees else 0.0
        metrics['max_in_degree'] = max(in_degrees.values()) if in_degrees else 0
        metrics['max_out_degree'] = max(out_degrees.values()) if out_degrees else 0
        
        return metrics
    
    def get_network_graph_representation(self) -> Dict[str, Any]:
        """
        Get complete network graph representation with all information.
        
        Returns:
            Dictionary with nodes, edges, positions, splits, merges, flows, metrics
        """
        splits_merges = self.detect_splits_and_merges()
        flows = self.analyze_pipeline_flows()
        metrics = self.calculate_graph_metrics()
        
        # Build node representation
        nodes = []
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            pos = self.positions.get(node_id, (0.0, 0.0))
            el = self.id_to_element.get(node_id, {})
            
            nodes.append({
                'id': node_id,
                'type': node_data.get('type'),
                'label': node_data.get('label'),
                'position': {'x': pos[0], 'y': pos[1]},
                'bbox': el.get('bbox'),
                'confidence': node_data.get('confidence', 0.0),
                'in_degree': self.graph.in_degree(node_id),
                'out_degree': self.graph.out_degree(node_id)
            })
        
        # Build edge representation
        edges = []
        for from_id, to_id, edge_data in self.graph.edges(data=True):
            edges.append({
                'from_id': from_id,
                'to_id': to_id,
                'id': edge_data.get('id'),
                'confidence': edge_data.get('confidence', 0.0),
                'kind': edge_data.get('kind', 'process'),
                'polyline': edge_data.get('polyline', [])
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'positions': {k: {'x': v[0], 'y': v[1]} for k, v in self.positions.items()},
            'splits': splits_merges['splits'],
            'merges': splits_merges['merges'],
            'flows': flows,
            'metrics': metrics,
            'metadata': {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'num_splits': len(splits_merges['splits']),
                'num_merges': len(splits_merges['merges']),
                'num_flows': len(flows)
            }
        }


