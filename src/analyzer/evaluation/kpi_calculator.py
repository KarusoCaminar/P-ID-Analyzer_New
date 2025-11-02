"""
KPI Calculator - Comprehensive KPI calculation with precision, recall, F1-score.

Calculates detailed KPIs including:
- Element precision/recall/F1
- Connection precision/recall/F1
- Type accuracy
- Structural KPIs
- Confidence metrics
"""

import logging
import networkx as nx
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class KPICalculator:
    """
    Comprehensive KPI calculator for P&ID analysis.
    """
    
    def __init__(self):
        """Initialize KPI calculator."""
        pass
    
    def calculate_comprehensive_kpis(
        self,
        analysis_data: Dict[str, Any],
        truth_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive KPIs.
        
        Args:
            analysis_data: Analysis result data
            truth_data: Optional ground truth for comparison
            
        Returns:
            Comprehensive KPI dictionary
        """
        kpis = {}
        
        # Basic structural KPIs (always available)
        kpis.update(self._calculate_structural_kpis(analysis_data))
        
        # Confidence metrics
        kpis.update(self._calculate_confidence_metrics(analysis_data))
        
        # Quality metrics (if truth data available)
        if truth_data:
            kpis.update(self._calculate_quality_metrics(analysis_data, truth_data))
        
        return kpis
    
    def _calculate_structural_kpis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate structural KPIs from graph structure."""
        elements = data.get('elements', [])
        connections = data.get('connections', [])
        
        kpis = {
            'total_elements': len(elements),
            'total_connections': len(connections),
            'unique_element_types': len(set(el.get('type', 'unknown') for el in elements)),
            'connected_elements': 0,
            'isolated_elements': 0,
            'graph_density': 0.0,
            'num_cycles': 0,
            'max_centrality': 0.0
        }
        
        if not elements:
            return kpis
        
        # Build graph
        try:
            G = nx.DiGraph()
            node_ids = [el.get('id') for el in elements if el.get('id')]
            G.add_nodes_from(node_ids)
            
            for conn in connections:
                from_id = conn.get('from_id')
                to_id = conn.get('to_id')
                if from_id and to_id and from_id in G and to_id in G:
                    G.add_edge(from_id, to_id)
            
            # Calculate metrics
            connected_node_ids = set()
            for conn in connections:
                from_id = conn.get('from_id')
                to_id = conn.get('to_id')
                if from_id:
                    connected_node_ids.add(from_id)
                if to_id:
                    connected_node_ids.add(to_id)
            
            kpis['connected_elements'] = len(connected_node_ids)
            kpis['isolated_elements'] = len(elements) - len(connected_node_ids)
            
            # Graph density
            if G.number_of_nodes() > 1:
                kpis['graph_density'] = round(nx.density(G), 4)
            
            # Cycles
            try:
                cycles = list(nx.simple_cycles(G))
                kpis['num_cycles'] = len(cycles)
            except:
                pass
            
            # Centrality
            if G.number_of_nodes() > 2:
                try:
                    centrality = nx.betweenness_centrality(G, normalized=True)
                    if centrality:
                        kpis['max_centrality'] = round(max(centrality.values()), 4)
                except:
                    pass
        
        except Exception as e:
            logger.error(f"Error calculating structural KPIs: {e}", exc_info=True)
        
        return kpis
    
    def _calculate_confidence_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics."""
        elements = data.get('elements', [])
        connections = data.get('connections', [])
        
        element_confidences = [el.get('confidence', 0.0) for el in elements if 'confidence' in el]
        connection_confidences = [conn.get('confidence', 0.0) for conn in connections if 'confidence' in conn]
        
        metrics = {}
        
        if element_confidences:
            metrics['avg_element_confidence'] = round(sum(element_confidences) / len(element_confidences), 3)
            metrics['min_element_confidence'] = round(min(element_confidences), 3)
            metrics['max_element_confidence'] = round(max(element_confidences), 3)
        
        if connection_confidences:
            metrics['avg_connection_confidence'] = round(sum(connection_confidences) / len(connection_confidences), 3)
            metrics['min_connection_confidence'] = round(min(connection_confidences), 3)
            metrics['max_connection_confidence'] = round(max(connection_confidences), 3)
        
        return metrics
    
    def _calculate_iou(self, box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        if not box_a or not box_b:
            return 0.0
        
        # Normalize bbox format (support both dict and object)
        def get_bbox_coords(bbox):
            if isinstance(bbox, dict):
                return bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
            elif hasattr(bbox, 'x'):
                return bbox.x, bbox.y, bbox.width, bbox.height
            return 0, 0, 0, 0
        
        x_a, y_a, w_a, h_a = get_bbox_coords(box_a)
        x_b, y_b, w_b, h_b = get_bbox_coords(box_b)
        
        # Calculate intersection
        x_inter = max(x_a, x_b)
        y_inter = max(y_a, y_b)
        w_inter = min(x_a + w_a, x_b + w_b) - x_inter
        h_inter = min(y_a + h_a, y_b + h_b) - y_inter
        
        if w_inter <= 0 or h_inter <= 0:
            return 0.0
        
        inter_area = w_inter * h_inter
        box_a_area = w_a * h_a
        box_b_area = w_b * h_b
        union_area = box_a_area + box_b_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _calculate_quality_metrics(
        self,
        analysis_data: Dict[str, Any],
        truth_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate quality metrics by comparing with truth data using IoU matching."""
        # Helper to normalize bbox format
        def normalize_bbox(bbox):
            """Convert bbox to dict format."""
            if bbox is None:
                return None
            if isinstance(bbox, dict):
                return bbox
            if hasattr(bbox, 'model_dump'):
                return bbox.model_dump()
            if hasattr(bbox, 'dict'):
                return bbox.dict()
            if hasattr(bbox, 'x'):
                return {
                    'x': float(bbox.x),
                    'y': float(bbox.y),
                    'width': float(bbox.width),
                    'height': float(bbox.height)
                }
            return None
        
        # Helper to get element field
        def get_el_field(el, field, default=None):
            if isinstance(el, dict):
                return el.get(field, default)
            return getattr(el, field, default)
        
        # Extract elements with bboxes (normalize bboxes)
        analysis_elements = []
        for el in analysis_data.get('elements', []):
            bbox = get_el_field(el, 'bbox')
            if bbox:
                normalized_el = el.copy() if isinstance(el, dict) else {k: getattr(el, k) for k in dir(el) if not k.startswith('_')}
                normalized_el['bbox'] = normalize_bbox(bbox)
                analysis_elements.append(normalized_el)
        
        truth_elements = []
        for el in truth_data.get('elements', []):
            bbox = get_el_field(el, 'bbox')
            if bbox:
                normalized_el = el.copy() if isinstance(el, dict) else {k: getattr(el, k) for k in dir(el) if not k.startswith('_')}
                normalized_el['bbox'] = normalize_bbox(bbox)
                truth_elements.append(normalized_el)
        
        logger.info(f"IoU matching: {len(analysis_elements)} analysis elements, {len(truth_elements)} truth elements")
        
        # Match elements using IoU (Intersection over Union) instead of IDs
        matches = {}  # truth_id -> analysis_element
        unmatched_analysis = list(analysis_elements)
        iou_scores = []  # Track IoU scores for debugging
        
        # Adaptive IoU threshold: Start high for precision, reduce if needed
        iou_threshold = 0.3  # Higher initial threshold for better precision
        matches_found = False
        
        # First pass: Try with high threshold (0.3) for precision
        for truth_el in truth_elements:
            best_match = None
            max_iou = 0.0
            best_match_idx = -1
            
            truth_bbox = truth_el.get('bbox') if isinstance(truth_el, dict) else get_el_field(truth_el, 'bbox')
            if not truth_bbox:
                continue
            
            for idx, analysis_el in enumerate(unmatched_analysis):
                analysis_bbox = analysis_el.get('bbox') if isinstance(analysis_el, dict) else get_el_field(analysis_el, 'bbox')
                if not analysis_bbox:
                    continue
                    
                iou = self._calculate_iou(truth_bbox, analysis_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_match = analysis_el
                    best_match_idx = idx
            
            if max_iou >= iou_threshold and best_match:
                matches_found = True
                truth_id = get_el_field(truth_el, 'id', f"truth_{len(matches)}")
                matches[truth_id] = best_match
                iou_scores.append(max_iou)
                if best_match_idx >= 0:
                    unmatched_analysis.pop(best_match_idx)
                logger.info(f"Matched truth element {truth_id} with IoU {max_iou:.3f} (threshold: {iou_threshold:.2f})")
            elif max_iou > 0.0:
                # Log close misses for debugging
                truth_id = get_el_field(truth_el, 'id', f"truth_{len(matches)}")
                logger.debug(f"Close miss: truth element {truth_id} best IoU {max_iou:.3f} < threshold {iou_threshold:.2f}")
        
        # If no matches found with high threshold, retry with lower thresholds
        if not matches_found and len(truth_elements) > 0:
            logger.warning(f"No matches found with IoU >= {iou_threshold}, retrying with lower thresholds")
            
            # Reset for retry
            unmatched_analysis = list(analysis_elements)
            matches = {}
            iou_scores = []
            
            # Try with medium threshold (0.2)
            iou_threshold = 0.2
            for truth_el in truth_elements:
                best_match = None
                max_iou = 0.0
                best_match_idx = -1
                
                truth_bbox = truth_el.get('bbox') if isinstance(truth_el, dict) else get_el_field(truth_el, 'bbox')
                if not truth_bbox:
                    continue
                
                for idx, analysis_el in enumerate(unmatched_analysis):
                    analysis_bbox = analysis_el.get('bbox') if isinstance(analysis_el, dict) else get_el_field(analysis_el, 'bbox')
                    if not analysis_bbox:
                        continue
                        
                    iou = self._calculate_iou(truth_bbox, analysis_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        best_match = analysis_el
                        best_match_idx = idx
                
                if max_iou >= iou_threshold and best_match:
                    matches_found = True
                    truth_id = get_el_field(truth_el, 'id', f"truth_{len(matches)}")
                    matches[truth_id] = best_match
                    iou_scores.append(max_iou)
                    if best_match_idx >= 0:
                        unmatched_analysis.pop(best_match_idx)
                    logger.info(f"Matched truth element {truth_id} with IoU {max_iou:.3f} (threshold: {iou_threshold:.2f})")
            
            # Final fallback: very low threshold (0.1) only if still no matches
            if not matches_found and len(truth_elements) > 0:
                logger.warning(f"No matches found with IoU >= {iou_threshold}, using fallback threshold 0.1")
                
                # Reset for final retry
                unmatched_analysis = list(analysis_elements)
                matches = {}
                iou_scores = []
                iou_threshold = 0.1
                
                for truth_el in truth_elements:
                    best_match = None
                    max_iou = 0.0
                    best_match_idx = -1
                    
                    truth_bbox = truth_el.get('bbox') if isinstance(truth_el, dict) else get_el_field(truth_el, 'bbox')
                    if not truth_bbox:
                        continue
                    
                    for idx, analysis_el in enumerate(unmatched_analysis):
                        analysis_bbox = analysis_el.get('bbox') if isinstance(analysis_el, dict) else get_el_field(analysis_el, 'bbox')
                        if not analysis_bbox:
                            continue
                            
                        iou = self._calculate_iou(truth_bbox, analysis_bbox)
                        if iou > max_iou:
                            max_iou = iou
                            best_match = analysis_el
                            best_match_idx = idx
                    
                    if max_iou >= iou_threshold and best_match:
                        matches_found = True
                        truth_id = get_el_field(truth_el, 'id', f"truth_{len(matches)}")
                        matches[truth_id] = best_match
                        iou_scores.append(max_iou)
                        if best_match_idx >= 0:
                            unmatched_analysis.pop(best_match_idx)
                        logger.info(f"Matched truth element {truth_id} with IoU {max_iou:.3f} (threshold: {iou_threshold:.2f})")
        
        if iou_scores:
            avg_iou = sum(iou_scores) / len(iou_scores)
            logger.info(f"Matched {len(matches)} elements with average IoU {avg_iou:.3f}")
        else:
            logger.warning(f"No elements matched! Check IoU threshold ({iou_threshold}) and bbox formats.")
        
        # Calculate metrics
        correctly_found = set(matches.keys())
        missed = {el.get('id', f"truth_{i}") for i, el in enumerate(truth_elements) if el.get('id', f"truth_{i}") not in correctly_found}
        hallucinated = [el for el in unmatched_analysis]
        
        analysis_ids = set()  # For precision calculation
        for el in analysis_elements:
            if el.get('id'):
                analysis_ids.add(el.get('id'))
        
        element_precision = len(correctly_found) / len(analysis_elements) if analysis_elements else 0.0
        element_recall = len(correctly_found) / len(truth_elements) if truth_elements else 0.0
        element_f1 = 2 * (element_precision * element_recall) / (element_precision + element_recall) if (element_precision + element_recall) > 0 else 0.0
        
        # Type accuracy (only for matched elements)
        type_correct = 0
        type_total = 0
        for truth_id, analysis_el in matches.items():
            truth_el = next((el for el in truth_elements if el.get('id', f"truth_{list(matches.keys()).index(truth_id)}") == truth_id), None)
            if truth_el:
                analysis_type = analysis_el.get('type', '').lower()
                truth_type = truth_el.get('type', '').lower()
            type_total += 1
            if analysis_type == truth_type:
                type_correct += 1
        
        type_accuracy = type_correct / type_total if type_total > 0 else 0.0
        
        # Connection matching (using matched element IDs)
        # Build ID mapping: truth_id -> analysis_id (using matched elements)
        id_mapping = {}
        for truth_id, analysis_el in matches.items():
            # Find truth element by its actual ID or by index
            truth_el = None
            for te in truth_elements:
                if te.get('id') == truth_id:
                    truth_el = te
                    break
            
            if truth_el and analysis_el.get('id'):
                truth_element_id = truth_el.get('id')
                analysis_element_id = analysis_el.get('id')
                id_mapping[truth_element_id] = analysis_element_id
        
        # Map truth connections to analysis connection space
        analysis_connections = {(c.get('from_id'), c.get('to_id')) for c in analysis_data.get('connections', [])
                               if c.get('from_id') and c.get('to_id')}
        truth_connections = {(c.get('from_id'), c.get('to_id')) for c in truth_data.get('connections', [])
                            if c.get('from_id') and c.get('to_id')}
        
        # Map truth connections to analysis space using id_mapping
        mapped_truth_connections = set()
        for from_id, to_id in truth_connections:
            mapped_from = id_mapping.get(from_id, from_id)
            mapped_to = id_mapping.get(to_id, to_id)
            mapped_truth_connections.add((mapped_from, mapped_to))
        
        # Connection metrics
        correctly_connected = analysis_connections & mapped_truth_connections
        missed_connections = mapped_truth_connections - analysis_connections
        hallucinated_connections = analysis_connections - mapped_truth_connections
        
        connection_precision = len(correctly_connected) / len(analysis_connections) if analysis_connections else 0.0
        connection_recall = len(correctly_connected) / len(truth_connections) if truth_connections else 0.0
        connection_f1 = 2 * (connection_precision * connection_recall) / (connection_precision + connection_recall) if (connection_precision + connection_recall) > 0 else 0.0
        
        # Overall quality score
        quality_score = (
            element_f1 * 0.4 +  # Element F1 (40%)
            connection_f1 * 0.3 +  # Connection F1 (30%)
            type_accuracy * 0.3  # Type accuracy (30%)
        ) * 100
        
        return {
            'element_precision': round(element_precision, 3),
            'element_recall': round(element_recall, 3),
            'element_f1': round(element_f1, 3),
            'type_accuracy': round(type_accuracy, 3),
            'connection_precision': round(connection_precision, 3),
            'connection_recall': round(connection_recall, 3),
            'connection_f1': round(connection_f1, 3),
            'quality_score': round(quality_score, 2),
            'missed_elements': len(missed),
            'hallucinated_elements': len(hallucinated),
            'missed_connections': len(missed_connections),
            'hallucinated_connections': len(hallucinated_connections)
        }


