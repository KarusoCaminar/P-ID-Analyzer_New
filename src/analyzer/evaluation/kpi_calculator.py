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
    
    def __init__(self, confidence_calibration_offset: float = 0.0):
        """
        Initialize KPI calculator.
        
        Args:
            confidence_calibration_offset: Offset to apply to confidence scores for calibration
        """
        self.confidence_calibration_offset = confidence_calibration_offset
    
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
            except (nx.NetworkXError, ValueError, RuntimeError) as e:
                logger.warning(f"Cycle detection failed: {e}")
                pass
            
            # Centrality
            if G.number_of_nodes() > 2:
                try:
                    centrality = nx.betweenness_centrality(G, normalized=True)
                    if centrality:
                        kpis['max_centrality'] = round(max(centrality.values()), 4)
                except (nx.NetworkXError, ValueError, ZeroDivisionError) as e:
                    logger.warning(f"Centrality calculation failed: {e}")
                    pass
        
        except Exception as e:
            logger.error(f"Error calculating structural KPIs: {e}", exc_info=True)
        
        return kpis
    
    def _calculate_confidence_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics with calibration."""
        elements = data.get('elements', [])
        connections = data.get('connections', [])
        
        # Apply calibration offset to confidence scores
        element_confidences = [
            min(1.0, max(0.0, el.get('confidence', 0.0) + self.confidence_calibration_offset))
            for el in elements if 'confidence' in el
        ]
        connection_confidences = [
            min(1.0, max(0.0, conn.get('confidence', 0.0) + self.confidence_calibration_offset))
            for conn in connections if 'confidence' in conn
        ]
        
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
    
    def _calculate_graph_edit_distance(
        self,
        analysis_connections: Set[Tuple[str, str]],
        truth_connections: Set[Tuple[str, str]],
        id_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Calculate Graph-Edit-Distance (GED) between analysis and truth graphs.
        
        Graph-Edit-Distance measures the structural similarity between two graphs by
        counting the minimum number of operations (add/delete edges) needed to transform
        one graph into another.
        
        This is a normalized approximation that uses edge differences:
        - Add operations: edges in truth but not in analysis (missed connections)
        - Delete operations: edges in analysis but not in truth (hallucinated connections)
        - Total GED = |missed| + |hallucinated|
        - Normalized GED = GED / max(|truth|, |analysis|, 1)
        
        Args:
            analysis_connections: Set of (from_id, to_id) tuples from analysis
            truth_connections: Set of (from_id, to_id) tuples from truth (already mapped)
            id_mapping: Mapping from truth IDs to analysis IDs (for documentation, not used in calculation)
            
        Returns:
            Dictionary with GED metrics:
            - raw_distance: Raw GED (number of operations)
            - normalized_distance: Normalized GED (0.0 = identical, 1.0 = completely different)
            - similarity_score: Similarity score (1.0 = identical, 0.0 = completely different)
            - add_operations: Number of edges to add (missed connections)
            - delete_operations: Number of edges to delete (hallucinated connections)
            - match_ratio: Ratio of matching edges to total edges
        """
        # Calculate edge differences
        missed_edges = truth_connections - analysis_connections
        hallucinated_edges = analysis_connections - truth_connections
        matched_edges = analysis_connections & truth_connections
        
        # Raw GED: minimum operations to transform analysis graph to truth graph
        add_ops = len(missed_edges)      # Add missed connections
        delete_ops = len(hallucinated_edges)  # Delete hallucinated connections
        raw_ged = add_ops + delete_ops
        
        # Normalized GED: normalize by the size of the larger graph
        max_graph_size = max(len(truth_connections), len(analysis_connections), 1)
        normalized_ged = raw_ged / max_graph_size if max_graph_size > 0 else 1.0
        
        # Match ratio: percentage of edges that match
        total_edges = len(truth_connections) + len(hallucinated_edges)  # Truth + extra
        match_ratio = len(matched_edges) / total_edges if total_edges > 0 else 0.0
        
        # Structural similarity score (1.0 = identical, 0.0 = completely different)
        # Based on normalized GED, inverted so higher is better
        similarity_score = 1.0 - normalized_ged
        
        logger.debug(f"Graph-Edit-Distance: raw={raw_ged}, normalized={normalized_ged:.3f}, "
                    f"similarity={similarity_score:.3f}, match_ratio={match_ratio:.3f}")
        
        return {
            'raw_distance': raw_ged,
            'normalized_distance': round(normalized_ged, 4),
            'similarity_score': round(similarity_score, 4),
            'add_operations': add_ops,
            'delete_operations': delete_ops,
            'match_ratio': round(match_ratio, 4),
            'matched_edges': len(matched_edges),
            'missed_edges': len(missed_edges),
            'hallucinated_edges': len(hallucinated_edges)
        }
    
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
        truth_elements_with_bbox = []
        truth_elements_without_bbox = []
        
        for el in truth_data.get('elements', []):
            bbox = get_el_field(el, 'bbox')
            normalized_el = el.copy() if isinstance(el, dict) else {k: getattr(el, k) for k in dir(el) if not k.startswith('_')}
            
            if bbox:
                normalized_el['bbox'] = normalize_bbox(bbox)
                truth_elements_with_bbox.append(normalized_el)
                truth_elements.append(normalized_el)
            else:
                # Store elements without bbox for ID-based matching
                truth_elements_without_bbox.append(normalized_el)
                truth_elements.append(normalized_el)
        
        logger.info(f"Matching: {len(analysis_elements)} analysis elements, {len(truth_elements)} truth elements "
                   f"({len(truth_elements_with_bbox)} with bbox, {len(truth_elements_without_bbox)} without bbox)")
        
        matches = {}  # truth_id -> analysis_element
        unmatched_analysis = list(analysis_elements)
        iou_scores = []  # Track IoU scores for debugging
        id_match_scores = []  # Track ID-based match scores
        
        # PHASE 1: IoU-based matching for elements with bboxes
        matches_found = False
        if truth_elements_with_bbox:
            iou_threshold = 0.3
            for truth_el in truth_elements_with_bbox:
                best_match = None
                max_iou = 0.0
                best_match_idx = -1
                
                truth_bbox = truth_el.get('bbox')
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
                    logger.debug(f"Matched truth element {truth_id} with IoU {max_iou:.3f}")
        
        # PHASE 2: ID-based matching for elements without bboxes (fuzzy matching)
        if truth_elements_without_bbox:
            logger.info(f"Using ID-based matching for {len(truth_elements_without_bbox)} truth elements without bboxes")
            
            def normalize_id(id_str: str) -> str:
                """Normalize ID for fuzzy matching (remove spaces, lowercase, etc.)"""
                if not id_str:
                    return ""
                return id_str.lower().replace(" ", "").replace("-", "").replace("_", "")
            
            def normalize_label(label_str: str) -> str:
                """Normalize label for fuzzy matching"""
                if not label_str:
                    return ""
                return label_str.lower().strip()
            
            for truth_el in truth_elements_without_bbox:
                truth_id = get_el_field(truth_el, 'id', '')
                truth_type = get_el_field(truth_el, 'type', '')
                truth_label = get_el_field(truth_el, 'label', '')
                
                best_match = None
                best_match_score = 0.0
                best_match_idx = -1
                
                for idx, analysis_el in enumerate(unmatched_analysis):
                    analysis_id = get_el_field(analysis_el, 'id', '')
                    analysis_type = get_el_field(analysis_el, 'type', '')
                    analysis_label = get_el_field(analysis_el, 'label', '')
                    
                    match_score = 0.0
                    
                    # Priority 1: Exact ID match (highest priority)
                    if truth_id and analysis_id:
                        if normalize_id(truth_id) == normalize_id(analysis_id):
                            match_score = 1.0
                        # Priority 2: ID contains truth ID or vice versa
                        elif normalize_id(truth_id) in normalize_id(analysis_id) or normalize_id(analysis_id) in normalize_id(truth_id):
                            match_score = 0.8
                    
                    # Priority 3: Label match (high priority)
                    if match_score < 0.8 and truth_label and analysis_label:
                        norm_truth_label = normalize_label(truth_label)
                        norm_analysis_label = normalize_label(analysis_label)
                        if norm_truth_label == norm_analysis_label:
                            match_score = max(match_score, 0.9)
                        elif norm_truth_label in norm_analysis_label or norm_analysis_label in norm_truth_label:
                            match_score = max(match_score, 0.7)
                    
                    # Priority 4: Type match (medium priority)
                    if match_score < 0.7 and truth_type and analysis_type:
                        if truth_type.lower() == analysis_type.lower():
                            match_score = max(match_score, 0.5)
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match = analysis_el
                        best_match_idx = idx
                
                # Accept match if score >= 0.5 (ID or label match)
                if best_match_score >= 0.5 and best_match:
                    truth_id_key = truth_id or f"truth_{len(matches)}"
                    matches[truth_id_key] = best_match
                    id_match_scores.append(best_match_score)
                    if best_match_idx >= 0:
                        unmatched_analysis.pop(best_match_idx)
                    logger.info(f"Matched truth element {truth_id_key} (type: {truth_type}, label: {truth_label}) "
                              f"with analysis element (id: {get_el_field(best_match, 'id')}, type: {get_el_field(best_match, 'type')}) "
                              f"using ID-based matching (score: {best_match_score:.2f})")
                elif truth_id or truth_label:
                    logger.debug(f"No ID match found for truth element {truth_id or truth_label} (type: {truth_type})")
        
        # Summary of matching results
        total_matches = len(matches)
        if iou_scores or id_match_scores:
            if iou_scores:
                avg_iou = sum(iou_scores) / len(iou_scores)
                logger.info(f"Matched {len(iou_scores)} elements using IoU (avg IoU: {avg_iou:.3f})")
            if id_match_scores:
                avg_id_score = sum(id_match_scores) / len(id_match_scores)
                logger.info(f"Matched {len(id_match_scores)} elements using ID-based matching (avg score: {avg_id_score:.3f})")
            logger.info(f"Total matches: {total_matches} out of {len(truth_elements)} truth elements")
        else:
            if truth_elements:
                logger.warning(f"No elements matched! ({len(truth_elements)} truth elements, {len(analysis_elements)} analysis elements)")
            else:
                logger.info("No truth elements provided for matching")
        
        # Calculate metrics
        correctly_found = set(matches.keys())  # Truth IDs that were matched
        missed = {el.get('id', f"truth_{i}") for i, el in enumerate(truth_elements) if el.get('id', f"truth_{i}") not in correctly_found}
        hallucinated = [el for el in unmatched_analysis]
        
        # --- KORREKTUR: Für Precision brauchen wir die Anzahl der korrekt gematchten Analysis-Elemente ---
        # matches.values() enthält die Analysis-Elemente, die gematcht wurden
        correctly_found_analysis_ids = set()
        for analysis_el in matches.values():
            analysis_id = get_el_field(analysis_el, 'id', '')
            if analysis_id:
                correctly_found_analysis_ids.add(analysis_id)
        
        # Precision = Anzahl korrekt gefundener Analysis-Elemente / Anzahl aller Analysis-Elemente
        element_precision = len(correctly_found_analysis_ids) / len(analysis_elements) if analysis_elements else 0.0
        # Recall = Anzahl gefundener Truth-Elemente / Anzahl aller Truth-Elemente
        element_recall = len(correctly_found) / len(truth_elements) if truth_elements else 0.0
        element_f1 = 2 * (element_precision * element_recall) / (element_precision + element_recall) if (element_precision + element_recall) > 0 else 0.0
        # --- ENDE KORREKTUR ---
        
        # Type accuracy (only for matched elements) - with case-insensitive matching and synonym handling
        def normalize_type(type_str: str) -> str:
            """Normalize type for matching (case-insensitive, handle synonyms)"""
            if not type_str:
                return ""
            # Normalize to lowercase for comparison
            normalized = type_str.lower().strip()
            
            # Handle synonyms and common variations
            synonyms = {
                "valve": "Valve",
                "pump": "Pump",
                "sensor": "Volume Flow Sensor",  # If context suggests flow sensor
                "flow sensor": "Volume Flow Sensor",
                "flow meter": "Volume Flow Sensor",
                "machine": "Mixer",  # If context suggests mixer
                "mixer": "Mixer",
                "blender": "Mixer",
                "source": "Source",
                "sink": "Sink",
                "sample point": "Sample Point",
                "samplepoint": "Sample Point",
                "sampling point": "Sample Point"
            }
            
            # Return canonical form from synonyms
            return synonyms.get(normalized, type_str)  # Fallback to original if no synonym
        
        type_correct = 0
        type_total = 0
        for truth_id, analysis_el in matches.items():
            # Find truth element by ID
            truth_el = None
            for te in truth_elements:
                if te.get('id') == truth_id:
                    truth_el = te
                    break
            
            if truth_el:
                # Normalize both types for comparison
                analysis_type_normalized = normalize_type(analysis_el.get('type', ''))
                truth_type_normalized = normalize_type(truth_el.get('type', ''))
                
                # Case-insensitive comparison
                type_match = analysis_type_normalized.lower() == truth_type_normalized.lower()
                
                type_total += 1
                if type_match:
                    type_correct += 1
                else:
                    # Log mismatches for debugging
                    logger.debug(f"Type mismatch for element {truth_id}: "
                               f"truth='{truth_el.get('type')}' (normalized='{truth_type_normalized}') "
                               f"vs analysis='{analysis_el.get('type')}' (normalized='{analysis_type_normalized}')")
        
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
        
        # Graph-Edit-Distance (GED) - Measures structural similarity between graphs
        # GED is the minimum number of operations (add/delete/rename edges) needed to transform one graph into another
        # For efficiency, we use a normalized approximation based on edge differences
        graph_edit_distance = self._calculate_graph_edit_distance(
            analysis_connections, mapped_truth_connections, id_mapping
        )
        
        # Overall quality score
        quality_score = (
            element_f1 * 0.4 +  # Element F1 (40%)
            connection_f1 * 0.3 +  # Connection F1 (30%)
            type_accuracy * 0.3  # Type accuracy (30%)
        ) * 100
        
        # Graph-Edit-Distance (GED) - Measures structural similarity between graphs
        # GED is the minimum number of operations (add/delete/rename edges) needed to transform one graph into another
        # For efficiency, we use a normalized approximation based on edge differences
        graph_edit_distance = self._calculate_graph_edit_distance(
            analysis_connections, mapped_truth_connections, id_mapping
        )
        
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
            'hallucinated_connections': len(hallucinated_connections),
            'graph_edit_distance': graph_edit_distance,
            'normalized_graph_edit_distance': graph_edit_distance.get('normalized_distance', 1.0),
            'graph_similarity_score': graph_edit_distance.get('similarity_score', 0.0)
        }


