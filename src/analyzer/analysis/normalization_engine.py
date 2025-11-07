"""
Normalization Engine - Validation and normalization logic for analysis results.

Extracted from pipeline_coordinator.py Phase 4 to improve maintainability and testability.

Handles:
- Confidence filtering (elements and connections)
- BBox validation and normalization
- Element/Connection type validation
- Data structure normalization
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.analyzer.models.elements import Element, Connection, BBox
from src.utils.type_utils import is_valid_bbox

logger = logging.getLogger(__name__)


class NormalizationEngine:
    """
    Validates and normalizes analysis results.
    
    Extracted from pipeline_coordinator.py Phase 4 to improve maintainability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Normalization Engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logic_parameters = config.get('logic_parameters', {})
    
    def process(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> Tuple[List[Element], List[Connection], Dict[str, Any]]:
        """
        Process and normalize elements and connections.
        
        Performs:
        1. Confidence filtering (remove low-confidence elements/connections)
        2. BBox validation (reject invalid bounding boxes)
        3. Data structure normalization (convert to Pydantic models)
        4. Connection filtering (remove connections to filtered elements)
        
        Args:
            elements: List of element dictionaries
            connections: List of connection dictionaries
            
        Returns:
            Tuple of (normalized_elements, normalized_connections, stats_dict)
            where stats_dict contains:
            - removed_elements: Number of removed elements
            - removed_connections: Number of removed connections
            - invalid_bboxes: Number of invalid bboxes rejected
        """
        logger.info("=== Starting normalization and validation ===")
        
        stats = {
            'removed_elements': 0,
            'removed_connections': 0,
            'invalid_bboxes': 0
        }
        
        # Step 1: Confidence filtering
        confidence_threshold = self.logic_parameters.get('confidence_threshold', 0.7)
        logger.info(f"Normalization: Filtering elements with confidence < {confidence_threshold}")
        
        filtered_elements, filtered_elements_ids, removed_count = self._filter_by_confidence(
            elements, confidence_threshold
        )
        stats['removed_elements'] = removed_count
        
        if removed_count > 0:
            logger.info(f"Normalization: Removed {removed_count} low-confidence elements (confidence < {confidence_threshold})")
        
        # Step 2: Filter connections (only keep connections between filtered elements)
        filtered_connections, removed_conn_count = self._filter_connections(
            connections, filtered_elements_ids, confidence_threshold
        )
        stats['removed_connections'] = removed_conn_count
        
        if removed_conn_count > 0:
            logger.info(f"Normalization: Removed {removed_conn_count} connections (missing elements or confidence < {confidence_threshold})")
        
        # Step 3: BBox validation and normalization
        normalized_elements, invalid_bbox_count = self._validate_and_normalize_elements(
            filtered_elements
        )
        stats['invalid_bboxes'] = invalid_bbox_count
        
        if invalid_bbox_count > 0:
            logger.info(f"Normalization: Rejected {invalid_bbox_count} elements with invalid bboxes")
        
        # Step 4: Normalize connections
        normalized_connections = self._normalize_connections(filtered_connections)
        
        logger.info(f"Normalization complete: {len(normalized_elements)} elements, {len(normalized_connections)} connections")
        
        return normalized_elements, normalized_connections, stats
    
    def _filter_by_confidence(
        self,
        elements: List[Dict[str, Any]],
        confidence_threshold: float
    ) -> Tuple[List[Dict[str, Any]], set, int]:
        """
        Filter elements by confidence threshold.
        
        Args:
            elements: List of element dictionaries
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (filtered_elements, filtered_element_ids, removed_count)
        """
        filtered_elements = []
        filtered_elements_ids = set()
        removed_count = 0
        
        for el in elements:
            el_dict = el if isinstance(el, dict) else el.model_dump() if hasattr(el, 'model_dump') else el.__dict__ if hasattr(el, '__dict__') else {}
            confidence = el_dict.get('confidence', 0.5)
            
            if confidence >= confidence_threshold:
                filtered_elements.append(el)
                el_id = el_dict.get('id')
                if el_id:
                    filtered_elements_ids.add(el_id)
            else:
                removed_count += 1
                logger.debug(f"Removed low-confidence element: {el_dict.get('id', 'unknown')} (confidence: {confidence:.2f} < {confidence_threshold})")
        
        return filtered_elements, filtered_elements_ids, removed_count
    
    def _filter_connections(
        self,
        connections: List[Dict[str, Any]],
        valid_element_ids: set,
        confidence_threshold: float
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Filter connections by element existence and confidence.
        
        Args:
            connections: List of connection dictionaries
            valid_element_ids: Set of valid element IDs
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (filtered_connections, removed_count)
        """
        filtered_connections = []
        removed_count = 0
        
        for conn in connections:
            conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ if hasattr(conn, '__dict__') else {}
            from_id = conn_dict.get('from_id')
            to_id = conn_dict.get('to_id')
            conn_confidence = conn_dict.get('confidence', 0.5)
            
            # Keep connection if both elements exist AND connection has sufficient confidence
            if from_id in valid_element_ids and to_id in valid_element_ids and conn_confidence >= confidence_threshold:
                filtered_connections.append(conn)
            else:
                removed_count += 1
                logger.debug(f"Removed connection: {from_id} -> {to_id} (missing elements or low confidence: {conn_confidence:.2f})")
        
        return filtered_connections, removed_count
    
    def _validate_and_normalize_elements(
        self,
        elements: List[Dict[str, Any]]
    ) -> Tuple[List[Element], int]:
        """
        Validate and normalize elements to Pydantic models.
        
        Performs:
        - BBox validation (reject invalid bboxes)
        - BBox normalization (ensure bounds 0.0-1.0)
        - Element model creation
        
        Args:
            elements: List of element dictionaries
            
        Returns:
            Tuple of (normalized_elements, invalid_bbox_count)
        """
        normalized_elements = []
        invalid_bbox_count = 0
        
        for el in elements:
            # Handle different element formats
            if isinstance(el, Element):
                # Validate existing Element
                if el.bbox.width > 0 and el.bbox.height > 0:
                    normalized_elements.append(el)
                else:
                    logger.warning(f"Skipping element {el.id}: Invalid bbox (width={el.bbox.width}, height={el.bbox.height})")
                    invalid_bbox_count += 1
                    continue
            elif isinstance(el, dict):
                try:
                    # CRITICAL: VALIDATE instead of REPAIR - reject invalid bboxes
                    if 'bbox' not in el or not isinstance(el['bbox'], dict):
                        logger.warning(f"Skipping element {el.get('id', 'unknown')}: Missing or invalid bbox structure")
                        invalid_bbox_count += 1
                        continue
                    
                    bbox = el['bbox']
                    element_id = el.get('id', 'unknown')
                    
                    # STRICT VALIDATION: BBox must have valid, positive dimensions
                    if not is_valid_bbox(bbox) or bbox.get('width', 0) <= 0 or bbox.get('height', 0) <= 0:
                        logger.warning(
                            f"REJECTING element {element_id}: Invalid bbox dimensions "
                            f"(width={bbox.get('width')}, height={bbox.get('height')}). "
                            f"Element is likely a hallucination."
                        )
                        invalid_bbox_count += 1
                        continue  # Element is rejected, not repaired
                    
                    # BBox must be within bounds (0.0 - 1.0)
                    # CRITICAL FIX: Convert to float before arithmetic operations
                    bbox['x'] = max(0.0, min(1.0, float(bbox.get('x', 0.0))))
                    bbox['y'] = max(0.0, min(1.0, float(bbox.get('y', 0.0))))
                    bbox['width'] = max(0.001, min(1.0 - float(bbox.get('x', 0.0)), float(bbox.get('width', 0.001))))
                    bbox['height'] = max(0.001, min(1.0 - float(bbox.get('y', 0.0)), float(bbox.get('height', 0.001))))
                    
                    # Create BBox Pydantic model
                    bbox_model = BBox(
                        x=bbox['x'],
                        y=bbox['y'],
                        width=bbox['width'],
                        height=bbox['height']
                    )
                    
                    # Create Element Pydantic model
                    element_model = Element(
                        id=el.get('id', 'unknown'),
                        type=el.get('type', 'Unknown'),
                        label=el.get('label', ''),
                        bbox=bbox_model,
                        confidence=el.get('confidence', 0.5),
                        ports=el.get('ports', [])
                    )
                    
                    normalized_elements.append(element_model)
                    
                except Exception as e:
                    logger.warning(f"Error normalizing element {el.get('id', 'unknown')}: {e}. Skipping.")
                    invalid_bbox_count += 1
                    continue
            else:
                logger.warning(f"Unknown element format: {type(el)}. Skipping.")
                invalid_bbox_count += 1
                continue
        
        return normalized_elements, invalid_bbox_count
    
    def _normalize_connections(
        self,
        connections: List[Dict[str, Any]]
    ) -> List[Connection]:
        """
        Normalize connections to Pydantic models.
        
        Args:
            connections: List of connection dictionaries
            
        Returns:
            List of Connection Pydantic models
        """
        normalized_connections = []
        
        for conn in connections:
            # Handle different connection formats
            if isinstance(conn, Connection):
                normalized_connections.append(conn)
            elif isinstance(conn, dict):
                try:
                    connection_model = Connection(
                        from_id=conn.get('from_id', ''),
                        to_id=conn.get('to_id', ''),
                        confidence=conn.get('confidence', 0.5),
                        polyline=conn.get('polyline', [])
                    )
                    normalized_connections.append(connection_model)
                except Exception as e:
                    logger.warning(f"Error normalizing connection {conn.get('from_id', 'unknown')} -> {conn.get('to_id', 'unknown')}: {e}. Skipping.")
                    continue
            else:
                logger.warning(f"Unknown connection format: {type(conn)}. Skipping.")
                continue
        
        return normalized_connections

