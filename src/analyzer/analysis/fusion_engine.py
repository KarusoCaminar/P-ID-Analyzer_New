"""
Fusion Engine - Intelligent merging of swarm and monolith analysis results.

Combines results from swarm (component-focused) and monolith (structure-focused)
analysis to produce a comprehensive, accurate representation of the P&ID diagram.
"""

import logging
import copy
from typing import Dict, Any, List, Optional

from src.utils.graph_utils import calculate_iou, dedupe_connections
from src.utils.type_utils import is_valid_bbox

logger = logging.getLogger(__name__)


class FusionEngine:
    """
    Fusion engine for combining swarm and monolith analysis results.
    
    Strategy:
    - Use swarm results as base (component completeness)
    - Enhance with monolith structure information
    - Correct element types and labels using monolith data
    - Merge connections intelligently
    """
    
    def __init__(self, iou_match_threshold: float = 0.1):
        """
        Initialize fusion engine.
        
        Args:
            iou_match_threshold: IoU threshold for matching elements
        """
        self.iou_match_threshold = iou_match_threshold
    
    def fuse(
        self,
        swarm_result: Dict[str, Any],
        monolith_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fuse swarm and monolith analysis results.
        
        Args:
            swarm_result: Results from swarm analysis
            monolith_result: Optional results from monolith analysis
            
        Returns:
            Fused results with elements and connections
        """
        logger.info("Starting intelligent master fusion (Swarm components + Monolith structure)...")
        
        # Handle case when swarm is empty - use monolith as fallback
        if not swarm_result or not swarm_result.get("elements"):
            if monolith_result and monolith_result.get("elements"):
                logger.warning("Fusion: Swarm data is empty. Using monolith data as fallback.")
                # Use monolith elements and connections as fallback
                fallback_elements = copy.deepcopy(monolith_result.get('elements', []))
                fallback_connections = copy.deepcopy(monolith_result.get('connections', []))
                # Mark elements with lower confidence since they're from fallback
                for el in fallback_elements:
                    if 'confidence' in el:
                        el['confidence'] = el['confidence'] * 0.8  # Reduce confidence for fallback
                    else:
                        el['confidence'] = 0.6  # Default fallback confidence
                logger.info(f"Fallback fusion: {len(fallback_elements)} elements, {len(fallback_connections)} connections from monolith")
                return {"elements": fallback_elements, "connections": fallback_connections}
            else:
                logger.warning("Fusion: Both swarm and monolith data are empty. Cannot fuse.")
                return {"elements": [], "connections": []}
        
        final_elements = copy.deepcopy(swarm_result.get('elements', []))
        swarm_connections = copy.deepcopy(swarm_result.get('connections', []))
        
        if not monolith_result or not monolith_result.get("elements"):
            logger.warning("Fusion: No monolith data available. Using swarm data only.")
            return {"elements": final_elements, "connections": swarm_connections}
        
        # Build element maps
        monolith_elements_map = {
            el['id']: el for el in monolith_result.get('elements', [])
        }
        swarm_elements_map = {
            el['id']: el for el in final_elements
        }
        
        # Correct swarm elements with monolith data
        fused_connections: List[Dict[str, Any]] = []
        
        for mono_el in monolith_result.get('elements', []):
            if not mono_el.get('bbox'):
                continue
            
            # Find best matching swarm element
            best_match_swarm_el = None
            max_iou = 0.0
            
            for swarm_el in final_elements:
                if not swarm_el.get('bbox'):
                    continue
                
                iou = calculate_iou(mono_el['bbox'], swarm_el['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_match_swarm_el = swarm_el
            
            # Update swarm element with monolith data if good match
            if best_match_swarm_el and max_iou > self.iou_match_threshold:
                logger.info(
                    f"FUSION-UPDATE: Correcting swarm element '{best_match_swarm_el.get('label')}' "
                    f"with monolith type '{mono_el.get('type')}'. (IoU: {max_iou:.2f})"
                )
                # Overwrite type and label with (presumably) better monolith data
                best_match_swarm_el['type'] = mono_el.get('type', best_match_swarm_el.get('type'))
                best_match_swarm_el['label'] = mono_el.get('label', best_match_swarm_el.get('label'))
                
                # Update confidence based on fusion match quality
                if 'confidence' not in best_match_swarm_el:
                    best_match_swarm_el['confidence'] = 0.5
                
                # Increase confidence if monolith confirms (high IoU = high confidence)
                current_conf = best_match_swarm_el.get('confidence', 0.5)
                mono_conf = mono_el.get('confidence', 0.7)
                fused_confidence = min(1.0, max(current_conf, mono_conf) * (1.0 + max_iou * 0.3))
                best_match_swarm_el['confidence'] = fused_confidence
        
        # Transplant monolith connections to corrected swarm elements
        for mono_conn in monolith_result.get('connections', []):
            mono_from_el = monolith_elements_map.get(mono_conn.get('from_id'))
            mono_to_el = monolith_elements_map.get(mono_conn.get('to_id'))
            
            if not mono_from_el or not mono_to_el:
                continue
            
            mono_from_bbox = mono_from_el.get('bbox')
            mono_to_bbox = mono_to_el.get('bbox')
            
            if not mono_from_bbox or not mono_to_bbox:
                continue
            
            # Find matching swarm elements
            valid_swarm_elements = [el for el in final_elements if el.get('bbox')]
            if not valid_swarm_elements:
                continue
            
            best_swarm_from = max(
                valid_swarm_elements,
                key=lambda el: calculate_iou(mono_from_bbox, el['bbox'])
            )
            best_swarm_to = max(
                valid_swarm_elements,
                key=lambda el: calculate_iou(mono_to_bbox, el['bbox'])
            )
            
            if best_swarm_from and best_swarm_to:
                import uuid
                
                # Calculate connection confidence based on element match quality
                from_iou = calculate_iou(mono_from_bbox, best_swarm_from.get('bbox', {}))
                to_iou = calculate_iou(mono_to_bbox, best_swarm_to.get('bbox', {}))
                avg_iou = (from_iou + to_iou) / 2.0
                
                conn_confidence = min(1.0, avg_iou * 1.2)  # Boost confidence for fused connections
                mono_conn_conf = mono_conn.get('confidence', 0.7)
                final_confidence = max(conn_confidence, mono_conn_conf)
                
                new_conn: Dict[str, Any] = {
                    "id": str(uuid.uuid4()),
                    "from_id": best_swarm_from['id'],
                    "to_id": best_swarm_to['id'],
                    "from_port_id": mono_conn.get("from_port_id"),
                    "to_port_id": mono_conn.get("to_port_id"),
                    "status": "fused_from_monolith",
                    "confidence": final_confidence
                }
                fused_connections.append(new_conn)
        
        # Deduplicate connections
        final_connections = dedupe_connections(swarm_connections + fused_connections)
        
        logger.info(
            f"Master fusion complete. Final: {len(final_elements)} elements / {len(final_connections)} connections."
        )
        
        return {"elements": final_elements, "connections": final_connections}

