"""
Fusion Engine - Intelligent merging of swarm and monolith analysis results.

CRITICAL: This is the heart of the self-correction loop.

Must carefully determine which data is correct before overwriting.

"""

import logging
import copy
import uuid
from typing import Dict, Any, List, Optional, Tuple

from src.utils.graph_utils import calculate_iou, dedupe_connections, propagate_connection_confidence
from src.utils.type_utils import is_valid_bbox

logger = logging.getLogger(__name__)


def normalize_type_for_comparison(type_str: str) -> str:
    """Normalize type string for comparison (case-insensitive, handle synonyms)."""
    if not type_str:
        return ""
    normalized = type_str.lower().strip()
    
    # Common synonyms
    synonyms = {
        "valve": "valve",
        "pump": "pump",
        "sensor": "flow_sensor",
        "flow sensor": "flow_sensor",
        "volume flow sensor": "flow_sensor",
        "machine": "mixer",
        "mixer": "mixer",
        "source": "source",
        "sink": "sink"
    }
    
    return synonyms.get(normalized, normalized)


class FusionEngine:
    """
    Intelligent fusion engine for combining swarm and monolith analysis results.
    
    CRITICAL STRATEGY:
    - Always evaluate quality BEFORE overwriting
    - Only overwrite if clearly better
    - Combine good data from both sources
    - Never destroy good data
    """
    
    def __init__(self, iou_match_threshold: float = 0.5):
        """
        Initialize fusion engine.
        
        Args:
            iou_match_threshold: IoU threshold for matching elements (default: 0.5)
        """
        self.iou_match_threshold = iou_match_threshold
        self.fusion_stats = {}
    
    def fuse(
        self,
        swarm_result: Dict[str, Any],
        monolith_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Basic fusion logic.
        
        Intelligently fuse swarm and monolith analysis results.
        
        CRITICAL: Only overwrites if monolith is clearly better.
        
        DEPRECATED: Use fuse_with_legend_authority() for confidence-based fusion.
        """
        return self.fuse_with_legend_authority(swarm_result, monolith_result, {}, 0.0, False)
    
    def fuse_with_legend_authority(
        self,
        swarm_result: Dict[str, Any],
        monolith_result: Optional[Dict[str, Any]],
        symbol_map: Dict[str, str] = {},
        legend_confidence: float = 0.0,
        is_plausible: bool = False,
        line_map: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Confidence-based fusion with legend authority.
        
        Implements three fusion cases:
        1. "Authority" (Legend has priority): Element in legend → use legend confidence
        2. "Dual Detection": Both Swarm and Monolith found it → combine high confidence
        3. "Potential Hallucination": Low evidence → keep with low confidence
        
        Args:
            swarm_result: Swarm analysis result
            monolith_result: Monolith analysis result
            symbol_map: Legend symbol map (symbol_key -> type)
            legend_confidence: Legend confidence score (0.0-1.0)
            is_plausible: Whether legend passed plausibility check
            line_map: Legend line map (line_key -> line_info) for connection authority
            
        Returns:
            Fused result with confidence-based scoring
        """
        logger.info("Starting CONFIDENCE-BASED fusion with legend authority...")
        logger.info(f"Legend confidence: {legend_confidence:.2f}, is_plausible: {is_plausible}, symbols: {len(symbol_map)}")
        
        # Reset stats
        self.fusion_stats = {
            'authority': 0,  # Case 1: Legend authority
            'dual_detection': 0,  # Case 2: Both found it
            'potential_hallucination': 0,  # Case 3: Low evidence
            'rejected': 0
        }
        
        # --- KORREKTUR 1: Falsche Default-Confidence ---
        # Eine fehlende Konfidenz ist ein FEHLER und muss bestraft werden (z.B. 0.1),
        # nicht belohnt (z.B. 0.7).
        PENALTY_CONFIDENCE = 0.1
        
        # Handle case when swarm is empty - use monolith as fallback
        if not swarm_result or not swarm_result.get("elements"):
            if monolith_result and monolith_result.get("elements"):
                logger.warning("Fusion: Swarm data is empty. Using monolith data as fallback.")
                fallback_elements = copy.deepcopy(monolith_result.get('elements', []))
                fallback_connections = copy.deepcopy(monolith_result.get('connections', []))
                for el in fallback_elements:
                    if 'confidence' not in el:
                        el['confidence'] = PENALTY_CONFIDENCE  # KORRIGIERT
                for conn in fallback_connections:
                    if 'confidence' not in conn:
                        conn['confidence'] = PENALTY_CONFIDENCE # KORRIGIERT
                logger.info(f"Fallback fusion: {len(fallback_elements)} elements, {len(fallback_connections)} connections")
                return {"elements": fallback_elements, "connections": fallback_connections}
            else:
                logger.warning("Fusion: Both swarm and monolith data are empty.")
                return {"elements": [], "connections": []}
        
        # Collect all elements from both sources
        swarm_elements = copy.deepcopy(swarm_result.get('elements', []))
        monolith_elements = copy.deepcopy(monolith_result.get('elements', [])) if monolith_result else []
        swarm_connections = copy.deepcopy(swarm_result.get('connections', []))
        monolith_connections = copy.deepcopy(monolith_result.get('connections', [])) if monolith_result else []
        
        # Ensure all elements have confidence scores (KORREKTUR 1)
        for el in swarm_elements:
            if 'confidence' not in el:
                el['confidence'] = PENALTY_CONFIDENCE
        for el in monolith_elements:
            if 'confidence' not in el:
                el['confidence'] = PENALTY_CONFIDENCE
        for conn in swarm_connections:
            if 'confidence' not in conn:
                conn['confidence'] = PENALTY_CONFIDENCE
        for conn in monolith_connections:
            if 'confidence' not in conn:
                conn['confidence'] = PENALTY_CONFIDENCE
        
        # Build element maps for matching
        final_elements: List[Dict[str, Any]] = []
        processed_element_ids = set()
        
        # Get legend types (for Case 1: Authority)
        legend_types = set(symbol_map.values()) if symbol_map else set()
        
        # Process all elements with confidence-based fusion
        all_elements = swarm_elements + monolith_elements
        
        for el in all_elements:
            el_id = el.get('id')
            if not el_id or el_id in processed_element_ids:
                continue
            
            el_type = el.get('type', '')
            
            # Find matching elements (same ID or high IoU)
            matching_elements = []
            for other_el in all_elements:
                if other_el.get('id') == el_id:
                    matching_elements.append(other_el)
                elif el.get('bbox') and other_el.get('bbox') and is_valid_bbox(el['bbox']) and is_valid_bbox(other_el['bbox']):
                    iou = calculate_iou(el['bbox'], other_el['bbox'])
                    if iou > self.iou_match_threshold:
                        matching_elements.append(other_el)
            
            if not matching_elements:
                continue
            
            # Determine fusion case
            in_legend = el_type in legend_types
            # Prüfe 'source' Attribut, das von den Analyzern gesetzt werden sollte
            swarm_found = any('swarm' in str(e.get('source', '')).lower() for e in matching_elements)
            monolith_found = any('monolith' in str(e.get('source', '')).lower() for e in matching_elements)
            
            # Fallback, falls 'source' fehlt (z.B. bei alten Daten)
            if not swarm_found and not monolith_found:
                 swarm_found = any(e.get('id') == el_id for e in swarm_elements)
                 monolith_found = any(e.get('id') == el_id for e in monolith_elements)
            
            # Get best confidence from matching elements
            best_confidence = max(e.get('confidence', PENALTY_CONFIDENCE) for e in matching_elements)
            best_element = max(matching_elements, key=lambda e: e.get('confidence', PENALTY_CONFIDENCE))
            
            # CASE 1: "Authority" (Legend has priority)
            if in_legend and is_plausible and legend_confidence > 0.8:
                final_confidence = max(best_confidence, legend_confidence)
                logger.debug(f"FUSION-AUTHORITY: Element '{el_id}' ({el_type}) in legend → confidence={final_confidence:.2f}")
                self.fusion_stats['authority'] += 1
                
                fused_el = copy.deepcopy(best_element)
                fused_el['confidence'] = final_confidence
                fused_el['fusion_case'] = 'authority'
                final_elements.append(fused_el)
                processed_element_ids.add(el_id)
            
            # CASE 2: "Dual Detection" (Both Swarm and Monolith found it)
            elif swarm_found and monolith_found:
                swarm_conf = max(e.get('confidence', PENALTY_CONFIDENCE) for e in matching_elements if 'swarm' in str(e.get('source', '')).lower())
                monolith_conf = max(e.get('confidence', PENALTY_CONFIDENCE) for e in matching_elements if 'monolith' in str(e.get('source', '')).lower())
                # Fallback, falls 'source' fehlt
                if swarm_conf == PENALTY_CONFIDENCE and monolith_conf == PENALTY_CONFIDENCE:
                    swarm_conf = max(e.get('confidence', PENALTY_CONFIDENCE) for e in swarm_elements if e.get('id') == el_id)
                    monolith_conf = max(e.get('confidence', PENALTY_CONFIDENCE) for e in monolith_elements if e.get('id') == el_id)
                # Combined confidence: C = C_s + (1 - C_s) * C_m
                final_confidence = swarm_conf + (1 - swarm_conf) * monolith_conf
                logger.debug(f"FUSION-DUAL: Element '{el_id}' ({el_type}) found by both → confidence={final_confidence:.2f}")
                self.fusion_stats['dual_detection'] += 1
                
                fused_el = copy.deepcopy(best_element)
                fused_el['confidence'] = final_confidence
                fused_el['fusion_case'] = 'dual_detection'
                final_elements.append(fused_el)
                processed_element_ids.add(el_id)
            
            # CASE 3: "Potential Hallucination" (Low evidence)
            else:
                final_confidence = best_confidence
                logger.debug(f"FUSION-POTENTIAL-HALLUCINATION: Element '{el_id}' ({el_type}) low evidence → confidence={final_confidence:.2f}")
                self.fusion_stats['potential_hallucination'] += 1
                
                fused_el = copy.deepcopy(best_element)
                fused_el['confidence'] = final_confidence
                fused_el['fusion_case'] = 'potential_hallucination'
                final_elements.append(fused_el)
                processed_element_ids.add(el_id)
        
        
        
        # --- KORREKTUR 2: Intelligente Fusion für Verbindungen ---
        # Die alte Logik (nur Mergen) wurde entfernt.
        # Wir implementieren die gleiche 3-Fall-Logik für Verbindungen.
        
        final_connections: List[Dict[str, Any]] = []
        processed_connection_keys = set()
        all_connections = swarm_connections + monolith_connections
        
        # Get legend line types (for Case 1: Authority)
        # Check if line_map contains connection type information
        line_map_types = set()
        if line_map and isinstance(line_map, dict):
            # Extract connection types from line_map
            for line_key, line_info in line_map.items():
                if isinstance(line_info, dict):
                    line_type = line_info.get('type', '')
                    if line_type:
                        line_map_types.add(line_type)
                elif isinstance(line_info, str):
                    line_map_types.add(line_info)
        
        for conn in all_connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            conn_key = (from_id, to_id)
            
            if not from_id or not to_id or conn_key in processed_connection_keys:
                continue
            
            # Finde passende Verbindungen
            matching_connections = [
                c for c in all_connections 
                if c.get('from_id') == from_id and c.get('to_id') == to_id
            ]
            
            if not matching_connections:
                continue
                
            conn_type = conn.get('type', 'process')  # Default type
            
            # Bestimme Fusions-Fall
            # Check if connection type is in legend line_map
            in_legend = conn_type in line_map_types if line_map_types else False
            
            # Also check if both endpoints are in legend (indirect authority)
            if not in_legend and symbol_map and legend_types:
                # Build element ID to type mapping
                element_id_to_type = {}
                for el in all_elements:
                    el_id = el.get('id')
                    el_type = el.get('type', '')
                    if el_id:
                        element_id_to_type[el_id] = el_type
                
                # Check if both endpoints' types are in legend
                from_type = element_id_to_type.get(from_id, '')
                to_type = element_id_to_type.get(to_id, '')
                from_in_legend = from_type in legend_types
                to_in_legend = to_type in legend_types
                
                # Connection is in legend if both endpoints are in legend
                in_legend = from_in_legend and to_in_legend
            
            swarm_found = any('swarm' in str(c.get('source', '')).lower() for c in matching_connections)
            monolith_found = any('monolith' in str(c.get('source', '')).lower() for c in matching_connections)
            
            # Fallback, falls 'source' fehlt
            if not swarm_found and not monolith_found:
                 swarm_found = any(c.get('from_id') == from_id and c.get('to_id') == to_id for c in swarm_connections)
                 monolith_found = any(c.get('from_id') == from_id and c.get('to_id') == to_id for c in monolith_connections)
            
            best_confidence = max(c.get('confidence', PENALTY_CONFIDENCE) for c in matching_connections)
            best_connection = max(matching_connections, key=lambda c: c.get('confidence', PENALTY_CONFIDENCE))
            
            # CASE 1: "Authority" (Legend has priority)
            if in_legend and is_plausible and legend_confidence > 0.8:
                final_confidence = max(best_confidence, legend_confidence)
                logger.debug(f"FUSION-AUTHORITY: Connection '{from_id} -> {to_id}' in legend → confidence={final_confidence:.2f}")
                self.fusion_stats['authority'] += 1
                
                fused_conn = copy.deepcopy(best_connection)
                fused_conn['confidence'] = final_confidence
                fused_conn['fusion_case'] = 'authority'
                final_connections.append(fused_conn)
                processed_connection_keys.add(conn_key)
            
            # CASE 2: "Dual Detection" (Both Swarm and Monolith found it)
            elif swarm_found and monolith_found:
                swarm_conf = max(c.get('confidence', PENALTY_CONFIDENCE) for c in matching_connections if 'swarm' in str(c.get('source', '')).lower())
                monolith_conf = max(c.get('confidence', PENALTY_CONFIDENCE) for c in matching_connections if 'monolith' in str(c.get('source', '')).lower())
                # Fallback, falls 'source' fehlt
                if swarm_conf == PENALTY_CONFIDENCE and monolith_conf == PENALTY_CONFIDENCE:
                    swarm_conf = max(c.get('confidence', PENALTY_CONFIDENCE) for c in swarm_connections if c.get('from_id') == from_id and c.get('to_id') == to_id)
                    monolith_conf = max(c.get('confidence', PENALTY_CONFIDENCE) for c in monolith_connections if c.get('from_id') == from_id and c.get('to_id') == to_id)
                final_confidence = swarm_conf + (1 - swarm_conf) * monolith_conf
                logger.debug(f"FUSION-DUAL: Connection '{from_id} -> {to_id}' found by both → confidence={final_confidence:.2f}")
                self.fusion_stats['dual_detection'] += 1
                
                fused_conn = copy.deepcopy(best_connection)
                fused_conn['confidence'] = final_confidence
                fused_conn['fusion_case'] = 'dual_detection'
                final_connections.append(fused_conn)
                processed_connection_keys.add(conn_key)
            
            # CASE 3: "Potential Hallucination" (Low evidence)
            else:
                final_confidence = best_confidence
                logger.debug(f"FUSION-POTENTIAL-HALLUCINATION: Connection '{from_id} -> {to_id}' low evidence → confidence={final_confidence:.2f}")
                self.fusion_stats['potential_hallucination'] += 1
                
                fused_conn = copy.deepcopy(best_connection)
                fused_conn['confidence'] = final_confidence
                fused_conn['fusion_case'] = 'potential_hallucination'
                final_connections.append(fused_conn)
                processed_connection_keys.add(conn_key)
        
        # Deduplicate connections
        final_connections = dedupe_connections(final_connections)
        
        # Log fusion statistics
        logger.info(f"Fusion complete: {len(final_elements)} elements, {len(final_connections)} connections")
        logger.info(f"Fusion stats: Authority={self.fusion_stats['authority']}, Dual={self.fusion_stats['dual_detection']}, "
                   f"Potential Hallucination={self.fusion_stats['potential_hallucination']}, Rejected={self.fusion_stats['rejected']}")
        
        return {
            "elements": final_elements,
            "connections": final_connections
        }
