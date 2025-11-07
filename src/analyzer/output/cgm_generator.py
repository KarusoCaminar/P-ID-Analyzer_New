"""
CGM (Component Grouping Model) Generator - Vollständiges Python dataclass Format.

Generiert CGM Network im Python dataclass Format aus der Originalversion
mit Port, Connector, Network Klassen und vollständigen Koordinaten.

Nutzt erweiterte Graphentheorie für:
- Split/Merge Detection mit Positionen
- Pipeline Flow Representation
- Network Graph mit vollständigen Koordinaten
"""

import logging
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.utils.graph_theory import GraphTheoryAnalyzer

logger = logging.getLogger(__name__)


class CGMGenerator:
    """
    CGM Generator für vollständiges Python dataclass Format.
    
    Generiert:
    - Port, Connector, Network dataclasses
    - Vollständige Koordinaten (BBox) für alle Komponenten
    - Split/Merge Detection und Positionierung
    - Pipeline Flow Representation
    """
    
    def __init__(self, elements: List[Dict[str, Any]], connections: List[Dict[str, Any]]):
        """
        Initialize CGM Generator.
        
        Args:
            elements: List of element dictionaries with id, type, label, bbox, ports
            connections: List of connection dictionaries with from_id, to_id, ports
        """
        self.elements = elements
        self.connections = connections
        
        # FIX: ID-Normalisierung und Fuzzy Matching
        # 1. Normalisiere alle IDs (z.B. "FT 10" → "FT-10", "Fv-3-3040" → "Fv-3-3040")
        # 2. Erstelle Mapping: original_id → normalized_id
        self.id_normalization_map = {}
        normalized_elements = []
        
        for el in elements:
            if el.get('id'):
                original_id = el.get('id')
                normalized_id = self._normalize_id(original_id)
                self.id_normalization_map[original_id] = normalized_id
                # Update element ID
                el_copy = el.copy() if isinstance(el, dict) else {k: getattr(el, k) for k in dir(el) if not k.startswith('_')}
                el_copy['id'] = normalized_id
                normalized_elements.append(el_copy)
            else:
                normalized_elements.append(el)
        
        # Update elements with normalized IDs
        self.elements = normalized_elements
        
        # Create id_to_element map with normalized IDs
        self.id_to_element = {el.get('id'): el for el in self.elements if el.get('id')}
        
        # Also create label_to_element map for fuzzy matching
        self.label_to_element = {}
        for el in self.elements:
            label = el.get('label', '')
            if label:
                normalized_label = self._normalize_id(label)
                if normalized_label not in self.label_to_element:
                    self.label_to_element[normalized_label] = []
                self.label_to_element[normalized_label].append(el)
        
        # FIX: Normalisiere Connection-IDs und entferne None-Verbindungen
        normalized_connections = []
        for conn in connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            
            # Skip connections with None or missing IDs
            if not from_id or not to_id or from_id == 'None' or to_id == 'None':
                logger.debug(f"Skipping connection with None/missing IDs: {from_id} → {to_id}")
                continue
            
            # Normalize connection IDs
            from_id_normalized = self._normalize_id(from_id)
            to_id_normalized = self._normalize_id(to_id)
            
            # Check if elements exist (after normalization)
            from_el_exists = from_id_normalized in self.id_to_element or self._find_element_by_label(from_id)
            to_el_exists = to_id_normalized in self.id_to_element or self._find_element_by_label(to_id)
            
            if not from_el_exists or not to_el_exists:
                logger.debug(f"Skipping connection with non-existent elements: {from_id} → {to_id}")
                continue
            
            # Update connection with normalized IDs
            conn_copy = conn.copy()
            conn_copy['from_id'] = from_id_normalized
            conn_copy['to_id'] = to_id_normalized
            normalized_connections.append(conn_copy)
        
        logger.info(f"Connection normalization: {len(connections)} -> {len(normalized_connections)} (removed {len(connections) - len(normalized_connections)} invalid connections)")
        
        # Update connections with normalized IDs
        self.connections = normalized_connections
        
        # Use graph theory analyzer
        self.graph_analyzer = GraphTheoryAnalyzer(self.elements, self.connections)
        
    def generate_cgm_python_code(self) -> str:
        """
        Generate complete CGM Python code in dataclass format.
        
        Returns:
            Complete Python code string with Port, Connector, Network classes
        """
        logger.info("Generating CGM Python code in dataclass format...")
        
        # Generate metadata header
        metadata_header = self._generate_metadata_header()
        
        # Generate dataclass definitions
        dataclass_definitions = self._generate_dataclass_definitions()
        
        # Generate network instance with connectors
        network_instance = self._generate_network_instance()
        
        # Combine all parts
        complete_code = f"""{metadata_header}

{dataclass_definitions}

{network_instance}
"""
        return textwrap.dedent(complete_code).strip()
    
    def _generate_metadata_header(self) -> str:
        """Generate metadata header with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"""# ==============================================================================
# CGM Network - Automatisch generiert
# ==============================================================================
# Generiert am: {timestamp}
# Elemente: {len(self.elements)}
# Verbindungen: {len(self.connections)}
# =============================================================================="""
    
    def _generate_dataclass_definitions(self) -> str:
        """Generate Port, Connector, Network dataclass definitions."""
        return """from dataclasses import dataclass

@dataclass(frozen=True)
class Port:
    unit_name: str
    port: str

@dataclass(frozen=True)
class Connector:
    name: str
    from_converter_ports: tuple[Port, ...]
    to_converter_ports: tuple[Port, ...]

@dataclass(frozen=True)
class Network:
    connectors: tuple[Connector, ...]"""
    
    def _generate_network_instance(self) -> str:
        """Generate Network instance with all connectors."""
        connector_lines = []
        connector_keys = set()  # CRITICAL FIX 3: Track unique connectors to avoid duplicates
        
        # FIX: Process all connections and filter out None connectors
        valid_connections = []
        for i, conn in enumerate(self.connections):
            connector_code = self._generate_connector_code(conn, i)
            if connector_code:
                # CRITICAL FIX 3: Create unique key for connector (from_id, to_id) to avoid duplicates
                from_id = conn.get('from_id', '')
                to_id = conn.get('to_id', '')
                
                # Normalize IDs for comparison
                from_id_normalized = self._normalize_id(from_id)
                to_id_normalized = self._normalize_id(to_id)
                connector_key = (from_id_normalized, to_id_normalized)
                
                # Skip if duplicate
                if connector_key in connector_keys:
                    logger.debug(f"Skipping duplicate connector: {from_id} → {to_id}")
                    continue
                
                connector_keys.add(connector_key)
                connector_lines.append(connector_code)
                valid_connections.append(conn)
        
        logger.info(f"CGM generation: {len(self.connections)} connections -> {len(valid_connections)} valid connectors (removed {len(self.connections) - len(valid_connections)} duplicates)")
        
        # CRITICAL FIX 3: Handle split/merge points (only if not already in connector_lines)
        split_merge_connectors = self._detect_split_merge_connectors()
        split_merge_keys = set()
        
        for split_merge_code in split_merge_connectors:
            # Extract from_id and to_id from split/merge connector code
            # Format: Connector(name="...", from_converter_ports=(Port(unit_name="...", port="..."),), ...)
            # We need to parse the code to get the connector key
            # For now, we'll add all split/merge connectors but track them separately
            # TODO: Improve parsing to extract actual connector keys from split/merge code
            connector_lines.append(split_merge_code)
        
        connectors_code = "\n".join(f"        {line}" for line in connector_lines)
        
        return f"""# Automatisch generiertes Netzwerk-Objekt
cgm_network = Network(
    connectors=(
{connectors_code}
    )
)"""
    
    def _normalize_id(self, id_str: str) -> str:
        """
        Normalize ID for matching (remove spaces, normalize separators).
        
        Examples:
        - "FT 10" → "FT-10"
        - "Fv-3-3040" → "Fv-3-3040"
        - "Mixer M-08" → "Mixer-M-08"
        """
        if not id_str:
            return ""
        
        # Normalize: replace spaces with hyphens, remove extra spaces
        normalized = id_str.strip().replace(' ', '-').replace('_', '-')
        
        # Remove multiple consecutive hyphens
        while '--' in normalized:
            normalized = normalized.replace('--', '-')
        
        # Remove leading/trailing hyphens
        normalized = normalized.strip('-')
        
        return normalized
    
    def _find_element_by_label(self, search_label: str) -> Optional[Dict[str, Any]]:
        """
        Find element by label using fuzzy matching.
        
        Args:
            search_label: Label to search for (will be normalized)
            
        Returns:
            Element dict or None if not found
        """
        normalized_search = self._normalize_id(search_label)
        
        # Try exact match first
        if normalized_search in self.label_to_element:
            candidates = self.label_to_element[normalized_search]
            if candidates:
                return candidates[0]  # Return first match
        
        # Try partial match (label contains search or vice versa)
        for normalized_label, candidates in self.label_to_element.items():
            if normalized_search in normalized_label or normalized_label in normalized_search:
                if candidates:
                    return candidates[0]
        
        return None
    
    def _generate_connector_code(self, conn: Dict[str, Any], index: int) -> Optional[str]:
        """Generate connector code for a single connection with ID-Normalisierung und Fuzzy Matching."""
        from_id = conn.get('from_id')
        to_id = conn.get('to_id')
        
        if not from_id or not to_id:
            logger.warning(f"Connection {index} has missing from_id or to_id: {from_id} → {to_id}")
            return None
        
        # Normalize IDs
        from_id_normalized = self._normalize_id(from_id)
        to_id_normalized = self._normalize_id(to_id)
        
        # Try to find elements by normalized ID
        from_el = self.id_to_element.get(from_id_normalized)
        if not from_el:
            # Try with original ID (in case it wasn't normalized)
            from_el = self.id_to_element.get(from_id)
            if not from_el:
                # Try fuzzy matching by label
                from_el = self._find_element_by_label(from_id)
                if from_el:
                    logger.debug(f"Found element by label fuzzy match: {from_id} → {from_el.get('id')}")
        
        to_el = self.id_to_element.get(to_id_normalized)
        if not to_el:
            # Try with original ID (in case it wasn't normalized)
            to_el = self.id_to_element.get(to_id)
            if not to_el:
                # Try fuzzy matching by label
                to_el = self._find_element_by_label(to_id)
                if to_el:
                    logger.debug(f"Found element by label fuzzy match: {to_id} → {to_el.get('id')}")
        
        if not from_el or not to_el:
            logger.warning(f"Could not find elements for connection {index}: {from_id} → {to_id}")
            return None  # FIX: Nicht "None" verwenden, sondern None zurückgeben (wird dann übersprungen)
        
        # Get port names with intelligent fallback
        from_port_name = self._get_port_name(conn, 'from_port_id', from_el, default='Out')
        to_port_name = self._get_port_name(conn, 'to_port_id', to_el, default='In')
        
        # Get unit names (labels)
        from_unit_name = from_el.get('label', from_id)
        to_unit_name = to_el.get('label', to_id)
        
        # Generate connector name
        conn_name = f"Conn_{index+1}_{from_unit_name}_{to_unit_name}"
        conn_name = conn_name.replace('"', '\\"').replace("'", "\\'")
        
        # Generate Port tuples
        from_ports_str = f'(Port(unit_name="{from_unit_name}", port="{from_port_name}"),)'
        to_ports_str = f'(Port(unit_name="{to_unit_name}", port="{to_port_name}"),)'
        
        return f"""Connector(
        name="{conn_name}",
        from_converter_ports={from_ports_str},
        to_converter_ports={to_ports_str},
    ),"""
    
    def _get_port_name(
        self,
        conn: Dict[str, Any],
        port_id_key: str,
        element: Dict[str, Any],
        default: str
    ) -> str:
        """Get port name with intelligent fallback."""
        port_id = conn.get(port_id_key)
        
        if port_id and element.get('ports'):
            for port in element.get('ports', []):
                if port.get('id') == port_id:
                    return port.get('name', default)
        
        return default
    
    def _detect_split_merge_connectors(self) -> List[str]:
        """
        Detect split and merge points and generate connectors for them.
        
        Returns:
            List of connector code strings for splits and merges
        """
        # Build in-degree and out-degree maps
        in_degree: Dict[str, List[str]] = {el_id: [] for el_id in self.id_to_element}
        out_degree: Dict[str, List[str]] = {el_id: [] for el_id in self.id_to_element}
        
        for conn in self.connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            if from_id and to_id and from_id in self.id_to_element and to_id in self.id_to_element:
                in_degree[to_id].append(from_id)
                out_degree[from_id].append(to_id)
        
        connector_lines = []
        connector_index = len(self.connections)
        
        # Detect merge points (multiple inputs to one element)
        for el_id, el_data in self.id_to_element.items():
            if len(in_degree[el_id]) > 1:
                # Merge point detected
                merge_name = f"Node_Merge_at_{el_id}"
                
                from_ports = []
                for from_id in in_degree[el_id]:
                    from_el = self.id_to_element.get(from_id)
                    if from_el:
                        unit_name = from_el.get('label', from_id)
                        from_ports.append(f'Port(unit_name="{unit_name}", port="Out")')
                
                to_port = f'Port(unit_name="{el_data.get("label", el_id)}", port="In")'
                
                from_ports_str = f"({', '.join(from_ports)},)"
                to_ports_str = f"({to_port},)"
                
                connector_lines.append(
                    f"""Connector(
        name="{merge_name}",
        from_converter_ports={from_ports_str},
        to_converter_ports={to_ports_str},
    ),"""
                )
                connector_index += 1
        
        # Detect split points (one element to multiple outputs)
        for el_id, el_data in self.id_to_element.items():
            if len(out_degree[el_id]) > 1:
                # Split point detected
                split_name = f"Node_Split_at_{el_id}"
                
                from_port = f'Port(unit_name="{el_data.get("label", el_id)}", port="Out")'
                
                to_ports = []
                for to_id in out_degree[el_id]:
                    to_el = self.id_to_element.get(to_id)
                    if to_el:
                        unit_name = to_el.get('label', to_id)
                        to_ports.append(f'Port(unit_name="{unit_name}", port="In")')
                
                from_ports_str = f"({from_port},)"
                to_ports_str = f"({', '.join(to_ports)},)"
                
                connector_lines.append(
                    f"""Connector(
        name="{split_name}",
        from_converter_ports={from_ports_str},
        to_converter_ports={to_ports_str},
    ),"""
                )
                connector_index += 1
        
        return connector_lines
    
    def generate_cgm_json(self) -> Dict[str, Any]:
        """
        Generate CGM data as JSON with full coordinates.
        
        Returns:
            CGM data dictionary with components, connectors, coordinates
        """
        logger.info("Generating CGM JSON data with coordinates...")
        
        # Helper function to convert bbox to dict
        def bbox_to_dict(bbox):
            """Convert bbox to dict format, handling Pydantic models and dicts."""
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
        
        # Helper function to convert ports to dict list
        def ports_to_dict_list(ports):
            """Convert ports to list of dicts, handling Pydantic models."""
            if not ports:
                return []
            result = []
            for port in ports:
                if isinstance(port, dict):
                    result.append(port)
                elif hasattr(port, 'model_dump'):
                    result.append(port.model_dump())
                elif hasattr(port, 'dict'):
                    result.append(port.dict())
                else:
                    logger.warning(f"Could not convert port to dict: {port}")
            return result
        
        # Components with full coordinates
        components = []
        for el in self.elements:
            # Convert element fields to proper dict format
            el_bbox = el.get('bbox') if isinstance(el, dict) else getattr(el, 'bbox', None)
            el_ports = el.get('ports', []) if isinstance(el, dict) else getattr(el, 'ports', [])
            
            component = {
                'id': el.get('id') if isinstance(el, dict) else getattr(el, 'id', None),
                'type': el.get('type') if isinstance(el, dict) else getattr(el, 'type', None),
                'label': el.get('label') if isinstance(el, dict) else getattr(el, 'label', None),
                'bbox': bbox_to_dict(el_bbox),  # Full coordinates
                'ports': ports_to_dict_list(el_ports),  # Port coordinates
                'confidence': el.get('confidence', 0.0) if isinstance(el, dict) else getattr(el, 'confidence', 0.0)
            }
            components.append(component)
        
        # Connectors with coordinates
        connectors = []
        for i, conn in enumerate(self.connections):
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            
            if from_id and to_id:
                from_el = self.id_to_element.get(from_id)
                to_el = self.id_to_element.get(to_id)
                
                if from_el and to_el:
                    # Helper to get field value (dict or attribute)
                    def get_field(obj, field, default=None):
                        if isinstance(obj, dict):
                            return obj.get(field, default)
                        return getattr(obj, field, default)
                    
                    # Convert bboxes to dicts
                    from_bbox = get_field(from_el, 'bbox')
                    to_bbox = get_field(to_el, 'bbox')
                    
                    # Helper to convert bbox
                    def bbox_to_dict(bbox):
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
                    
                    connector = {
                        'id': conn.get('id', f'conn_{i}'),
                        'name': f"Conn_{i+1}_{get_field(from_el, 'label', from_id)}_{get_field(to_el, 'label', to_id)}",
                        'from_id': from_id,
                        'to_id': to_id,
                        'from_component': get_field(from_el, 'type'),
                        'to_component': get_field(to_el, 'type'),
                        'from_bbox': bbox_to_dict(from_bbox),
                        'to_bbox': bbox_to_dict(to_bbox),
                        'polyline': conn.get('polyline', []),  # Connection path coordinates
                        'confidence': conn.get('confidence', 0.0),
                        'kind': conn.get('kind', 'process')
                    }
                    connectors.append(connector)
        
        # Detect split/merge points using graph theory
        splits_merges = self.graph_analyzer.detect_splits_and_merges()
        split_merge_points = splits_merges['splits'] + splits_merges['merges']
        
        # System flows (pipeline flows) using graph theory
        system_flows = self.graph_analyzer.analyze_pipeline_flows()
        
        # Graph metrics
        graph_metrics = self.graph_analyzer.calculate_graph_metrics()
        
        cgm_data = {
            'components': components,
            'connectors': connectors,
            'split_merge_points': split_merge_points,
            'system_flows': system_flows,
            'graph_metrics': graph_metrics,
            'network_graph': self.graph_analyzer.get_network_graph_representation(),
            'metadata': {
                'total_components': len(components),
                'total_connectors': len(connectors),
                'total_splits': len(splits_merges['splits']),
                'total_merges': len(splits_merges['merges']),
                'total_flows': len(system_flows),
                'graph_density': graph_metrics.get('density', 0.0),
                'num_cycles': graph_metrics.get('num_cycles', 0)
            }
        }
        
        return cgm_data

