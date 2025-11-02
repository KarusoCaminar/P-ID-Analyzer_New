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
        self.id_to_element = {el.get('id'): el for el in elements if el.get('id')}
        
        # Use graph theory analyzer
        self.graph_analyzer = GraphTheoryAnalyzer(elements, connections)
        
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
        
        # Process all connections
        for i, conn in enumerate(self.connections):
            connector_code = self._generate_connector_code(conn, i)
            if connector_code:
                connector_lines.append(connector_code)
        
        # Handle split/merge points
        split_merge_connectors = self._detect_split_merge_connectors()
        connector_lines.extend(split_merge_connectors)
        
        connectors_code = "\n".join(f"        {line}" for line in connector_lines)
        
        return f"""# Automatisch generiertes Netzwerk-Objekt
cgm_network = Network(
    connectors=(
{connectors_code}
    )
)"""
    
    def _generate_connector_code(self, conn: Dict[str, Any], index: int) -> Optional[str]:
        """Generate connector code for a single connection."""
        from_id = conn.get('from_id')
        to_id = conn.get('to_id')
        
        if not from_id or not to_id:
            return None
        
        from_el = self.id_to_element.get(from_id)
        to_el = self.id_to_element.get(to_id)
        
        if not from_el or not to_el:
            return None
        
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

