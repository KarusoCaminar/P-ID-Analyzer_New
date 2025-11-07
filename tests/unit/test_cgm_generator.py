"""
Unit tests for CGMGenerator.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from src.analyzer.output.cgm_generator import CGMGenerator


@pytest.fixture
def mock_elements():
    """Create mock elements."""
    return [
        {
            "id": "P-101",
            "type": "Pump",
            "label": "Main Pump",
            "bbox": {"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.04},
            "ports": [
                {"id": "out_1", "name": "Out", "type": "output"}
            ]
        },
        {
            "id": "V-101",
            "type": "Valve",
            "label": "Control Valve",
            "bbox": {"x": 0.3, "y": 0.4, "width": 0.02, "height": 0.03},
            "ports": [
                {"id": "in_1", "name": "In", "type": "input"},
                {"id": "out_1", "name": "Out", "type": "output"},
                {"id": "control_1", "name": "Control", "type": "control"}
            ]
        }
    ]


@pytest.fixture
def mock_connections():
    """Create mock connections."""
    return [
        {
            "from_id": "P-101",
            "to_id": "V-101",
            "from_port_id": "out_1",
            "to_port_id": "in_1",
            "kind": "process"
        },
        {
            "from_id": "ISA",
            "to_id": "V-101",
            "from_port_id": "out_1",
            "to_port_id": "control_1",
            "kind": "control"
        }
    ]


@pytest.fixture
def cgm_generator(mock_elements, mock_connections):
    """Create CGMGenerator instance."""
    return CGMGenerator(elements=mock_elements, connections=mock_connections)


class TestCGMGenerator:
    """Tests for CGMGenerator."""
    
    def test_cgm_generator_initialization(self, cgm_generator):
        """Test CGMGenerator initialization."""
        assert cgm_generator.elements is not None
        assert cgm_generator.connections is not None
    
    def test_generate_cgm_json_returns_structure(self, cgm_generator):
        """Test that generate_cgm_json returns correct structure."""
        cgm_data = cgm_generator.generate_cgm_json()
        
        assert cgm_data is not None
        assert "components" in cgm_data
        assert "connectors" in cgm_data
        assert isinstance(cgm_data["components"], list)
        assert isinstance(cgm_data["connectors"], list)
    
    def test_generate_cgm_json_has_components(self, cgm_generator):
        """Test that generate_cgm_json includes components."""
        cgm_data = cgm_generator.generate_cgm_json()
        
        assert len(cgm_data["components"]) > 0
        # Each component should have required fields
        for component in cgm_data["components"]:
            assert "id" in component
            assert "type" in component
    
    def test_generate_cgm_json_has_connectors(self, cgm_generator):
        """Test that generate_cgm_json includes connectors."""
        cgm_data = cgm_generator.generate_cgm_json()
        
        # Should have connectors if connections exist
        assert isinstance(cgm_data["connectors"], list)
    
    def test_generate_cgm_network_returns_code(self, cgm_generator):
        """Test that generate_cgm_network returns Python code."""
        network_code = cgm_generator.generate_cgm_network()
        
        assert network_code is not None
        assert isinstance(network_code, str)
        assert "class" in network_code or "def" in network_code  # Should contain Python code
    
    def test_generate_cgm_json_with_empty_inputs(self):
        """Test generate_cgm_json with empty inputs."""
        generator = CGMGenerator(elements=[], connections=[])
        cgm_data = generator.generate_cgm_json()
        
        assert cgm_data is not None
        assert cgm_data["components"] == []
        assert cgm_data["connectors"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

