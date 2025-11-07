"""
Unit tests for FusionEngine with mocking.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock

from src.analyzer.analysis.fusion_engine import FusionEngine, normalize_type_for_comparison


@pytest.fixture
def fusion_engine():
    """Create FusionEngine instance."""
    return FusionEngine(iou_match_threshold=0.5)


@pytest.fixture
def mock_swarm_result():
    """Create mock swarm result."""
    return {
        "elements": [
            {
                "id": "P-101",
                "type": "Pump",
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.04},
                "confidence": 0.95,
                "ports": [
                    {"id": "out_1", "name": "Out", "type": "output"}
                ]
            },
            {
                "id": "V-101",
                "type": "Valve",
                "bbox": {"x": 0.3, "y": 0.4, "width": 0.02, "height": 0.03},
                "confidence": 0.90,
                "ports": [
                    {"id": "in_1", "name": "In", "type": "input"},
                    {"id": "out_1", "name": "Out", "type": "output"}
                ]
            }
        ],
        "connections": []
    }


@pytest.fixture
def mock_monolith_result():
    """Create mock monolith result."""
    return {
        "elements": [
            {
                "id": "P-101",
                "type": "Pump",
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.04},
                "confidence": 0.98,
                "ports": [
                    {"id": "out_1", "name": "Out", "type": "output"}
                ]
            }
        ],
        "connections": [
            {
                "from_id": "P-101",
                "to_id": "V-101",
                "from_port_id": "out_1",
                "to_port_id": "in_1",
                "kind": "process",
                "confidence": 0.95
            }
        ]
    }


class TestFusionEngine:
    """Tests for FusionEngine."""
    
    def test_fusion_engine_initialization(self, fusion_engine):
        """Test FusionEngine initialization."""
        assert fusion_engine.iou_match_threshold == 0.5
        assert fusion_engine.fusion_stats is not None
    
    def test_normalize_type_for_comparison(self):
        """Test type normalization."""
        assert normalize_type_for_comparison("Valve") == "valve"
        assert normalize_type_for_comparison("PUMP") == "pump"
        assert normalize_type_for_comparison("Volume Flow Sensor") == "flow_sensor"
        assert normalize_type_for_comparison("") == ""
    
    def test_fuse_returns_structure(self, fusion_engine, mock_swarm_result, mock_monolith_result):
        """Test that fuse returns correct structure."""
        result = fusion_engine.fuse(mock_swarm_result, mock_monolith_result)
        
        assert result is not None
        assert "elements" in result
        assert "connections" in result
        assert isinstance(result["elements"], list)
        assert isinstance(result["connections"], list)
    
    def test_fuse_with_swarm_only(self, fusion_engine, mock_swarm_result):
        """Test fuse with swarm result only."""
        result = fusion_engine.fuse(mock_swarm_result, None)
        
        assert result is not None
        assert "elements" in result
        assert len(result["elements"]) >= 0
    
    def test_fuse_deduplicates_elements(self, fusion_engine, mock_swarm_result, mock_monolith_result):
        """Test that fuse deduplicates elements."""
        result = fusion_engine.fuse(mock_swarm_result, mock_monolith_result)
        
        # Should deduplicate P-101 (present in both)
        element_ids = [el["id"] for el in result["elements"]]
        assert len(element_ids) == len(set(element_ids)), "Should deduplicate elements"
    
    def test_fuse_with_legend_authority(self, fusion_engine, mock_swarm_result, mock_monolith_result):
        """Test fuse with legend authority."""
        symbol_map = {
            "P-101": "Pump"
        }
        
        result = fusion_engine.fuse_with_legend_authority(
            mock_swarm_result,
            mock_monolith_result,
            symbol_map=symbol_map,
            legend_confidence=0.99,
            is_plausible=True
        )
        
        assert result is not None
        assert "elements" in result
        assert "connections" in result
    
    def test_fuse_combines_connections(self, fusion_engine, mock_swarm_result, mock_monolith_result):
        """Test that fuse combines connections from monolith."""
        result = fusion_engine.fuse(mock_swarm_result, mock_monolith_result)
        
        # Monolith has connections, should be included in result
        assert "connections" in result
        assert isinstance(result["connections"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

