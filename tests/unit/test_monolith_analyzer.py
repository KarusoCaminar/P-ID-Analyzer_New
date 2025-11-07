"""
Unit tests for MonolithAnalyzer with mocking (no real API calls).
"""

import sys
import json
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from src.analyzer.analysis.monolith_analyzer import MonolithAnalyzer
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock(spec=LLMClient)
    client.call_llm = Mock(return_value={
        "elements": [],
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
    })
    return client


@pytest.fixture
def mock_knowledge_manager():
    """Create mock knowledge manager."""
    km = Mock(spec=KnowledgeManager)
    km.get_known_types = Mock(return_value=["Pump", "Valve", "Sensor", "Mixer", "Source"])
    km.get_all_aliases = Mock(return_value={})
    return km


@pytest.fixture
def mock_config_service():
    """Create mock config service."""
    config_service = Mock(spec=ConfigService)
    mock_config = Mock()
    mock_config.prompts = Mock()
    mock_config.prompts.monolithic_analysis_prompt_template = "Test prompt"
    config_service.get_config = Mock(return_value=mock_config)
    config_service.get_raw_config = Mock(return_value={})
    return config_service


@pytest.fixture
def test_image():
    """Create a temporary test image."""
    img = Image.new('RGB', (2000, 1500), color='white')
    temp_dir = Path(tempfile.mkdtemp())
    img_path = temp_dir / "test_image.png"
    img.save(img_path)
    yield str(img_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_image_small():
    """Create a small test image."""
    img = Image.new('RGB', (500, 400), color='black')
    temp_dir = Path(tempfile.mkdtemp())
    img_path = temp_dir / "test_image_small.png"
    img.save(img_path)
    yield str(img_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def monolith_analyzer(mock_llm_client, mock_knowledge_manager, mock_config_service):
    """Create MonolithAnalyzer instance with mocks."""
    model_strategy = {
        'monolith_model': {'id': 'gemini-2.5-pro'},
        'monolith_whole_image': False
    }
    logic_parameters = {}
    
    return MonolithAnalyzer(
        llm_client=mock_llm_client,
        knowledge_manager=mock_knowledge_manager,
        config_service=mock_config_service,
        model_strategy=model_strategy,
        logic_parameters=logic_parameters,
        symbol_library=None
    )


class TestMonolithAnalyzer:
    """Tests for MonolithAnalyzer."""
    
    def test_monolith_analyzer_initialization(self, monolith_analyzer):
        """Test MonolithAnalyzer initialization."""
        assert monolith_analyzer.llm_client is not None
        assert monolith_analyzer.knowledge_manager is not None
        assert monolith_analyzer.config_service is not None
        assert monolith_analyzer.model_strategy is not None
    
    def test_analyze_returns_elements_and_connections(self, monolith_analyzer, test_image):
        """Test that analyze returns elements and connections."""
        # Set element list JSON for connection detection
        monolith_analyzer.element_list_json = json.dumps([
            {"id": "P-101", "type": "Pump"},
            {"id": "V-101", "type": "Valve"}
        ])
        
        result = monolith_analyzer.analyze(
            image_path=test_image,
            output_dir=None,
            excluded_zones=None
        )
        
        assert result is not None
        assert "elements" in result
        assert "connections" in result
        assert isinstance(result["elements"], list)
        assert isinstance(result["connections"], list)
    
    def test_analyze_with_invalid_image(self, monolith_analyzer):
        """Test analyze with invalid image path."""
        result = monolith_analyzer.analyze(
            image_path="nonexistent/image.png",
            output_dir=None,
            excluded_zones=None
        )
        
        assert result is not None
        assert result == {"elements": [], "connections": []}
    
    def test_analyze_with_whole_image_strategy(self, monolith_analyzer, test_image):
        """Test analyze with whole-image strategy."""
        monolith_analyzer.model_strategy['monolith_whole_image'] = True
        monolith_analyzer.element_list_json = json.dumps([
            {"id": "P-101", "type": "Pump"},
            {"id": "V-101", "type": "Valve"}
        ])
        
        result = monolith_analyzer.analyze(
            image_path=test_image,
            output_dir=None,
            excluded_zones=None
        )
        
        assert result is not None
        assert "elements" in result
        assert "connections" in result
    
    def test_analyze_with_excluded_zones(self, monolith_analyzer, test_image):
        """Test analyze with excluded zones."""
        excluded_zones = [
            {"x": 0.0, "y": 0.0, "width": 0.2, "height": 0.2}
        ]
        monolith_analyzer.element_list_json = json.dumps([
            {"id": "P-101", "type": "Pump"}
        ])
        
        result = monolith_analyzer.analyze(
            image_path=test_image,
            output_dir=None,
            excluded_zones=excluded_zones
        )
        
        assert result is not None
        assert "elements" in result
        assert "connections" in result
    
    def test_calculate_optimal_quadrant_strategy_small(self, monolith_analyzer):
        """Test quadrant strategy calculation for small images."""
        # Small image (< 3000px) should return 0 (whole image)
        num_quadrants = monolith_analyzer._calculate_optimal_quadrant_strategy(1000, 800)
        assert num_quadrants == 0
    
    def test_calculate_optimal_quadrant_strategy_medium(self, monolith_analyzer):
        """Test quadrant strategy calculation for medium images."""
        # Medium image (3000-6000px) should return 4 quadrants
        num_quadrants = monolith_analyzer._calculate_optimal_quadrant_strategy(4000, 3000)
        assert num_quadrants == 4
    
    def test_calculate_optimal_quadrant_strategy_large(self, monolith_analyzer):
        """Test quadrant strategy calculation for large images."""
        # Large image (> 6000px) should return more quadrants
        num_quadrants = monolith_analyzer._calculate_optimal_quadrant_strategy(12000, 10000)
        assert num_quadrants >= 4  # Should have at least 4 quadrants for large images


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

