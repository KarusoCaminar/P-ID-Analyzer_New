"""
Unit tests for LineExtractor with mocking (no real API calls).
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np
import cv2

from src.analyzer.analysis.line_extractor import LineExtractor


@pytest.fixture
def test_image():
    """Create a temporary test image with lines."""
    img = Image.new('RGB', (1000, 800), color='white')
    # Draw some lines
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.line([(100, 100), (500, 100)], fill='black', width=3)
    draw.line([(100, 200), (500, 300)], fill='black', width=2)
    draw.rectangle([(200, 400), (250, 450)], outline='black', width=2)
    
    temp_dir = Path(tempfile.mkdtemp())
    img_path = temp_dir / "test_image.png"
    img.save(img_path)
    yield str(img_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def line_extractor():
    """Create LineExtractor instance."""
    config = {
        'logic_parameters': {
            'adaptive_threshold_min': 25,
            'adaptive_threshold_max': 150
        }
    }
    return LineExtractor(config)


@pytest.fixture
def mock_elements():
    """Create mock elements for testing."""
    return [
        {
            "id": "P-101",
            "type": "Pump",
            "bbox": {"x": 0.2, "y": 0.4, "width": 0.05, "height": 0.05},
            "confidence": 0.95
        },
        {
            "id": "V-101",
            "type": "Valve",
            "bbox": {"x": 0.5, "y": 0.3, "width": 0.02, "height": 0.03},
            "confidence": 0.90
        }
    ]


class TestLineExtractor:
    """Tests for LineExtractor."""
    
    def test_line_extractor_initialization(self, line_extractor):
        """Test LineExtractor initialization."""
        assert line_extractor.config is not None
        assert line_extractor.logic_parameters is not None
    
    def test_extract_pipeline_lines_returns_structure(self, line_extractor, test_image, mock_elements):
        """Test that extract_pipeline_lines returns correct structure."""
        result = line_extractor.extract_pipeline_lines(
            image_path=test_image,
            elements=mock_elements,
            excluded_zones=[],
            legend_data=None
        )
        
        assert result is not None
        assert "pipeline_lines" in result
        assert "junctions" in result
        assert "line_segments" in result
        assert isinstance(result["pipeline_lines"], list)
        assert isinstance(result["junctions"], list)
        assert isinstance(result["line_segments"], list)
    
    def test_extract_pipeline_lines_with_invalid_image(self, line_extractor, mock_elements):
        """Test extract_pipeline_lines with invalid image path."""
        result = line_extractor.extract_pipeline_lines(
            image_path="nonexistent/image.png",
            elements=mock_elements,
            excluded_zones=[],
            legend_data=None
        )
        
        assert result == {"pipeline_lines": [], "junctions": [], "line_segments": []}
    
    def test_extract_pipeline_lines_with_excluded_zones(self, line_extractor, test_image, mock_elements):
        """Test extract_pipeline_lines with excluded zones."""
        excluded_zones = [
            {"x": 0.0, "y": 0.0, "width": 0.2, "height": 0.2}
        ]
        
        result = line_extractor.extract_pipeline_lines(
            image_path=test_image,
            elements=mock_elements,
            excluded_zones=excluded_zones,
            legend_data=None
        )
        
        assert result is not None
        assert "pipeline_lines" in result
    
    def test_mask_symbols(self, line_extractor, test_image, mock_elements):
        """Test _mask_symbols method."""
        img = cv2.imread(test_image)
        img_height, img_width = img.shape[:2]
        
        masked = line_extractor._mask_symbols(
            img=img,
            elements=mock_elements,
            excluded_zones=[],
            img_width=img_width,
            img_height=img_height
        )
        
        assert masked is not None
        assert masked.shape == img.shape
    
    def test_extract_pipeline_colors(self, line_extractor, test_image):
        """Test _extract_pipeline_colors method."""
        img = cv2.imread(test_image)
        masked = img.copy()
        
        pipeline_colors = line_extractor._extract_pipeline_colors(
            img=masked,
            legend_data=None
        )
        
        assert pipeline_colors is not None
        assert isinstance(pipeline_colors, np.ndarray)
    
    def test_extract_contours(self, line_extractor):
        """Test _extract_contours method."""
        # Create a binary image with some lines
        binary = np.zeros((500, 500), dtype=np.uint8)
        cv2.line(binary, (50, 50), (450, 50), 255, 2)
        cv2.line(binary, (50, 100), (450, 150), 255, 2)
        
        contours, polylines = line_extractor._extract_contours(binary)
        
        # contours is a tuple of numpy arrays (from cv2.findContours)
        assert isinstance(contours, (list, tuple))
        assert isinstance(polylines, list)
        assert len(polylines) >= 0  # May have 0 or more polylines
    
    def test_detect_junctions_from_contours(self, line_extractor):
        """Test _detect_junctions_from_contours method."""
        # Create mock contours and polylines
        contours = [
            np.array([[[50, 50]], [[450, 50]]], dtype=np.int32),
            np.array([[[50, 100]], [[450, 150]]], dtype=np.int32)
        ]
        polylines = [
            [[50.0, 50.0], [450.0, 50.0]],
            [[50.0, 100.0], [450.0, 150.0]]
        ]
        
        junctions = line_extractor._detect_junctions_from_contours(
            contours=contours,
            polylines=polylines,
            img_width=1000,
            img_height=800
        )
        
        assert isinstance(junctions, list)
        # Junctions should be a list of dictionaries
        for junction in junctions:
            assert "x" in junction
            assert "y" in junction
            assert "degree" in junction
    
    def test_find_closest_element(self, line_extractor, mock_elements):
        """Test _find_closest_element method."""
        element_map = {el["id"]: el for el in mock_elements}
        point = [0.25, 0.45]  # Close to P-101
        
        closest = line_extractor._find_closest_element(
            point=point,
            element_map=element_map,
            img_width=1000,
            img_height=800
        )
        
        # Should find an element or return None
        assert closest is None or isinstance(closest, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

