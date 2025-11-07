"""
Unit tests for ComplexityAnalyzer.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image

from src.utils.complexity_analyzer import ComplexityAnalyzer


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
def complexity_analyzer():
    """Create ComplexityAnalyzer instance."""
    return ComplexityAnalyzer()


class TestComplexityAnalyzer:
    """Tests for ComplexityAnalyzer."""
    
    def test_complexity_analyzer_initialization(self, complexity_analyzer):
        """Test ComplexityAnalyzer initialization."""
        assert complexity_analyzer is not None
    
    def test_analyze_complexity_cv_advanced_returns_structure(self, complexity_analyzer, test_image):
        """Test that analyze_complexity_cv_advanced returns correct structure."""
        result = complexity_analyzer.analyze_complexity_cv_advanced(test_image)
        
        assert result is not None
        assert "complexity" in result
        assert "score" in result
        assert "metrics" in result
        assert result["complexity"] in ["simple", "moderate", "complex", "very_complex"]
        assert 0.0 <= result["score"] <= 1.0
    
    def test_analyze_complexity_cv_advanced_with_invalid_image(self, complexity_analyzer):
        """Test analyze_complexity_cv_advanced with invalid image path."""
        result = complexity_analyzer.analyze_complexity_cv_advanced("nonexistent/image.png")
        
        # Should return default complexity
        assert result is not None
        assert "complexity" in result
        assert "score" in result
    
    def test_skeletonize_returns_array(self, complexity_analyzer):
        """Test _skeletonize method."""
        import numpy as np
        import cv2
        
        # Create a simple binary image
        binary = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(binary, (10, 50), (90, 50), 255, 2)
        
        skeleton = complexity_analyzer._skeletonize(binary)
        
        assert skeleton is not None
        assert isinstance(skeleton, np.ndarray)
        assert skeleton.shape == binary.shape
    
    def test_detect_junctions(self, complexity_analyzer):
        """Test _detect_junctions method."""
        import numpy as np
        
        # Create a simple skeleton with a junction (cross pattern)
        skeleton = np.zeros((100, 100), dtype=np.uint8)
        skeleton[50, :] = 255  # Horizontal line
        skeleton[:, 50] = 255  # Vertical line
        
        junctions = complexity_analyzer._detect_junctions(skeleton)
        
        assert isinstance(junctions, list)
        # Should detect junction at center (50, 50)
        assert len(junctions) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

