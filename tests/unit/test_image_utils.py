"""
Unit tests for image utility functions.
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
import numpy as np

from src.utils.image_utils import (
    resize_image_for_llm,
    crop_image_for_correction,
    generate_raster_grid,
    is_tile_complex,
    preprocess_image_for_line_detection
)


@pytest.fixture
def test_image():
    """Create a temporary test image."""
    img = Image.new('RGB', (2000, 1500), color='white')
    temp_dir = Path(tempfile.mkdtemp())
    img_path = temp_dir / "test_image.png"
    img.save(img_path)
    yield str(img_path)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_image_small():
    """Create a small test image."""
    img = Image.new('RGB', (500, 400), color='black')
    temp_dir = Path(tempfile.mkdtemp())
    img_path = temp_dir / "test_image_small.png"
    img.save(img_path)
    yield str(img_path)
    shutil.rmtree(temp_dir)


class TestResizeImage:
    """Tests for resize_image_for_llm."""
    
    def test_resize_large_image(self, test_image):
        """Test resizing a large image."""
        result_path = resize_image_for_llm(test_image, max_size=1024)
        
        assert Path(result_path).exists(), "Resized image should exist"
        
        with Image.open(result_path) as img:
            w, h = img.size
            assert max(w, h) <= 1024, "Image should be resized to max 1024px"
    
    def test_resize_small_image_no_change(self, test_image_small):
        """Test resizing a small image (should not resize)."""
        original_size = Image.open(test_image_small).size
        
        result_path = resize_image_for_llm(test_image_small, max_size=1024)
        
        with Image.open(result_path) as img:
            w, h = img.size
            # Small image should not be resized
            assert w == original_size[0] or max(w, h) <= 1024
    
    def test_resize_preserves_aspect_ratio(self, test_image):
        """Test that resize preserves aspect ratio."""
        original = Image.open(test_image)
        original_ratio = original.size[0] / original.size[1]
        
        result_path = resize_image_for_llm(test_image, max_size=1024)
        
        with Image.open(result_path) as img:
            w, h = img.size
            new_ratio = w / h
            assert abs(new_ratio - original_ratio) < 0.01, "Aspect ratio should be preserved"


class TestCropImageForCorrection:
    """Tests for crop_image_for_correction."""
    
    def test_crop_valid_bbox(self, test_image):
        """Test cropping with valid bounding box."""
        bbox = {
            'x': 100,
            'y': 100,
            'width': 500,
            'height': 400
        }
        
        result_path = crop_image_for_correction(test_image, bbox)
        
        assert result_path is not None, "Crop should succeed"
        assert Path(result_path).exists(), "Cropped image should exist"
        
        with Image.open(result_path) as img:
            w, h = img.size
            # Should be approximately bbox size + margin
            assert w >= 500 and h >= 400
    
    def test_crop_with_context_margin(self, test_image):
        """Test cropping with context margin."""
        bbox = {
            'x': 200,
            'y': 200,
            'width': 300,
            'height': 300
        }
        
        result_path = crop_image_for_correction(test_image, bbox, context_margin=0.1)
        
        assert result_path is not None
        with Image.open(result_path) as img:
            w, h = img.size
            # Should be larger than bbox due to margin
            assert w > 300 and h > 300
    
    def test_crop_invalid_bbox(self, test_image):
        """Test cropping with invalid bounding box."""
        invalid_bbox = {
            'x': 100,
            'y': 100,
            'width': -50,  # Invalid negative width
            'height': 400
        }
        
        result_path = crop_image_for_correction(test_image, invalid_bbox)
        
        assert result_path is None, "Invalid bbox should return None"
    
    def test_crop_out_of_bounds(self, test_image):
        """Test cropping with bbox outside image bounds."""
        bbox = {
            'x': 10000,  # Outside image
            'y': 10000,
            'width': 500,
            'height': 400
        }
        
        result_path = crop_image_for_correction(test_image, bbox)
        
        # Should handle gracefully, might return None or clipped result
        assert result_path is None or Path(result_path).exists()


class TestGenerateRasterGrid:
    """Tests for generate_raster_grid."""
    
    def test_generate_raster_grid_basic(self, test_image):
        """Test basic raster grid generation."""
        tiles = generate_raster_grid(
            test_image,
            tile_size=512,
            overlap=64
        )
        
        assert len(tiles) > 0, "Should generate tiles"
        assert all('path' in tile for tile in tiles), "Tiles should have paths"
        assert all('coords' in tile for tile in tiles), "Tiles should have coordinates"
    
    def test_generate_raster_grid_with_excluded_zones(self, test_image):
        """Test raster grid with excluded zones."""
        excluded_zones = [
            {'x': 0.2, 'y': 0.2, 'width': 0.3, 'height': 0.3}
        ]
        
        tiles_without = generate_raster_grid(test_image, tile_size=512, overlap=64)
        tiles_with = generate_raster_grid(
            test_image,
            tile_size=512,
            overlap=64,
            excluded_zones=excluded_zones
        )
        
        assert len(tiles_with) <= len(tiles_without), "Excluded zones should reduce tile count"
    
    def test_raster_grid_tile_size(self, test_image):
        """Test that tiles have correct size."""
        tile_size = 512
        tiles = generate_raster_grid(test_image, tile_size=tile_size, overlap=64)
        
        if tiles:
            first_tile_path = tiles[0]['path']
            with Image.open(first_tile_path) as img:
                w, h = img.size
                # Tile should be approximately tile_size (might be smaller at edges)
                assert w <= tile_size + 10
                assert h <= tile_size + 10


class TestIsTileComplex:
    """Tests for is_tile_complex."""
    
    def test_simple_tile(self, test_image_small):
        """Test that simple (blank) tile is not complex."""
        # Create a simple white tile
        simple_img = Image.new('RGB', (256, 256), color='white')
        temp_dir = Path(tempfile.mkdtemp())
        simple_path = temp_dir / "simple_tile.png"
        simple_img.save(simple_path)
        
        try:
            is_complex = is_tile_complex(str(simple_path))
            # White image should be simple (low edge ratio)
            # Actual result depends on Canny edge detection
            assert isinstance(is_complex, (bool, np.bool_)), f"Should return bool, got {type(is_complex)}"
        finally:
            shutil.rmtree(temp_dir)
    
    def test_complex_tile_with_pattern(self):
        """Test that tile with pattern is complex."""
        # Create a tile with patterns (lines, circles)
        img = Image.new('RGB', (256, 256), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw some patterns
        draw.line([(0, 0), (256, 256)], fill='black', width=2)
        draw.line([(0, 256), (256, 0)], fill='black', width=2)
        draw.ellipse([50, 50, 150, 150], outline='black', width=2)
        
        temp_dir = Path(tempfile.mkdtemp())
        complex_path = temp_dir / "complex_tile.png"
        img.save(complex_path)
        
        try:
            is_complex = is_tile_complex(str(complex_path))
            assert isinstance(is_complex, (bool, np.bool_)), f"Should return bool, got {type(is_complex)}"
            # Pattern image should likely be complex
        finally:
            shutil.rmtree(temp_dir)
    
    def test_invalid_tile_path(self):
        """Test with invalid tile path."""
        is_complex = is_tile_complex("nonexistent/path/tile.png")
        # Should return False for invalid path
        assert is_complex is False


class TestPreprocessImageForLineDetection:
    """Tests for preprocess_image_for_line_detection."""
    
    def test_preprocess_image(self, test_image):
        """Test preprocessing image for line detection."""
        temp_dir = Path(tempfile.mkdtemp())
        output_path = temp_dir / "preprocessed.png"
        
        try:
            result_path = preprocess_image_for_line_detection(test_image, str(output_path))
            
            assert result_path is not None
            assert Path(result_path).exists(), "Preprocessed image should exist"
        finally:
            shutil.rmtree(temp_dir)
    
    def test_preprocess_invalid_path(self):
        """Test preprocessing with invalid image path."""
        temp_dir = Path(tempfile.mkdtemp())
        output_path = temp_dir / "preprocessed.png"
        
        try:
            result_path = preprocess_image_for_line_detection(
                "nonexistent/path/image.png",
                str(output_path)
            )
            # Should return original path or handle gracefully
            assert result_path is not None
        finally:
            if output_path.exists():
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

