"""
Unit tests for graph utility functions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from src.utils.graph_utils import (
    calculate_iou,
    dedupe_connections
)


class TestCalculateIoU:
    """Tests for calculate_iou function."""
    
    def test_iou_overlapping_boxes(self):
        """Test IoU calculation for overlapping boxes."""
        bbox1 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        bbox2 = {'x': 50, 'y': 50, 'width': 100, 'height': 100}
        
        iou = calculate_iou(bbox1, bbox2)
        
        assert 0 <= iou <= 1, "IoU should be between 0 and 1"
        assert iou > 0, "Overlapping boxes should have IoU > 0"
    
    def test_iou_identical_boxes(self):
        """Test IoU calculation for identical boxes."""
        bbox1 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        bbox2 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        
        iou = calculate_iou(bbox1, bbox2)
        
        assert iou == 1.0, "Identical boxes should have IoU = 1.0"
    
    def test_iou_non_overlapping_boxes(self):
        """Test IoU calculation for non-overlapping boxes."""
        bbox1 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        bbox2 = {'x': 200, 'y': 200, 'width': 100, 'height': 100}
        
        iou = calculate_iou(bbox1, bbox2)
        
        assert iou == 0.0, "Non-overlapping boxes should have IoU = 0.0"
    
    def test_iou_partially_overlapping(self):
        """Test IoU calculation for partially overlapping boxes."""
        bbox1 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        bbox2 = {'x': 50, 'y': 0, 'width': 100, 'height': 100}
        
        iou = calculate_iou(bbox1, bbox2)
        
        assert 0 < iou < 1, "Partially overlapping boxes should have 0 < IoU < 1"
        # Expected IoU: intersection = 50x100 = 5000, union = 15000, IoU = 1/3 ≈ 0.333
        assert abs(iou - 0.333) < 0.1, "IoU should be approximately 0.333"
    
    def test_iou_invalid_bbox(self):
        """Test IoU calculation with invalid bbox."""
        bbox1 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        invalid_bbox = {'x': 0, 'y': 0}  # Missing width/height
        
        iou = calculate_iou(bbox1, invalid_bbox)
        
        assert iou == 0.0, "Invalid bbox should return IoU = 0.0"
    
    def test_iou_zero_size_bbox(self):
        """Test IoU calculation with zero-size bbox."""
        bbox1 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        zero_bbox = {'x': 50, 'y': 50, 'width': 0, 'height': 0}
        
        iou = calculate_iou(bbox1, zero_bbox)
        
        assert iou == 0.0, "Zero-size bbox should return IoU = 0.0"
    
    def test_iou_edge_cases(self):
        """Test IoU calculation with edge cases."""
        # Same center, different sizes
        bbox1 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        bbox2 = {'x': 25, 'y': 25, 'width': 50, 'height': 50}
        
        iou = calculate_iou(bbox1, bbox2)
        assert 0 < iou <= 1, "IoU should be valid"
        
        # bbox2 is inside bbox1
        # Expected: IoU = area(bbox2) / area(bbox1) = 2500 / 10000 = 0.25
        assert abs(iou - 0.25) < 0.01, "IoU for contained box should be 0.25"


class TestDedupeConnections:
    """Tests for dedupe_connections function."""
    
    def test_dedupe_simple_duplicates(self):
        """Test deduplication of simple duplicate connections."""
        connections = [
            {'from_id': 'a', 'to_id': 'b'},
            {'from_id': 'a', 'to_id': 'b'},  # Duplicate
            {'from_id': 'b', 'to_id': 'c'}
        ]
        
        deduped = dedupe_connections(connections)
        
        assert len(deduped) == 2, "Should remove duplicate"
        assert {'from_id': 'a', 'to_id': 'b'} in deduped
        assert {'from_id': 'b', 'to_id': 'c'} in deduped
    
    def test_dedupe_bidirectional(self):
        """Test deduplication of bidirectional connections."""
        connections = [
            {'from_id': 'a', 'to_id': 'b'},
            {'from_id': 'b', 'to_id': 'a'},  # Same as first (bidirectional)
            {'from_id': 'c', 'to_id': 'd'}
        ]
        
        deduped = dedupe_connections(connections)
        
        assert len(deduped) == 2, "Should remove bidirectional duplicate"
    
    def test_dedupe_no_duplicates(self):
        """Test deduplication when no duplicates exist."""
        connections = [
            {'from_id': 'a', 'to_id': 'b'},
            {'from_id': 'b', 'to_id': 'c'},
            {'from_id': 'c', 'to_id': 'd'}
        ]
        
        deduped = dedupe_connections(connections)
        
        assert len(deduped) == 3, "Should keep all connections"
    
    def test_dedupe_empty_list(self):
        """Test deduplication of empty list."""
        connections = []
        
        deduped = dedupe_connections(connections)
        
        assert len(deduped) == 0, "Should return empty list"
    
    def test_dedupe_missing_ids(self):
        """Test deduplication with missing from_id or to_id."""
        connections = [
            {'from_id': 'a', 'to_id': 'b'},
            {'from_id': None, 'to_id': 'c'},  # Missing from_id
            {'from_id': 'd', 'to_id': None},  # Missing to_id
            {'from_id': 'e', 'to_id': 'f'}
        ]
        
        deduped = dedupe_connections(connections)
        
        # Should skip connections with missing IDs
        assert len(deduped) == 2, "Should skip invalid connections"
        assert {'from_id': 'a', 'to_id': 'b'} in deduped
        assert {'from_id': 'e', 'to_id': 'f'} in deduped
    
    def test_dedupe_multiple_duplicates(self):
        """Test deduplication with multiple duplicates."""
        connections = [
            {'from_id': 'a', 'to_id': 'b'},
            {'from_id': 'a', 'to_id': 'b'},  # Duplicate 1
            {'from_id': 'a', 'to_id': 'b'},  # Duplicate 2
            {'from_id': 'c', 'to_id': 'd'},
            {'from_id': 'c', 'to_id': 'd'},  # Duplicate 3
            {'from_id': 'e', 'to_id': 'f'}
        ]
        
        deduped = dedupe_connections(connections)
        
        assert len(deduped) == 3, "Should remove all duplicates"
        assert {'from_id': 'a', 'to_id': 'b'} in deduped
        assert {'from_id': 'c', 'to_id': 'd'} in deduped
        assert {'from_id': 'e', 'to_id': 'f'} in deduped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

