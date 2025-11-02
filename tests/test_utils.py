"""
Test utility functions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_type_utils():
    """Test type utility functions."""
    try:
        from src.utils.type_utils import is_valid_bbox, bbox_from_connection
        
        # Test valid bbox
        valid_bbox = {'x': 10, 'y': 20, 'width': 100, 'height': 50}
        assert is_valid_bbox(valid_bbox) == True, "Valid bbox should return True"
        print("[OK] is_valid_bbox works for valid bbox")
        
        # Test invalid bbox
        invalid_bbox = {'x': 10, 'y': 20}  # Missing width/height
        assert is_valid_bbox(invalid_bbox) == False, "Invalid bbox should return False"
        print("[OK] is_valid_bbox works for invalid bbox")
        
        # Test bbox_from_connection
        conn = {'from_id': 'elem1', 'to_id': 'elem2'}
        elements_map = {
            'elem1': {'id': 'elem1', 'bbox': {'x': 0, 'y': 0, 'width': 50, 'height': 50}},
            'elem2': {'id': 'elem2', 'bbox': {'x': 100, 'y': 100, 'width': 50, 'height': 50}}
        }
        result = bbox_from_connection(conn, elements_map)
        assert result is not None, "bbox_from_connection should return bbox"
        assert 'x' in result and 'y' in result and 'width' in result and 'height' in result
        print("[OK] bbox_from_connection works")
        
        return True
    except Exception as e:
        print(f"[FAIL] Type utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graph_utils():
    """Test graph utility functions."""
    try:
        from src.utils.graph_utils import calculate_iou, dedupe_connections
        
        # Test IoU calculation
        bbox1 = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
        bbox2 = {'x': 50, 'y': 50, 'width': 100, 'height': 100}
        iou = calculate_iou(bbox1, bbox2)
        assert 0 <= iou <= 1, "IoU should be between 0 and 1"
        print("[OK] calculate_iou works")
        
        # Test connection deduplication
        connections = [
            {'from_id': 'a', 'to_id': 'b'},
            {'from_id': 'a', 'to_id': 'b'},  # Duplicate
            {'from_id': 'b', 'to_id': 'c'}
        ]
        deduped = dedupe_connections(connections)
        assert len(deduped) == 2, "Deduplicated connections should have 2 items"
        print("[OK] dedupe_connections works")
        
        return True
    except Exception as e:
        print(f"[FAIL] Graph utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_type_utils()
    success2 = test_graph_utils()
    sys.exit(0 if (success1 and success2) else 1)


