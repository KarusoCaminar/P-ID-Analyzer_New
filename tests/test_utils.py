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
    """Test graph utility functions with precise assertions."""
    try:
        from src.utils.graph_utils import calculate_iou, dedupe_connections
        
        # Test IoU calculation with precise values
        # Test 1: Identical boxes (IoU = 1.0)
        bbox1 = {'x': 0, 'y': 0, 'width': 10, 'height': 10}
        bbox_identisch = {'x': 0, 'y': 0, 'width': 10, 'height': 10}
        iou_1 = calculate_iou(bbox1, bbox_identisch)
        assert abs(iou_1 - 1.0) < 1e-6, f"Identical bbox IoU should be 1.0, got {iou_1}"
        print("[OK] calculate_iou: Identical boxes (IoU = 1.0)")
        
        # Test 2: No overlap (IoU = 0.0)
        bbox_kein_overlap = {'x': 10, 'y': 0, 'width': 10, 'height': 10}
        iou_0 = calculate_iou(bbox1, bbox_kein_overlap)
        assert abs(iou_0 - 0.0) < 1e-6, f"Non-overlapping bbox IoU should be 0.0, got {iou_0}"
        print("[OK] calculate_iou: Non-overlapping boxes (IoU = 0.0)")
        
        # Test 3: Partial overlap (50% overlap area / 150% union area = 0.333...)
        # bbox1: (0,0) to (10,10), area = 100
        # bbox_halb_overlap: (5,0) to (15,10), area = 100
        # Intersection: (5,0) to (10,10), area = 50
        # Union: 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 1/3 ≈ 0.333...
        bbox_halb_overlap = {'x': 5, 'y': 0, 'width': 10, 'height': 10}
        iou_teil = calculate_iou(bbox1, bbox_halb_overlap)
        expected_iou = 50.0 / 150.0  # 1/3 ≈ 0.333...
        assert abs(iou_teil - expected_iou) < 1e-6, f"Partial IoU should be {expected_iou:.6f}, got {iou_teil:.6f}"
        print(f"[OK] calculate_iou: Partial overlap (IoU = {expected_iou:.6f})")
        
        # Test 4: One box inside another (IoU = inner_area / outer_area)
        # bbox1: (0,0) to (10,10), area = 100
        # bbox_inside: (2,2) to (8,8), area = 36
        # Intersection = 36, Union = 100
        # IoU = 36 / 100 = 0.36
        bbox_inside = {'x': 2, 'y': 2, 'width': 6, 'height': 6}
        iou_inside = calculate_iou(bbox1, bbox_inside)
        expected_iou_inside = 36.0 / 100.0  # 0.36
        assert abs(iou_inside - expected_iou_inside) < 1e-6, f"Contained box IoU should be {expected_iou_inside:.6f}, got {iou_inside:.6f}"
        print(f"[OK] calculate_iou: Contained box (IoU = {expected_iou_inside:.6f})")
        
        # Test 5: Edge case - touching boxes (IoU = 0.0, no intersection area)
        bbox_touching = {'x': 10, 'y': 0, 'width': 10, 'height': 10}  # Touches at x=10
        iou_touching = calculate_iou(bbox1, bbox_touching)
        assert abs(iou_touching - 0.0) < 1e-6, f"Touching boxes IoU should be 0.0, got {iou_touching}"
        print("[OK] calculate_iou: Touching boxes (IoU = 0.0)")
        
        print("[OK] calculate_iou works with specific values")
        
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


