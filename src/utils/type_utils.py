"""
Type utilities for P&ID analysis.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def is_valid_bbox(bbox: Optional[Dict[str, Any]]) -> bool:
    """
    Check if a bounding box is valid.
    
    Args:
        bbox: Bounding box dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        return (
            isinstance(bbox, dict)
            and all(k in bbox for k in ("x", "y", "width", "height"))
            and all(isinstance(bbox[k], (int, float)) for k in ("x", "y", "width", "height"))
            and bbox["width"] > 0
            and bbox["height"] > 0
        )
    except Exception:
        return False


def normalize_bbox(
    bbox: Any,
    img_width: int = 1,
    img_height: int = 1
) -> Optional[Dict[str, float]]:
    """
    Normalize a bounding box to dict format.
    
    Args:
        bbox: Bounding box (dict, list, or other)
        img_width: Image width for normalization
        img_height: Image height for normalization
        
    Returns:
        Normalized bbox dict or None if invalid
    """
    if isinstance(bbox, dict) and all(k in bbox for k in ('x', 'y', 'width', 'height')):
        return {
            'x': float(bbox['x']),
            'y': float(bbox['y']),
            'width': float(bbox['width']),
            'height': float(bbox['height'])
        }
    elif isinstance(bbox, list) and len(bbox) == 4:
        try:
            x, y, width, height = bbox
            return {
                'x': float(x),
                'y': float(y),
                'width': float(width),
                'height': float(height)
            }
        except (ValueError, IndexError):
            logger.error(f"Could not convert bbox list {bbox} to dict format.")
            return None
    
    return None


def bbox_from_connection(
    conn: Dict[str, Any],
    elements_map: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, float]]:
    """
    Calculate bounding box that encompasses a connection between two elements.
    
    Args:
        conn: Connection dictionary with 'from_id' and 'to_id'
        elements_map: Dictionary mapping element IDs to element dictionaries
        
    Returns:
        Bounding box dictionary or None if invalid
    """
    from_id = conn.get('from_id')
    to_id = conn.get('to_id')
    
    if not from_id or not to_id:
        return None
    
    from_elem = elements_map.get(from_id)
    to_elem = elements_map.get(to_id)
    
    if not from_elem or not to_elem:
        return None
    
    from_bbox = from_elem.get('bbox')
    to_bbox = to_elem.get('bbox')
    
    if not from_bbox or not to_bbox:
        return None
    
    if not is_valid_bbox(from_bbox) or not is_valid_bbox(to_bbox):
        return None
    
    # Calculate encompassing bbox
    x1 = min(float(from_bbox['x']), float(to_bbox['x']))
    y1 = min(float(from_bbox['y']), float(to_bbox['y']))
    x2 = max(
        float(from_bbox['x']) + float(from_bbox['width']),
        float(to_bbox['x']) + float(to_bbox['width'])
    )
    y2 = max(
        float(from_bbox['y']) + float(from_bbox['height']),
        float(to_bbox['y']) + float(to_bbox['height'])
    )
    
    return {
        'x': x1,
        'y': y1,
        'width': x2 - x1,
        'height': y2 - y1
    }

