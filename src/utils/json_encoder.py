"""
Custom JSON encoder for P&ID analysis objects.

Handles serialization of Pydantic models (BBox, Element, Connection) and other
non-serializable objects to JSON-compatible dictionaries.
"""

import json
from typing import Any
from datetime import datetime


class PydanticJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles Pydantic models and other non-serializable objects.
    
    Automatically converts:
    - Pydantic models (BBox, Element, Connection, etc.) to dicts
    - BBox objects to dicts with x, y, width, height
    - Other Pydantic models using model_dump() or dict()
    - Datetime objects to ISO format strings
    """
    
    def default(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to JSON-serializable format.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation of the object
        """
        # Handle Pydantic models (BBox, Element, Connection, etc.)
        if hasattr(obj, 'model_dump'):
            # Pydantic v2
            try:
                return obj.model_dump()
            except Exception:
                # Fallback to dict conversion
                pass
        elif hasattr(obj, 'dict'):
            # Pydantic v1
            try:
                return obj.dict()
            except Exception:
                # Fallback to dict conversion
                pass
        
        # Handle BBox-like objects with attributes
        if hasattr(obj, '__dict__') and hasattr(obj, 'x') and hasattr(obj, 'y') and hasattr(obj, 'width') and hasattr(obj, 'height'):
            return {
                'x': float(getattr(obj, 'x', 0)),
                'y': float(getattr(obj, 'y', 0)),
                'width': float(getattr(obj, 'width', 0)),
                'height': float(getattr(obj, 'height', 0))
            }
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle other objects with __dict__
        if hasattr(obj, '__dict__'):
            return {k: self.default(v) for k, v in obj.__dict__.items()}
        
        # Fallback to default JSON encoder
        return super().default(obj)


def json_dump_safe(obj: Any, fp: Any, **kwargs) -> None:
    """
    Safely serialize object to JSON file, handling Pydantic models and other non-serializable objects.
    
    Args:
        obj: Object to serialize
        fp: File-like object to write to
        **kwargs: Additional arguments for json.dump
    """
    json.dump(obj, fp, cls=PydanticJSONEncoder, ensure_ascii=False, **kwargs)


def json_dumps_safe(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string, handling Pydantic models and other non-serializable objects.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string representation of the object
    """
    return json.dumps(obj, cls=PydanticJSONEncoder, ensure_ascii=False, **kwargs)

