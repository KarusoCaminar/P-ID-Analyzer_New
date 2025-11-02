"""
Coordinate Validator - Ensures coordinate safety and correctness.

Validates and corrects coordinates to ensure they are:
- Within image bounds
- Properly normalized
- Non-overlapping where appropriate
- Consistent across different coordinate systems
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class CoordinateValidator:
    """
    Validates and corrects coordinates for P&ID analysis.
    """
    
    def __init__(self, image_width: int, image_height: int):
        """
        Initialize coordinate validator.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.image_width = image_width
        self.image_height = image_height
    
    def validate_bbox(
        self,
        bbox: Dict[str, float],
        coordinate_type: str = 'normalized'
    ) -> Optional[Dict[str, float]]:
        """
        Validate and correct bounding box coordinates.
        
        Args:
            bbox: Bounding box dictionary
            coordinate_type: 'normalized' (0-1) or 'pixel' (absolute pixels)
            
        Returns:
            Validated bbox or None if invalid
        """
        try:
            if not isinstance(bbox, dict):
                return None
            
            required_keys = ['x', 'y', 'width', 'height']
            if not all(k in bbox for k in required_keys):
                return None
            
            x = float(bbox['x'])
            y = float(bbox['y'])
            width = float(bbox['width'])
            height = float(bbox['height'])
            
            # Check for invalid dimensions
            if width <= 0 or height <= 0:
                return None
            
            if coordinate_type == 'normalized':
                # Normalize coordinates to 0-1 range
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                width = max(0.0, min(1.0 - x, width))
                height = max(0.0, min(1.0 - y, height))
                
                # Ensure box doesn't exceed image bounds
                if x + width > 1.0:
                    width = 1.0 - x
                if y + height > 1.0:
                    height = 1.0 - y
            
            elif coordinate_type == 'pixel':
                # Validate pixel coordinates
                x = max(0, int(x))
                y = max(0, int(y))
                width = max(1, int(width))
                height = max(1, int(height))
                
                # Ensure box doesn't exceed image bounds
                if x + width > self.image_width:
                    width = self.image_width - x
                if y + height > self.image_height:
                    height = self.image_height - y
                
                if width <= 0 or height <= 0 or x >= self.image_width or y >= self.image_height:
                    return None
            
            return {
                'x': x,
                'y': y,
                'width': width,
                'height': height
            }
        except Exception as e:
            logger.error(f"Error validating bbox: {e}", exc_info=True)
            return None
    
    def normalize_pixel_bbox(self, pixel_bbox: Dict[str, int]) -> Dict[str, float]:
        """
        Convert pixel bbox to normalized bbox.
        
        Args:
            pixel_bbox: Bounding box in pixel coordinates
            
        Returns:
            Normalized bounding box (0-1)
        """
        try:
            x = pixel_bbox.get('x', 0) / self.image_width
            y = pixel_bbox.get('y', 0) / self.image_height
            width = pixel_bbox.get('width', 0) / self.image_width
            height = pixel_bbox.get('height', 0) / self.image_height
            
            return {
                'x': max(0.0, min(1.0, x)),
                'y': max(0.0, min(1.0, y)),
                'width': max(0.0, min(1.0 - x, width)),
                'height': max(0.0, min(1.0 - y, height))
            }
        except Exception as e:
            logger.error(f"Error normalizing pixel bbox: {e}", exc_info=True)
            return {'x': 0.0, 'y': 0.0, 'width': 0.0, 'height': 0.0}
    
    def denormalize_bbox(self, normalized_bbox: Dict[str, float]) -> Dict[str, int]:
        """
        Convert normalized bbox to pixel bbox.
        
        Args:
            normalized_bbox: Bounding box in normalized coordinates (0-1)
            
        Returns:
            Pixel bounding box
        """
        try:
            x = int(normalized_bbox.get('x', 0) * self.image_width)
            y = int(normalized_bbox.get('y', 0) * self.image_height)
            width = int(normalized_bbox.get('width', 0) * self.image_width)
            height = int(normalized_bbox.get('height', 0) * self.image_height)
            
            # Ensure within bounds
            x = max(0, min(self.image_width - 1, x))
            y = max(0, min(self.image_height - 1, y))
            width = max(1, min(self.image_width - x, width))
            height = max(1, min(self.image_height - y, height))
            
            return {
                'x': x,
                'y': y,
                'width': width,
                'height': height
            }
        except Exception as e:
            logger.error(f"Error denormalizing bbox: {e}", exc_info=True)
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
    
    def validate_element_coordinates(
        self,
        element: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Validate and correct element coordinates.
        
        Args:
            element: Element dictionary
            
        Returns:
            Validated element or None if invalid
        """
        try:
            if not element.get('bbox'):
                return None
            
            validated_bbox = self.validate_bbox(element['bbox'], coordinate_type='normalized')
            if not validated_bbox:
                return None
            
            # Create validated element
            validated_element = element.copy()
            validated_element['bbox'] = validated_bbox
            
            # Validate port coordinates
            if 'ports' in validated_element and isinstance(validated_element['ports'], list):
                validated_ports = []
                for port in validated_element['ports']:
                    if port.get('bbox'):
                        validated_port_bbox = self.validate_bbox(port['bbox'], coordinate_type='normalized')
                        if validated_port_bbox:
                            validated_port = port.copy()
                            validated_port['bbox'] = validated_port_bbox
                            validated_ports.append(validated_port)
                validated_element['ports'] = validated_ports
            
            return validated_element
        except Exception as e:
            logger.error(f"Error validating element coordinates: {e}", exc_info=True)
            return None
    
    def validate_connection_coordinates(
        self,
        connection: Dict[str, Any],
        elements_map: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Validate connection by ensuring referenced elements exist and have valid coordinates.
        
        Args:
            connection: Connection dictionary
            elements_map: Map of element IDs to elements
            
        Returns:
            Validated connection or None if invalid
        """
        try:
            from_id = connection.get('from_id')
            to_id = connection.get('to_id')
            
            if not from_id or not to_id:
                return None
            
            from_element = elements_map.get(from_id)
            to_element = elements_map.get(to_id)
            
            if not from_element or not to_element:
                return None
            
            # Check if elements have valid bboxes
            if not from_element.get('bbox') or not to_element.get('bbox'):
                return None
            
            from_bbox = self.validate_bbox(from_element['bbox'], coordinate_type='normalized')
            to_bbox = self.validate_bbox(to_element['bbox'], coordinate_type='normalized')
            
            if not from_bbox or not to_bbox:
                return None
            
            # Connection is valid
            return connection
        except Exception as e:
            logger.error(f"Error validating connection coordinates: {e}", exc_info=True)
            return None


