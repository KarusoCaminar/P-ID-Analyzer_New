"""
Pydantic models for P&ID elements and connections.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class BBox(BaseModel):
    """Bounding box in normalized coordinates (0.0-1.0)."""
    x: float = Field(ge=0.0, le=1.0, description="Normalized x coordinate")
    y: float = Field(ge=0.0, le=1.0, description="Normalized y coordinate")
    width: float = Field(gt=0.0, le=1.0, description="Normalized width")
    height: float = Field(gt=0.0, le=1.0, description="Normalized height")
    
    @field_validator('x', 'y', 'width', 'height')
    @classmethod
    def validate_coordinates(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Coordinates must be non-negative")
        return v


class PixelBBox(BaseModel):
    """Bounding box in pixel coordinates."""
    x: int = Field(ge=0, description="Pixel x coordinate")
    y: int = Field(ge=0, description="Pixel y coordinate")
    width: int = Field(gt=0, description="Pixel width")
    height: int = Field(gt=0, description="Pixel height")


class Port(BaseModel):
    """Port/connection point of an element."""
    id: str = Field(description="Unique port identifier")
    name: str = Field(description="Port name (e.g., 'in', 'out')")
    bbox: BBox = Field(description="Port bounding box")
    port_type: Optional[str] = Field(None, description="Port type (e.g., 'fluid', 'electrical')")


class ElementType(str, Enum):
    """Common P&ID element types."""
    PUMP = "Pump"
    VALVE = "Valve"
    HEAT_EXCHANGER = "Heat Exchanger"
    BOILER = "Boiler"
    REACTOR = "Reactor"
    TANK = "Tank"
    SENSOR = "Sensor"
    CONTROLLER = "Controller"
    LINE_SPLIT = "Line_Split"
    LINE_MERGE = "Line_Merge"
    DIAGRAM_INLET = "Diagram_Inlet"
    DIAGRAM_OUTLET = "Diagram_Outlet"


class Element(BaseModel):
    """A single P&ID element (component)."""
    id: str = Field(description="Unique element identifier")
    label: str = Field(description="Element label/text")
    type: str = Field(description="Element type")
    bbox: BBox = Field(description="Element bounding box")
    ports: List[Port] = Field(default_factory=list, description="Element ports")
    
    # CRITICAL FIX: Confidence is not optional and has a PENALTY default
    # If no confidence is provided by LLM, it defaults to 0.1 (PENALTY)
    # This ensures fusion_engine.py never receives None values
    confidence: float = Field(
        default=0.1,  # PENALTY: If no confidence is provided, it's 0.1 (not 0.0)
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0). Default 0.1 if missing (PENALTY for missing confidence)"
    )
    
    system_group: Optional[str] = Field(None, description="System group identifier")
    tile_coords: Optional[tuple[int, int, int, int]] = Field(None, description="Tile coordinates for swarm analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('type')
    @classmethod
    def validate_element_type(cls, v: str) -> str:
        """Validate that element type is not empty."""
        if not v:
            raise ValueError("Element type must be a non-empty string")
        return v


class ConnectionKind(str, Enum):
    """Connection types."""
    PROCESS = "process"
    CONTROL = "control"
    INSTRUMENT = "instrument"
    ELECTRICAL = "electrical"


class Connection(BaseModel):
    """Connection between two elements."""
    id: Optional[str] = Field(None, description="Unique connection identifier")
    from_id: str = Field(description="Source element ID")
    to_id: str = Field(description="Target element ID")
    from_port_id: Optional[str] = Field(None, description="Source port ID")
    to_port_id: Optional[str] = Field(None, description="Target port ID")
    from_port_type: Optional[str] = Field(None, description="Source port type")
    to_port_type: Optional[str] = Field(None, description="Target port type")
    line_semantic_type: Optional[str] = Field(None, description="Line semantic type")
    kind: ConnectionKind = Field(default=ConnectionKind.PROCESS, description="Connection kind")
    color: Optional[str] = Field(None, description="Line color")
    style: Optional[str] = Field(None, description="Line style (solid, dashed, etc.)")
    status: Optional[str] = Field(None, description="Line status")
    predicted: Optional[bool] = Field(None, description="Whether connection was predicted (gap-filled)")
    hops: int = Field(default=1, ge=1, description="Number of hops in connection path")
    raw_line_data: Optional[Dict[str, Any]] = Field(None, description="Raw line extraction data")
    
    # CRITICAL FIX: Confidence is not optional and has a PENALTY default
    # If no confidence is provided by LLM, it defaults to 0.1 (PENALTY)
    # This ensures fusion_engine.py never receives None values
    confidence: float = Field(
        default=0.1,  # PENALTY: If no confidence is provided, it's 0.1 (not 0.0)
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0). Default 0.1 if missing (PENALTY for missing confidence)"
    )
    
    @field_validator('from_id', 'to_id')
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate that element IDs are not empty."""
        if not v:
            raise ValueError("Element IDs must be non-empty strings")
        return v

