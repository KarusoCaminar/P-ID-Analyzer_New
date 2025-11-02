"""
Pipeline state and result models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from .elements import Element, Connection


class PipelineState(BaseModel):
    """Current state of the analysis pipeline."""
    image_path: str
    current_phase: str = Field(default="initialization")
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    elements: List[Element] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    excluded_zones: List[Dict[str, float]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    legend_data: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True


class AnalysisResult(BaseModel):
    """Final analysis result."""
    image_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    elements: List[Element] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    score_history: List[float] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    legend_data: Optional[Dict[str, Any]] = None
    cgm_data: Optional[Dict[str, Any]] = None
    kpis: Optional[Dict[str, Any]] = None
    uncertain_zones: List[Dict[str, Any]] = Field(default_factory=list)
    corrections_applied: int = Field(default=0, ge=0)

