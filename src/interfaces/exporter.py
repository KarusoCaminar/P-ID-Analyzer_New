"""
Interface for result exporters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

from src.analyzer.models.pipeline import AnalysisResult


class IExporter(ABC):
    """Interface for exporting analysis results."""
    
    @abstractmethod
    def export_json(
        self,
        result: AnalysisResult,
        output_path: Path
    ) -> None:
        """Export results as JSON."""
        pass
    
    @abstractmethod
    def export_cgm(
        self,
        result: AnalysisResult,
        output_path: Path
    ) -> Dict[str, Any]:
        """Export CGM abstraction."""
        pass
    
    @abstractmethod
    def export_visualization(
        self,
        result: AnalysisResult,
        image_path: str,
        output_path: Path
    ) -> None:
        """Export visualization/debug images."""
        pass

