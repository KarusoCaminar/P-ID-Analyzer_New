"""
Interface for the main pipeline processor.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from src.analyzer.models.pipeline import AnalysisResult, PipelineState


class IProcessor(ABC):
    """Interface for the main analysis processor."""
    
    @abstractmethod
    def process(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        params_override: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Process a P&ID image and return analysis results.
        
        Args:
            image_path: Path to the P&ID image
            output_dir: Optional output directory for artifacts
            params_override: Optional parameter overrides
            
        Returns:
            AnalysisResult with detected elements and connections
        """
        pass
    
    @abstractmethod
    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        pass
    
    @abstractmethod
    def pretrain_symbols(
        self,
        pretraining_path: Path,
        model_info: Dict[str, Any]
    ) -> list[Dict]:
        """
        Train symbol library from pretraining images.
        
        Args:
            pretraining_path: Path to directory with symbol images
            model_info: Model configuration for symbol extraction
            
        Returns:
            List of training reports
        """
        pass

