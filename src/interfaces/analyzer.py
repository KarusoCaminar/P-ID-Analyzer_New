"""
Interface for analysis components (swarm, monolith, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.analyzer.models.elements import Element, Connection


class IAnalyzer(ABC):
    """Interface for analysis components."""
    
    @abstractmethod
    def analyze(
        self,
        image_path: str,
        output_dir: Optional[Path] = None,
        excluded_zones: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image and return detected elements and connections.
        
        Args:
            image_path: Path to the image
            output_dir: Optional output directory
            excluded_zones: Zones to exclude from analysis
            
        Returns:
            Dictionary with 'elements' and 'connections' lists
        """
        pass

