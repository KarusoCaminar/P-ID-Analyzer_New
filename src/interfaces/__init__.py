"""
Interfaces for analyzer components.
"""

from .processor import IProcessor
from .analyzer import IAnalyzer
from .exporter import IExporter

__all__ = ["IProcessor", "IAnalyzer", "IExporter"]

