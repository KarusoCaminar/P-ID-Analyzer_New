"""
Unit tests for KPICalculator.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from src.analyzer.evaluation.kpi_calculator import KPICalculator


@pytest.fixture
def kpi_calculator():
    """Create KPICalculator instance."""
    return KPICalculator(confidence_calibration_offset=0.0)


@pytest.fixture
def mock_analysis_data():
    """Create mock analysis data."""
    return {
        "elements": [
            {
                "id": "P-101",
                "type": "Pump",
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.04},
                "confidence": 0.95
            },
            {
                "id": "V-101",
                "type": "Valve",
                "bbox": {"x": 0.3, "y": 0.4, "width": 0.02, "height": 0.03},
                "confidence": 0.90
            }
        ],
        "connections": [
            {
                "from_id": "P-101",
                "to_id": "V-101",
                "confidence": 0.95
            }
        ]
    }


@pytest.fixture
def mock_truth_data():
    """Create mock truth data."""
    return {
        "elements": [
            {
                "id": "P-101",
                "type": "Pump",
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.04}
            },
            {
                "id": "V-101",
                "type": "Valve",
                "bbox": {"x": 0.3, "y": 0.4, "width": 0.02, "height": 0.03}
            },
            {
                "id": "FT-10",
                "type": "Volume Flow Sensor",
                "bbox": {"x": 0.5, "y": 0.6, "width": 0.03, "height": 0.03}
            }
        ],
        "connections": [
            {
                "from_id": "P-101",
                "to_id": "V-101"
            },
            {
                "from_id": "V-101",
                "to_id": "FT-10"
            }
        ]
    }


class TestKPICalculator:
    """Tests for KPICalculator."""
    
    def test_kpi_calculator_initialization(self, kpi_calculator):
        """Test KPICalculator initialization."""
        assert kpi_calculator.confidence_calibration_offset == 0.0
    
    def test_calculate_comprehensive_kpis_structure(self, kpi_calculator, mock_analysis_data):
        """Test that calculate_comprehensive_kpis returns correct structure."""
        kpis = kpi_calculator.calculate_comprehensive_kpis(mock_analysis_data, None)
        
        assert kpis is not None
        assert "total_elements" in kpis
        assert "total_connections" in kpis
        assert "unique_element_types" in kpis
    
    def test_calculate_comprehensive_kpis_with_truth(self, kpi_calculator, mock_analysis_data, mock_truth_data):
        """Test calculate_comprehensive_kpis with truth data."""
        kpis = kpi_calculator.calculate_comprehensive_kpis(mock_analysis_data, mock_truth_data)
        
        assert kpis is not None
        assert "element_precision" in kpis
        assert "element_recall" in kpis
        assert "element_f1" in kpis
        assert "connection_precision" in kpis
        assert "connection_recall" in kpis
        assert "connection_f1" in kpis
        assert "quality_score" in kpis
    
    def test_calculate_structural_kpis(self, kpi_calculator, mock_analysis_data):
        """Test structural KPI calculation."""
        kpis = kpi_calculator._calculate_structural_kpis(mock_analysis_data)
        
        assert kpis["total_elements"] == 2
        assert kpis["total_connections"] == 1
        assert kpis["unique_element_types"] == 2
    
    def test_calculate_confidence_metrics(self, kpi_calculator, mock_analysis_data):
        """Test confidence metrics calculation."""
        kpis = kpi_calculator._calculate_confidence_metrics(mock_analysis_data)
        
        assert "avg_element_confidence" in kpis
        assert "avg_connection_confidence" in kpis
        assert kpis["avg_element_confidence"] > 0.0
    
    def test_calculate_quality_metrics(self, kpi_calculator, mock_analysis_data, mock_truth_data):
        """Test quality metrics calculation."""
        kpis = kpi_calculator._calculate_quality_metrics(mock_analysis_data, mock_truth_data)
        
        assert "element_precision" in kpis
        assert "element_recall" in kpis
        assert "element_f1" in kpis
        assert "connection_precision" in kpis
        assert "connection_recall" in kpis
        assert "connection_f1" in kpis
        assert "quality_score" in kpis
        
        # Quality score should be between 0 and 100
        assert 0.0 <= kpis["quality_score"] <= 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

