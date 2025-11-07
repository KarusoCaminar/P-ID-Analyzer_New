"""
Output Structure Manager - Gold Standard für strukturierte Output-Ordner.

Dieser Manager erzwingt eine konsistente Ordnerstruktur für alle Test-Läufe,
unabhängig davon, ob sie über GUI, CLI-Runner oder Testskripte gestartet werden.

STRUKTUR:
outputs/
  live_test/  (oder andere Test-Typen)
    YYYYMMDD_HHMMSS/
      logs/
        test.log
        analysis.log
        errors.log
      visualizations/
        score_curve.png
        confidence_map.png
        debug_map.png
        kpi_dashboard.png
        debug_map_iteration_*.png
      data/
        test_result.json
        results.json
        cgm_data.json
        cgm_network_generated.py
        kpis.json
        legend_info.json
        output_phase_*.json
      artifacts/
        config_snapshot.yaml
        prompts_snapshot.json
        test_metadata.md
        report.html
      temp/
        temp_quadrants/
        temp_polylines/
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputStructureManager:
    """
    Gold Standard Output Structure Manager.
    
    Verwaltet und erzwingt eine konsistente Ordnerstruktur für alle Test-Läufe.
    """
    
    # Ordnerstruktur-Definition
    SUBDIRS = {
        'logs': 'logs',
        'visualizations': 'visualizations',
        'data': 'data',
        'artifacts': 'artifacts',
        'temp': 'temp'
    }
    
    def __init__(self, base_output_dir: Path, test_type: str = "live_test"):
        """
        Initialize Output Structure Manager.
        
        Args:
            base_output_dir: Base output directory (e.g., project_root / "outputs")
            test_type: Type of test (e.g., "live_test", "iterative_tests", "phase1_tests")
        """
        self.base_output_dir = Path(base_output_dir)
        self.test_type = test_type
        
        # Create timestamp-based subdirectory
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_dir = self.base_output_dir / test_type / timestamp_str
        
        # Initialize subdirectories
        self._initialize_structure()
        
        logger.info(f"Output Structure Manager initialized: {self.run_output_dir}")
    
    def _initialize_structure(self) -> None:
        """Create all required subdirectories."""
        for subdir_name in self.SUBDIRS.values():
            subdir_path = self.run_output_dir / subdir_name
            subdir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created subdirectory: {subdir_path}")
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self.run_output_dir / self.SUBDIRS['logs']
    
    @property
    def visualizations_dir(self) -> Path:
        """Get visualizations directory."""
        return self.run_output_dir / self.SUBDIRS['visualizations']
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.run_output_dir / self.SUBDIRS['data']
    
    @property
    def artifacts_dir(self) -> Path:
        """Get artifacts directory."""
        return self.run_output_dir / self.SUBDIRS['artifacts']
    
    @property
    def temp_dir(self) -> Path:
        """Get temp directory."""
        return self.run_output_dir / self.SUBDIRS['temp']
    
    def get_log_file(self, log_name: str = "test.log") -> Path:
        """
        Get path to log file.
        
        Args:
            log_name: Name of log file (default: "test.log")
            
        Returns:
            Path to log file
        """
        return self.logs_dir / log_name
    
    def get_visualization_path(self, filename: str) -> Path:
        """
        Get path to visualization file.
        
        Args:
            filename: Name of visualization file (e.g., "score_curve.png")
            
        Returns:
            Path to visualization file
        """
        return self.visualizations_dir / filename
    
    def get_data_path(self, filename: str) -> Path:
        """
        Get path to data file.
        
        Args:
            filename: Name of data file (e.g., "test_result.json")
            
        Returns:
            Path to data file
        """
        return self.data_dir / filename
    
    def get_artifact_path(self, filename: str) -> Path:
        """
        Get path to artifact file.
        
        Args:
            filename: Name of artifact file (e.g., "config_snapshot.yaml")
            
        Returns:
            Path to artifact file
        """
        return self.artifacts_dir / filename
    
    def get_temp_path(self, subdir: Optional[str] = None) -> Path:
        """
        Get path to temp directory or subdirectory.
        
        Args:
            subdir: Optional subdirectory name (e.g., "temp_quadrants")
            
        Returns:
            Path to temp directory or subdirectory
        """
        if subdir:
            temp_subdir = self.temp_dir / subdir
            temp_subdir.mkdir(parents=True, exist_ok=True)
            return temp_subdir
        return self.temp_dir
    
    def get_output_dir(self) -> Path:
        """Get main output directory for this run."""
        return self.run_output_dir
    
    def get_output_dir_str(self) -> str:
        """Get main output directory as string."""
        return str(self.run_output_dir)
    
    def cleanup_temp(self) -> None:
        """Clean up temporary files (optional, can be called after test completion)."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {e}")
    
    def create_readme(self) -> None:
        """Create README.md file explaining the folder structure."""
        readme_path = self.run_output_dir / "README.md"
        
        readme_content = f"""# Test Run Output Structure

This directory contains the output of a test run from {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

## Folder Structure

### `logs/`
Contains all log files:
- `test.log`: Main test log file
- `analysis.log`: Analysis-specific logs (if any)
- `errors.log`: Error logs (if any)

### `visualizations/`
Contains all visualization files:
- `score_curve.png`: Quality score improvement over iterations
- `confidence_map.png`: Confidence visualization for elements
- `debug_map.png`: Debug map with elements and connections
- `kpi_dashboard.png`: KPI dashboard visualization
- `debug_map_iteration_*.png`: Debug maps for each iteration

### `data/`
Contains all data files:
- `test_result.json`: Complete test result with KPIs
- `results.json`: Analysis results
- `cgm_data.json`: CGM (Component Grouping Model) data in JSON format
- `cgm_network_generated.py`: CGM network in Python dataclass format
- `kpis.json`: Key Performance Indicators
- `legend_info.json`: Legend information
- `output_phase_*.json`: Intermediate results from each phase

### `artifacts/`
Contains configuration and metadata:
- `config_snapshot.yaml`: Configuration snapshot at test time
- `prompts_snapshot.json`: Prompt templates snapshot
- `test_metadata.md`: Test metadata and description
- `report.html`: HTML report

### `temp/`
Contains temporary files (can be safely deleted):
- `temp_quadrants/`: Temporary quadrant images
- `temp_polylines/`: Temporary polyline extraction files

## Notes

- All file paths in this structure are managed by `OutputStructureManager`
- The structure is enforced across all test runners (GUI, CLI, scripts)
- Future test scripts will automatically use this structure
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"Created README.md: {readme_path}")


def create_output_structure(base_output_dir: Path, test_type: str = "live_test") -> OutputStructureManager:
    """
    Factory function to create Output Structure Manager.
    
    Args:
        base_output_dir: Base output directory
        test_type: Type of test
        
    Returns:
        OutputStructureManager instance
    """
    return OutputStructureManager(base_output_dir, test_type)


def ensure_output_structure(output_dir: Path) -> None:
    """
    Ensure structured output directories exist in given output directory.
    
    This function can be called from any code to enforce the structure,
    even if OutputStructureManager is not used.
    
    Args:
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    for subdir in ['logs', 'visualizations', 'data', 'artifacts', 'temp']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Ensured output structure in: {output_path}")

