#!/usr/bin/env python3
"""
Test-Run Script for Simple PID + Uni-Bild 3.

Dieses Skript:
1. Führt Analyse auf Simple PID durch
2. Führt Analyse auf Uni-Bild 3 durch
3. Validiert Ergebnisse
4. Speichert umfassende Reports
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.evaluation.kpi_calculator import KPICalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_analysis(image_path: Path, truth_path: Path = None):
    """Run analysis on a single image."""
    logger.info(f"=== Analyzing: {image_path.name} ===")
    
    try:
        # Initialize services
        config_service = ConfigService()
        config = config_service.get_raw_config()
        
        # Get GCP Project ID and location
        gcp_project_id = os.getenv("GCP_PROJECT_ID")
        gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        if not gcp_project_id:
            logger.error("GCP_PROJECT_ID environment variable not set. Please set it in your .env file.")
            return None
        
        # Get config as dict
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = config_service.get_raw_config()
        
        # Initialize LLM Client (korrekte Signatur: project_id, default_location, config)
        llm_client = LLMClient(
            project_id=gcp_project_id,
            default_location=gcp_location,
            config=config_dict
        )
        
        # Initialize Knowledge Manager (korrekte Signatur)
        learning_db_path = Path(config.get('paths', {}).get('learning_db', 'learning_db.json'))
        element_type_list_path = Path(config.get('paths', {}).get('element_type_list', 'element_type_list.json'))
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_handler=llm_client,
            config=config
        )
        
        # Dummy progress callback
        class DummyProgressCallback:
            def update_progress(self, progress: float, message: str):
                logger.info(f"Progress: {progress:.1f}% - {message}")
            def update_status_label(self, message: str):
                logger.info(f"Status: {message}")
            def log_message(self, message: str, level: str = 'INFO'):
                if level == 'INFO':
                    logger.info(message)
                elif level == 'WARNING':
                    logger.warning(message)
                elif level == 'ERROR':
                    logger.error(message)
            def report_truth_mode(self, active: bool):
                logger.info(f"Truth Mode: {active}")
            def report_correction(self, correction_text: str):
                logger.info(f"Correction: {correction_text}")
        
        progress_callback = DummyProgressCallback()
        
        # Use default model strategy
        model_strategy_name = config.get('model_strategies', {}).get('default_strategy', 'mixed_speed_accuracy')
        model_strategy = config.get('model_strategies', {}).get(model_strategy_name, {})
        
        # Initialize Pipeline Coordinator (korrekte Signatur)
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service,
            model_strategy=model_strategy,
            progress_callback=progress_callback
        )
        
        # Load truth data if available
        truth_data = None
        if truth_path and truth_path.exists():
            logger.info(f"Loading truth data from: {truth_path}")
            import json
            with open(truth_path, 'r', encoding='utf-8') as f:
                truth_data = json.load(f)
        
        # Run analysis
        logger.info(f"Starting analysis for: {image_path.name}")
        result = coordinator.process(
            image_path=str(image_path),
            output_dir=None,  # Let coordinator decide output dir
            params_override={
                'truth_data': truth_data,
                'iteration': 1
            }
        )
        
        if result:
            logger.info(f"Analysis complete for {image_path.name}")
            logger.info(f"  - Output Directory: {result.output_dir}")
            logger.info(f"  - Quality Score: {result.quality_score:.2f}%")
            
            if result.final_ai_data:
                logger.info(f"  - Elements: {len(result.final_ai_data.get('elements', []))}")
                logger.info(f"  - Connections: {len(result.final_ai_data.get('connections', []))}")
                
                # Calculate KPIs if truth data available
                if truth_data:
                    kpi_calculator = KPICalculator()
                    kpis = kpi_calculator.calculate_kpis(
                        analysis_result=result.final_ai_data,
                        truth_data=truth_data
                    )
                    logger.info(f"  - Element Precision: {kpis.get('element_precision', 0.0):.3f}")
                    logger.info(f"  - Element Recall: {kpis.get('element_recall', 0.0):.3f}")
                    logger.info(f"  - Element F1: {kpis.get('element_f1', 0.0):.3f}")
                    logger.info(f"  - Type Accuracy: {kpis.get('type_accuracy', 0.0):.3f}")
        else:
            logger.error(f"Analysis failed for {image_path.name}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {image_path.name}: {e}", exc_info=True)
        return None


def main():
    """Main test function."""
    logger.info("=== Starting Test Run: Simple PID + Uni-Bild 3 ===")
    
    # Define paths
    project_root = Path(__file__).parent.parent
    
    # Search in multiple possible locations
    search_dirs = [
        project_root / "training_data" / "uni_bilder",
        project_root / "training_data",
        project_root / "test_images",
        project_root
    ]
    
    # Find test images
    simple_pid_path = None
    uni_bild_3_path = None
    
    # Look for Simple PID
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in ["simple*.png", "simple*.jpg", "Simple*.png", "Simple*.jpg", "*simple*.png"]:
            matches = list(search_dir.glob(pattern))
            if matches:
                simple_pid_path = matches[0]
                logger.info(f"Found Simple PID: {simple_pid_path}")
                break
        if simple_pid_path:
            break
    
    # Look for Uni-Bild 3
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in ["*uni*3*.png", "*uni*3*.jpg", "*Uni*3*.png", "*Uni*3*.jpg", "*3*.png"]:
            matches = list(search_dir.glob(pattern))
            if matches:
                # Filter for Uni-Bild 3 specifically
                for match in matches:
                    name_lower = match.name.lower()
                    if ("3" in name_lower or "drei" in name_lower) and ("uni" in name_lower or "bild" in name_lower):
                        uni_bild_3_path = match
                        logger.info(f"Found Uni-Bild 3: {uni_bild_3_path}")
                        break
                if uni_bild_3_path:
                    break
        if uni_bild_3_path:
            break
    
    # If still not found, try more specific patterns
    if not simple_pid_path:
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for file in search_dir.iterdir():
                if file.is_file() and file.suffix.lower() in ['.png', '.jpg']:
                    if 'simple' in file.name.lower() or 'pid' in file.name.lower():
                        simple_pid_path = file
                        logger.info(f"Found Simple PID (alternative): {simple_pid_path}")
                        break
            if simple_pid_path:
                break
    
    if not uni_bild_3_path:
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for file in search_dir.iterdir():
                if file.is_file() and file.suffix.lower() in ['.png', '.jpg']:
                    name_lower = file.name.lower()
                    if ('uni' in name_lower or 'bild' in name_lower) and ('3' in name_lower or 'drei' in name_lower):
                        uni_bild_3_path = file
                        logger.info(f"Found Uni-Bild 3 (alternative): {uni_bild_3_path}")
                        break
            if uni_bild_3_path:
                break
    
    # Run tests
    results = {}
    
    # Test 1: Simple PID - use "Einfaches P&I" if found
    if not simple_pid_path:
        # Try "Einfaches P&I"
        einfaches_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
        if einfaches_path.exists():
            simple_pid_path = einfaches_path
            logger.info(f"Using 'Einfaches P&I': {simple_pid_path}")
    
    if simple_pid_path and simple_pid_path.exists():
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST 1: Simple PID ({simple_pid_path.name})")
        logger.info(f"{'='*60}")
        truth_path = simple_pid_path.parent / f"{simple_pid_path.stem}_truth.json"
        if not truth_path.exists():
            truth_path = None
        results['simple_pid'] = run_analysis(simple_pid_path, truth_path)
    else:
        logger.warning("Simple PID not found. Skipping...")
    
    # Test 2: Uni-Bild - use "Verfahrensfließbild_Uni" if Uni-Bild 3 not found
    if not uni_bild_3_path:
        # Try "Verfahrensfließbild_Uni"
        uni_path = project_root / "training_data" / "complex_pids" / "Verfahrensfließbild_Uni.png"
        if uni_path.exists():
            uni_bild_3_path = uni_path
            logger.info(f"Using 'Verfahrensfließbild_Uni': {uni_bild_3_path}")
    
    if uni_bild_3_path and uni_bild_3_path.exists():
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST 2: Uni-Bild ({uni_bild_3_path.name})")
        logger.info(f"{'='*60}")
        truth_path = uni_bild_3_path.parent / f"{uni_bild_3_path.stem}_truth.json"
        if not truth_path.exists():
            truth_path = None
        results['uni_bild'] = run_analysis(uni_bild_3_path, truth_path)
    else:
        logger.warning("Uni-Bild not found. Skipping...")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    for test_name, result in results.items():
        if result:
            elements = len(result.final_ai_data.get('elements', [])) if result.final_ai_data else 0
            connections = len(result.final_ai_data.get('connections', [])) if result.final_ai_data else 0
            quality = result.quality_score if hasattr(result, 'quality_score') else 0.0
            logger.info(f"{test_name}: ✓ {elements} elements, {connections} connections, Quality: {quality:.2f}%")
        else:
            logger.info(f"{test_name}: ✗ Failed")
    
    logger.info("=== Test Run Complete ===")


if __name__ == "__main__":
    main()

