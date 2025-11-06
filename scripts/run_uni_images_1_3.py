#!/usr/bin/env python3
"""
Test-Run Script for Uni-Bilder 1-3.

This script:
1. Runs analysis on Uni-Bild 1, 2, and 3
2. Validates results
3. Saves comprehensive reports
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


def run_analysis(image_path: Path, truth_path: Optional[Path] = None):
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
        
        # Initialize Knowledge Manager
        learning_db_path = Path(config.get('paths', {}).get('learning_db', 'learning_db.json'))
        element_type_list_path = Path(config.get('paths', {}).get('element_type_list', 'element_type_list.json'))
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_handler=llm_client,
            config=config
        )
        
        # Load truth data if available
        truth_data = None
        if truth_path and truth_path.exists():
            import json
            with open(truth_path, 'r', encoding='utf-8') as f:
                truth_data = json.load(f)
            logger.info(f"Loaded truth data from: {truth_path}")
        
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
                elif level == 'SUCCESS':
                    logger.info(f"SUCCESS: {message}")
                elif level == 'PHASE':
                    logger.info(f"--- {message} ---")
        
        progress_callback = DummyProgressCallback()
        
        # Use default model strategy
        model_strategy_name = config.get('model_strategies', {}).get('default_strategy', 'mixed_speed_accuracy')
        model_strategy = config.get('model_strategies', {}).get(model_strategy_name, {})
        
        # Initialize Pipeline Coordinator
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service,
            model_strategy=model_strategy,
            progress_callback=progress_callback
        )
        
        # Run analysis
        result = coordinator.process(
            image_path=str(image_path),
            output_dir=None,  # Let coordinator decide output dir
            params_override={
                'truth_data': truth_data,
                'iteration': 1
            }
        )
        
        if result:
            logger.info(f"Analysis complete: {result.output_dir}")
            logger.info(f"Quality Score: {result.quality_score:.2f}%")
            return result
        else:
            logger.error("Analysis failed")
            return None
            
    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {e}", exc_info=True)
        return None


def main():
    """Main function to run analyses on Uni-Bilder 1-3."""
    logger.info("=== Starting Uni-Bilder 1-3 Analysis ===")
    
    try:
        # Define image paths
        base_path = Path("training_data")
        
        uni_images = [
            ("Uni-Bild 1", base_path / "uni_bilder" / "Uni-Bild 1.png"),
            ("Uni-Bild 2", base_path / "uni_bilder" / "Uni-Bild 2.png"),
            ("Uni-Bild 3", base_path / "uni_bilder" / "Uni-Bild 3.png"),
        ]
        
        # Alternative paths if above don't exist
        if not (base_path / "uni_bilder").exists():
            uni_images = [
                ("Uni-Bild 1", base_path / "Uni-Bild 1.png"),
                ("Uni-Bild 2", base_path / "Uni-Bild 2.png"),
                ("Uni-Bild 3", base_path / "Uni-Bild 3.png"),
            ]
        
        results = []
        
        for name, image_path in uni_images:
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Look for truth data
            truth_path = image_path.parent / f"{image_path.stem}_truth.json"
            if not truth_path.exists():
                truth_path = image_path.parent / f"{image_path.stem}_truth.json"
            
            # Run analysis
            result = run_analysis(image_path, truth_path if truth_path.exists() else None)
            
            if result:
                results.append({
                    'name': name,
                    'result': result,
                    'quality_score': result.quality_score
                })
                logger.info(f"{name}: Quality Score = {result.quality_score:.2f}%")
            else:
                logger.error(f"{name}: Analysis failed")
            
            logger.info("")  # Empty line between analyses
        
        # Summary
        logger.info("=== FINAL SUMMARY ===")
        for r in results:
            logger.info(f"{r['name']}: {r['quality_score']:.2f}%")
        
        if results:
            avg_score = sum(r['quality_score'] for r in results) / len(results)
            logger.info(f"Average Quality Score: {avg_score:.2f}%")
        
        logger.info("=== Uni-Bilder Analysis Complete ===")
        
    except Exception as e:
        logger.error(f"Error in Uni-Bilder analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

