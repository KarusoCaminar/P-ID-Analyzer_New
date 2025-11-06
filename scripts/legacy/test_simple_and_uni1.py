#!/usr/bin/env python3
"""
Test Script: Simple PID (with/without truth) + Uni-Bild 1

This script:
1. Tests Simple PID WITHOUT truth mode
2. Tests Simple PID WITH truth mode
3. Tests Uni-Bild 1 (in parallel if possible)
4. Evaluates results independently
"""

import os
import sys
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DummyProgressCallback:
    """Dummy progress callback for testing."""
    def update_progress(self, progress: int, message: str):
        logger.info(f"[Progress {progress}%] {message}")
    
    def update_status_label(self, message: str):
        logger.info(f"[Status] {message}")
    
    def log_message(self, message: str, level: str = 'INFO'):
        if level == 'ERROR':
            logger.error(message)
        elif level == 'WARNING':
            logger.warning(message)
        else:
            logger.info(message)
    
    def report_truth_mode(self, active: bool):
        logger.info(f"[Truth Mode] {'ACTIVE' if active else 'INACTIVE'}")
    
    def report_correction(self, correction_text: str):
        logger.debug(f"[Correction] {correction_text}")


def run_analysis(image_path: Path, output_dir: Path, truth_data: bool = False) -> dict:
    """Run analysis on a single image."""
    test_name = f"{image_path.stem}_{'truth' if truth_data else 'no_truth'}"
    logger.info(f"\n{'='*80}")
    logger.info(f"=== Starting Analysis: {test_name} ===")
    logger.info(f"{'='*80}\n")
    
    try:
        # Initialize services
        config_service = ConfigService()
        config = config_service.get_raw_config()
        
        # Get GCP Project ID
        gcp_project_id = os.getenv("GCP_PROJECT_ID")
        if not gcp_project_id:
            logger.error("GCP_PROJECT_ID environment variable not set. Please set it in your .env file.")
            return {"success": False, "error": "GCP_PROJECT_ID not set", "test_name": test_name}
        
        # Initialize LLM Client
        llm_client = LLMClient(
            project_id=gcp_project_id,
            default_location="us-central1",
            config=config
        )
        
        # Initialize Knowledge Manager
        learning_db_path = Path(config.get('paths', {}).get('learning_db', 'learning_db.json'))
        element_type_list_path = Path(config.get('paths', {}).get('element_type_list', 'element_type_list.json'))
        
        knowledge_manager = KnowledgeManager(
            element_type_list_path=element_type_list_path,
            learning_db_path=learning_db_path,
            llm_handler=llm_client,
            config=config
        )
        
        # Initialize Pipeline Coordinator
        progress_callback = DummyProgressCallback()
        model_strategy_name = config.get('model_strategies', {}).get('default_strategy', 'mixed_speed_accuracy')
        model_strategy = config.get('model_strategies', {}).get(model_strategy_name, {})
        
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service,
            model_strategy=model_strategy,
            progress_callback=progress_callback
        )
        
        # Load truth data if requested
        truth_data_dict = None
        if truth_data:
            truth_path = image_path.parent / f"{image_path.stem}_truth.json"
            if truth_path.exists():
                with open(truth_path, 'r', encoding='utf-8') as f:
                    truth_data_dict = json.load(f)
                logger.info(f"Loaded truth data from {truth_path}")
            else:
                logger.warning(f"Truth data file not found: {truth_path}")
        
        # Run analysis
        result = coordinator.process(
            str(image_path),
            str(output_dir),
            params_override={
                'truth_data': truth_data_dict,
                'iteration': 1
            }
        )
        
        if result:
            # Extract key metrics
            if hasattr(result, 'final_ai_data'):
                elements = result.final_ai_data.get('elements', []) if result.final_ai_data else []
                connections = result.final_ai_data.get('connections', []) if result.final_ai_data else []
                quality_score = result.quality_score if hasattr(result, 'quality_score') else 0.0
            else:
                elements = result.get('final_ai_data', {}).get('elements', [])
                connections = result.get('final_ai_data', {}).get('connections', [])
                quality_score = result.get('quality_score', 0.0)
            
            logger.info(f"\n✅ Analysis complete for {test_name}")
            logger.info(f"   Quality Score: {quality_score:.2f}%")
            logger.info(f"   Elements: {len(elements)}")
            logger.info(f"   Connections: {len(connections)}")
            
            return {
                "success": True,
                "test_name": test_name,
                "quality_score": quality_score,
                "elements_count": len(elements),
                "connections_count": len(connections),
                "result": result
            }
        else:
            logger.error(f"Analysis failed for {test_name}")
            return {"success": False, "error": "Analysis returned None", "test_name": test_name}
            
    except Exception as e:
        logger.error(f"Error analyzing {test_name}: {e}", exc_info=True)
        return {"success": False, "error": str(e), "test_name": test_name, "exception": type(e).__name__}


def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("TEST SUITE: Simple PID (with/without truth) + Uni-Bild 1")
    logger.info("="*80)
    
    project_root = Path(__file__).parent.parent
    
    # Define image paths
    simple_pid_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
    uni_bild_1_path = project_root / "training_data" / "complex_pids" / "page_1_original.png"
    
    # Check if files exist
    if not simple_pid_path.exists():
        logger.error(f"Simple PID not found: {simple_pid_path}")
        return
    
    if not uni_bild_1_path.exists():
        logger.error(f"Uni-Bild 1 not found: {uni_bild_1_path}")
        return
    
    # Create output directories
    output_base_dir = project_root / "outputs" / "test_simple_and_uni1"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    simple_pid_no_truth_dir = output_base_dir / "simple_pid_no_truth"
    simple_pid_no_truth_dir.mkdir(exist_ok=True)
    
    simple_pid_truth_dir = output_base_dir / "simple_pid_truth"
    simple_pid_truth_dir.mkdir(exist_ok=True)
    
    uni_bild_1_dir = output_base_dir / "uni_bild_1"
    uni_bild_1_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # Run tests
    logger.info("\n" + "="*80)
    logger.info("Starting Test Suite...")
    logger.info("="*80 + "\n")
    
    # Test 1: Simple PID WITHOUT truth mode
    logger.info(">>> Test 1: Simple PID (NO TRUTH MODE) <<<")
    results['simple_pid_no_truth'] = run_analysis(
        simple_pid_path,
        simple_pid_no_truth_dir,
        truth_data=False
    )
    
    # Test 2: Simple PID WITH truth mode
    logger.info("\n>>> Test 2: Simple PID (TRUTH MODE) <<<")
    results['simple_pid_truth'] = run_analysis(
        simple_pid_path,
        simple_pid_truth_dir,
        truth_data=True
    )
    
    # Test 3: Uni-Bild 1 (can run in parallel, but for now sequential for logging clarity)
    logger.info("\n>>> Test 3: Uni-Bild 1 (NO TRUTH MODE) <<<")
    results['uni_bild_1'] = run_analysis(
        uni_bild_1_path,
        uni_bild_1_dir,
        truth_data=False
    )
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUITE SUMMARY")
    logger.info("="*80)
    
    for test_name, result in results.items():
        if result.get("success"):
            logger.info(f"\n✅ {test_name}:")
            logger.info(f"   Quality Score: {result.get('quality_score', 0):.2f}%")
            logger.info(f"   Elements: {result.get('elements_count', 0)}")
            logger.info(f"   Connections: {result.get('connections_count', 0)}")
        else:
            logger.error(f"\n❌ {test_name}: FAILED")
            logger.error(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Comparison: Simple PID with vs without truth
    if results.get('simple_pid_no_truth', {}).get('success') and results.get('simple_pid_truth', {}).get('success'):
        logger.info("\n" + "="*80)
        logger.info("COMPARISON: Simple PID (Truth Mode vs No Truth Mode)")
        logger.info("="*80)
        
        no_truth = results['simple_pid_no_truth']
        truth = results['simple_pid_truth']
        
        logger.info(f"\nQuality Score:")
        logger.info(f"   No Truth: {no_truth.get('quality_score', 0):.2f}%")
        logger.info(f"   Truth:    {truth.get('quality_score', 0):.2f}%")
        logger.info(f"   Difference: {truth.get('quality_score', 0) - no_truth.get('quality_score', 0):.2f}%")
        
        logger.info(f"\nElements:")
        logger.info(f"   No Truth: {no_truth.get('elements_count', 0)}")
        logger.info(f"   Truth:    {truth.get('elements_count', 0)}")
        logger.info(f"   Difference: {truth.get('elements_count', 0) - no_truth.get('elements_count', 0)}")
        
        logger.info(f"\nConnections:")
        logger.info(f"   No Truth: {no_truth.get('connections_count', 0)}")
        logger.info(f"   Truth:    {truth.get('connections_count', 0)}")
        logger.info(f"   Difference: {truth.get('connections_count', 0) - no_truth.get('connections_count', 0)}")
    
    logger.info("\n" + "="*80)
    logger.info("TEST SUITE COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

