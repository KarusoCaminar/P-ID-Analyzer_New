#!/usr/bin/env python3
"""
Test Pretraining Script - Tests and evaluates pretraining before generating viewshots.

This script:
1. Tests pretraining symbol extraction
2. Evaluates extraction quality
3. Saves results to outputs/pretraining_tests/
4. Provides metrics for pretraining evaluation

Usage:
    python scripts/training/test_pretraining.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.learning.symbol_library import SymbolLibrary
from src.analyzer.learning.active_learner import ActiveLearner
from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService

# Setup logging
output_dir = project_root / "outputs" / "pretraining_tests"
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / f"test_pretraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LoggingService.setup_logging(
    log_level=logging.INFO,
    log_file=log_file
)
logger = logging.getLogger(__name__)


def test_pretraining() -> Dict[str, Any]:
    """
    Test pretraining and return evaluation results.
    
    Returns:
        Dictionary with test results and metrics
    """
    logger.info("=" * 60)
    logger.info("[TEST] Starting Pretraining Test")
    logger.info("=" * 60)
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "errors": [],
        "warnings": [],
        "metrics": {},
        "symbols_extracted": 0,
        "symbols_learned": 0,
        "collections_processed": 0,
        "individual_symbols_processed": 0
    }
    
    try:
        # Initialize services
        logger.info("[INIT] Initializing services...")
        config_service = ConfigService()
        config = config_service.get_raw_config()
        
        # Get paths from config
        pretraining_path = Path(config.get('paths', {}).get('pretraining_symbols', 'training_data/pretraining_symbols'))
        learning_db_path = Path(config.get('paths', {}).get('learning_db', 'learning_db.json'))
        
        # Validate pretraining path
        if not pretraining_path.exists():
            error_msg = f"Pretraining path not found: {pretraining_path}"
            logger.error(f"[ERROR] {error_msg}")
            test_results["errors"].append(error_msg)
            return test_results
        
        logger.info(f"[OK] Pretraining path: {pretraining_path}")
        
        # Count files before processing
        image_files = list(pretraining_path.glob("*.png")) + \
                     list(pretraining_path.glob("*.jpg")) + \
                     list(pretraining_path.glob("*.jpeg"))
        
        logger.info(f"[INFO] Found {len(image_files)} image files in {pretraining_path}")
        test_results["metrics"]["files_found"] = len(image_files)
        
        if len(image_files) == 0:
            warning_msg = "No image files found in pretraining directory"
            logger.warning(f"[WARNING] {warning_msg}")
            test_results["warnings"].append(warning_msg)
            return test_results
        
        # Initialize LLM client
        project_id = os.getenv('GCP_PROJECT_ID')
        if not project_id:
            error_msg = "GCP_PROJECT_ID environment variable not set"
            logger.error(f"[ERROR] {error_msg}")
            test_results["errors"].append(error_msg)
            return test_results
        
        logger.info("[INIT] Initializing LLM client...")
        llm_client = LLMClient(
            project_id=project_id,
            default_location=config.get('models', {}).get('Google Gemini 2.5 Flash', {}).get('location', 'us-central1'),
            config=config
        )
        
        # Initialize knowledge manager
        logger.info("[INIT] Initializing knowledge manager...")
        element_type_list_path = Path(config.get('paths', {}).get('element_type_list', 'element_type_list.json'))
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_handler=llm_client,
            config=config
        )
        
        # Initialize symbol library
        logger.info("[INIT] Initializing symbol library...")
        learned_symbols_images_dir = Path(config.get('paths', {}).get('learned_symbols_images_dir', 'learned_symbols_images'))
        symbol_library = SymbolLibrary(
            llm_client=llm_client,
            learning_db_path=learning_db_path,
            images_dir=learned_symbols_images_dir
        )
        
        # Get initial symbol count
        initial_symbol_count = symbol_library.get_symbol_count()
        logger.info(f"[INFO] Initial symbol count: {initial_symbol_count}")
        test_results["metrics"]["initial_symbol_count"] = initial_symbol_count
        
        # Initialize active learner
        logger.info("[INIT] Initializing active learner...")
        active_learner = ActiveLearner(
            knowledge_manager=knowledge_manager,
            symbol_library=symbol_library,
            llm_client=llm_client,
            config=config
        )
        
        # Get model info for symbol extraction
        models_config = config.get('models', {})
        model_info = models_config.get('Google Gemini 2.5 Flash', {})
        if not model_info:
            model_info = list(models_config.values())[0] if models_config else {}
        
        logger.info(f"[INFO] Using model: {model_info.get('id', 'unknown')}")
        
        # Run pretraining
        logger.info("=" * 60)
        logger.info("[PRETRAINING] Starting pretraining process...")
        logger.info("=" * 60)
        
        report = active_learner.learn_from_pretraining_symbols(
            pretraining_path=pretraining_path,
            model_info=model_info
        )
        
        # Get final symbol count
        final_symbol_count = symbol_library.get_symbol_count()
        logger.info(f"[INFO] Final symbol count: {final_symbol_count}")
        test_results["metrics"]["final_symbol_count"] = final_symbol_count
        test_results["metrics"]["symbols_added"] = final_symbol_count - initial_symbol_count
        
        # Extract results from report
        test_results["symbols_extracted"] = report.get('symbols_processed', 0)
        test_results["symbols_learned"] = report.get('symbols_learned', 0)
        test_results["collections_processed"] = report.get('collections_processed', 0)
        test_results["individual_symbols_processed"] = report.get('individual_symbols_processed', 0)
        test_results["metrics"]["symbols_updated"] = report.get('symbols_updated', 0)
        test_results["metrics"]["duplicates_found"] = report.get('duplicates_found', 0)
        
        if report.get('errors'):
            test_results["errors"].extend(report.get('errors', []))
            logger.warning(f"[WARNING] Errors encountered: {len(report.get('errors', []))}")
            for error in report.get('errors', [])[:5]:
                logger.warning(f"  - {error}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("[RESULTS] Pretraining Test Results")
        logger.info("=" * 60)
        logger.info(f"Files processed: {test_results['symbols_extracted']}")
        logger.info(f"Collections processed: {test_results['collections_processed']}")
        logger.info(f"Individual symbols processed: {test_results['individual_symbols_processed']}")
        logger.info(f"New symbols learned: {test_results['symbols_learned']}")
        logger.info(f"Symbols updated: {test_results['metrics']['symbols_updated']}")
        logger.info(f"Duplicates found: {test_results['metrics']['duplicates_found']}")
        logger.info(f"Symbols added to library: {test_results['metrics']['symbols_added']}")
        logger.info(f"Total symbols in library: {final_symbol_count}")
        
        test_results["success"] = True
        
        logger.info("=" * 60)
        logger.info("[DONE] Pretraining test completed successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        error_msg = f"Error in pretraining test: {e}"
        logger.error(f"[ERROR] {error_msg}", exc_info=True)
        test_results["errors"].append(error_msg)
        test_results["success"] = False
    
    return test_results


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("[START] Pretraining Test Script")
    logger.info("=" * 60)
    
    # Run test
    test_results = test_pretraining()
    
    # Save results
    results_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[SAVE] Test results saved to: {results_file}")
    
    # Print final summary
    logger.info("=" * 60)
    logger.info("[SUMMARY] Final Test Summary")
    logger.info("=" * 60)
    logger.info(f"Success: {test_results['success']}")
    logger.info(f"Symbols extracted: {test_results['symbols_extracted']}")
    logger.info(f"Symbols learned: {test_results['symbols_learned']}")
    logger.info(f"Errors: {len(test_results['errors'])}")
    logger.info(f"Warnings: {len(test_results['warnings'])}")
    logger.info("=" * 60)
    
    if not test_results["success"]:
        logger.error("[ERROR] Pretraining test failed!")
        sys.exit(1)
    
    logger.info("[OK] Pretraining test completed successfully!")


if __name__ == "__main__":
    main()

