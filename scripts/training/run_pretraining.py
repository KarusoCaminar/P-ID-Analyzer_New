#!/usr/bin/env python3
"""
Pretraining Script - Automatically processes all symbols from pretraining_symbols/

This script:
1. Processes all images in pretraining_symbols/ directory
2. Detects if images are collections (large) or individual symbols
3. Extracts symbols from collections automatically
4. Integrates symbols into library with duplicate checking
5. Saves to learning database

Usage:
    python scripts/run_pretraining.py
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
from src.analyzer.learning.symbol_library import SymbolLibrary
from src.analyzer.learning.active_learner import ActiveLearner
from src.services.config_service import ConfigService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pretraining function."""
    logger.info("=== Starting Pretraining Process ===")
    
    try:
        # Initialize services
        config_service = ConfigService()
        config = config_service.get_raw_config()
        
        # Get paths from config
        pretraining_path = Path(config.get('paths', {}).get('pretraining_symbols', 'training_data/pretraining_symbols'))
        learning_db_path = Path(config.get('paths', {}).get('learning_db', 'learning_db.json'))
        
        if not pretraining_path.exists():
            logger.error(f"Pretraining path not found: {pretraining_path}")
            logger.info(f"Please create directory: {pretraining_path}")
            return
        
        # Initialize LLM client
        project_id = os.getenv('GCP_PROJECT_ID')
        if not project_id:
            logger.error("GCP_PROJECT_ID environment variable not set")
            return
        
        llm_client = LLMClient(
            project_id=project_id,
            default_location=config.get('models', {}).get('Google Gemini 2.5 Flash', {}).get('location', 'us-central1'),
            config=config
        )
        
        # Initialize knowledge manager
        element_type_list_path = Path(config.get('paths', {}).get('element_type_list', 'element_type_list.json'))
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_handler=llm_client,
            config=config
        )
        
        # Initialize symbol library
        symbol_library = SymbolLibrary(
            llm_client=llm_client,
            learning_db_path=learning_db_path
        )
        
        # Initialize active learner
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
            # Fallback to first available model
            model_info = list(models_config.values())[0] if models_config else {}
        
        logger.info(f"Using model: {model_info.get('id', 'unknown')}")
        logger.info(f"Pretraining path: {pretraining_path}")
        
        # Count files before processing
        image_files = list(pretraining_path.glob("*.png")) + \
                     list(pretraining_path.glob("*.jpg")) + \
                     list(pretraining_path.glob("*.jpeg"))
        logger.info(f"Found {len(image_files)} image files in {pretraining_path}")
        
        # Run pretraining
        report = active_learner.learn_from_pretraining_symbols(
            pretraining_path=pretraining_path,
            model_info=model_info
        )
        
        # Print summary
        logger.info("=== Pretraining Complete ===")
        logger.info(f"Files processed: {report.get('symbols_processed', 0)}")
        logger.info(f"Collections processed: {report.get('collections_processed', 0)}")
        logger.info(f"Individual symbols processed: {report.get('individual_symbols_processed', 0)}")
        logger.info(f"New symbols learned: {report.get('symbols_learned', 0)}")
        logger.info(f"Symbols updated: {report.get('symbols_updated', 0)}")
        logger.info(f"Duplicates found: {report.get('duplicates_found', 0)}")
        
        if report.get('errors'):
            logger.warning(f"Errors encountered: {len(report.get('errors', []))}")
            for error in report.get('errors', [])[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
        
        # Show symbol library stats
        symbol_count = symbol_library.get_symbol_count()
        logger.info(f"Total symbols in library: {symbol_count}")
        
        logger.info("=== Pretraining Finished Successfully ===")
        
    except Exception as e:
        logger.error(f"Error in pretraining: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

