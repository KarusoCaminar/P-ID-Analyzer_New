#!/usr/bin/env python3
"""
Test-Run Script with Validation and Iterative Loop.

This script:
1. Runs pretraining first
2. Runs test with simple P&ID
3. Validates results
4. Runs iterative improvement loop
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
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.evaluation.kpi_calculator import KPICalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pretraining(config_service, llm_client, knowledge_manager, symbol_library, active_learner):
    """Run pretraining."""
    logger.info("=== STEP 1: Running Pretraining ===")
    
    try:
        config = config_service.get_raw_config()
        pretraining_path = Path(config.get('paths', {}).get('pretraining_symbols', 'training_data/pretraining_symbols'))
        
        if not pretraining_path.exists():
            logger.warning(f"Pretraining path not found: {pretraining_path}")
            return False
        
        models_config = config.get('models', {})
        model_info = models_config.get('Google Gemini 2.5 Flash', {})
        if not model_info:
            model_info = list(models_config.values())[0] if models_config else {}
        
        report = active_learner.learn_from_pretraining_symbols(
            pretraining_path=pretraining_path,
            model_info=model_info
        )
        
        logger.info(f"Pretraining complete: {report.get('symbols_learned', 0)} new symbols learned")
        return True
        
    except Exception as e:
        logger.error(f"Error in pretraining: {e}", exc_info=True)
        return False


def run_test_analysis(pipeline_coordinator, test_image_path: str, truth_path: Optional[str] = None):
    """Run test analysis."""
    logger.info(f"=== STEP 2: Running Test Analysis ===")
    logger.info(f"Test image: {test_image_path}")
    
    try:
        # Load truth data if available
        truth_data = None
        if truth_path and Path(truth_path).exists():
            import json
            with open(truth_path, 'r', encoding='utf-8') as f:
                truth_data = json.load(f)
            logger.info(f"Loaded truth data from {truth_path}")
        
        # Run analysis
        result = pipeline_coordinator.process(
            image_path=test_image_path,
            output_dir=None,
            params_override={}
        )
        
        if result and result.elements:
            logger.info(f"Analysis complete: {len(result.elements)} elements, {len(result.connections)} connections")
            return result, truth_data
        else:
            logger.error("Analysis failed: No elements detected")
            return None, truth_data
            
    except Exception as e:
        logger.error(f"Error in test analysis: {e}", exc_info=True)
        return None, None


def validate_results(result, truth_data, kpi_calculator):
    """Validate results."""
    logger.info("=== STEP 3: Validating Results ===")
    
    if not result or not truth_data:
        logger.warning("Cannot validate: Missing result or truth data")
        return None
    
    try:
        # Convert result to dict format
        analysis_data = {
            'elements': [el.model_dump() if hasattr(el, 'model_dump') else el.__dict__ for el in result.elements],
            'connections': [conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ for conn in result.connections]
        }
        
        # Calculate KPIs
        kpis = kpi_calculator.calculate_kpis(analysis_data, truth_data)
        
        logger.info(f"Validation complete:")
        logger.info(f"  Quality Score: {kpis.get('quality_score', 0.0):.2f}%")
        logger.info(f"  Element Precision: {kpis.get('element_precision', 0.0):.2f}")
        logger.info(f"  Element Recall: {kpis.get('element_recall', 0.0):.2f}")
        logger.info(f"  Element F1: {kpis.get('element_f1', 0.0):.2f}")
        logger.info(f"  Type Accuracy: {kpis.get('type_accuracy', 0.0):.2f}")
        
        return kpis
        
    except Exception as e:
        logger.error(f"Error in validation: {e}", exc_info=True)
        return None


def run_iterative_loop(pipeline_coordinator, test_image_path: str, truth_path: Optional[str], kpi_calculator, max_iterations: int = 5):
    """Run iterative improvement loop."""
    logger.info("=== STEP 4: Starting Iterative Improvement Loop ===")
    
    best_score = 0.0
    best_result = None
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"\n--- Iteration {iteration}/{max_iterations} ---")
        
        # Run analysis
        result, truth_data = run_test_analysis(pipeline_coordinator, test_image_path, truth_path)
        
        if not result:
            logger.warning(f"Iteration {iteration}: Analysis failed")
            continue
        
        # Validate
        kpis = validate_results(result, truth_data, kpi_calculator)
        
        if not kpis:
            logger.warning(f"Iteration {iteration}: Validation failed")
            continue
        
        current_score = kpis.get('quality_score', 0.0)
        logger.info(f"Iteration {iteration}: Quality Score = {current_score:.2f}%")
        
        # Check if improved
        if current_score > best_score:
            improvement = current_score - best_score
            logger.info(f"✓ Improvement: {improvement:.2f}% (new best: {current_score:.2f}%)")
            best_score = current_score
            best_result = result
        else:
            logger.info(f"✗ No improvement (current: {current_score:.2f}%, best: {best_score:.2f}%)")
        
        # Early stop if score is very high
        if current_score >= 95.0:
            logger.info(f"✓ Excellent score reached ({current_score:.2f}%), stopping early")
            break
        
        # Stop if no improvement for 2 iterations
        if iteration >= 2 and current_score <= best_score:
            logger.info("No improvement for 2 iterations, stopping")
            break
    
    logger.info(f"\n=== Iterative Loop Complete ===")
    logger.info(f"Best Score: {best_score:.2f}%")
    logger.info(f"Total Iterations: {iteration}")
    
    return best_result, best_score


def main():
    """Main function."""
    logger.info("=== Starting Test-Run with Validation and Iterative Loop ===")
    
    try:
        # Initialize services
        config_service = ConfigService()
        config = config_service.get_raw_config()
        
        # Get GCP_PROJECT_ID
        project_id = os.getenv('GCP_PROJECT_ID')
        if not project_id:
            logger.error("GCP_PROJECT_ID environment variable not set")
            logger.info("Please set GCP_PROJECT_ID in .env file")
            return
        
        # Initialize LLM client
        llm_client = LLMClient(
            project_id=project_id,
            default_location=config.get('models', {}).get('Google Gemini 2.5 Flash', {}).get('location', 'us-central1'),
            config=config
        )
        
        # Initialize knowledge manager
        element_type_list_path = Path(config.get('paths', {}).get('element_type_list', 'element_type_list.json'))
        learning_db_path = Path(config.get('paths', {}).get('learning_db', 'learning_db.json'))
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
        
        # STEP 1: Run pretraining
        pretraining_success = run_pretraining(config_service, llm_client, knowledge_manager, symbol_library, active_learner)
        
        if not pretraining_success:
            logger.warning("Pretraining had issues, but continuing with test...")
        
        # Initialize pipeline coordinator
        pipeline_coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service,
            model_strategy={},
            progress_callback=None
        )
        
        # Initialize KPI calculator
        kpi_calculator = KPICalculator()
        
        # STEP 2: Run test analysis
        test_image_path = "training_data/simple_pids/Einfaches P&I.png"
        truth_path = "training_data/simple_pids/Einfaches P&I_truth.json"
        
        if not Path(test_image_path).exists():
            logger.error(f"Test image not found: {test_image_path}")
            return
        
        result, truth_data = run_test_analysis(pipeline_coordinator, test_image_path, truth_path)
        
        if not result:
            logger.error("Test analysis failed")
            return
        
        # STEP 3: Validate results
        kpis = validate_results(result, truth_data, kpi_calculator)
        
        if not kpis:
            logger.error("Validation failed")
            return
        
        # STEP 4: Run iterative loop
        best_result, best_score = run_iterative_loop(
            pipeline_coordinator,
            test_image_path,
            truth_path,
            kpi_calculator,
            max_iterations=5
        )
        
        logger.info("\n=== FINAL SUMMARY ===")
        logger.info(f"Best Quality Score: {best_score:.2f}%")
        logger.info(f"Pretraining: {'Success' if pretraining_success else 'Had issues'}")
        logger.info("=== Test-Run Complete ===")
        
    except Exception as e:
        logger.error(f"Error in test-run: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

