#!/usr/bin/env python3
"""
Strategy Comparison Test Script

Tests three strategies with Simple PID:
1. all_flash - All Flash models
2. optimal_swarm_monolith - Flash for Swarm, Pro for Monolith
3. optimal_swarm_monolith_lite - Flash-Lite Preview for Swarm, Flash for Monolith

Saves results with clear naming including strategy and active settings.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
import time

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
        logger.info(f"[Correction] {correction_text}")


def get_active_settings(config_service: ConfigService, strategy_name: str) -> dict:
    """Extract active settings from config."""
    config = config_service.get_config()
    
    # Handle Pydantic models - convert to dict if needed
    # Use ConfigService method which handles conversion
    logic_params = config_service.get_logic_parameters()
    
    # Get strategy dict - handle both dict and Pydantic model
    if isinstance(config.strategies, dict):
        strategy_dict = config.strategies.get(strategy_name, {})
    else:
        # If strategies is a Pydantic model, try to get it
        strategy_dict = getattr(config.strategies, strategy_name, {}) if hasattr(config.strategies, strategy_name) else {}
        if hasattr(strategy_dict, 'model_dump'):
            strategy_dict = strategy_dict.model_dump()
        elif not isinstance(strategy_dict, dict):
            strategy_dict = {}
    
    return {
        'strategy': strategy_name,
        'swarm_model': strategy_dict.get('swarm_model', 'N/A'),
        'monolith_model': strategy_dict.get('monolith_model', 'N/A'),
        'detail_model': strategy_dict.get('detail_model', 'N/A'),
        'correction_model': strategy_dict.get('correction_model', 'N/A'),
        'critic_model': strategy_dict.get('critic_model_name', 'N/A'),
        'use_swarm_analysis': logic_params.get('use_swarm_analysis', True),
        'use_monolith_analysis': logic_params.get('use_monolith_analysis', True),
        'use_fusion': logic_params.get('use_fusion', True),
        'use_self_correction_loop': logic_params.get('use_self_correction_loop', True),
        'use_type_validation': logic_params.get('use_type_validation', True),
        'use_confidence_filtering': logic_params.get('use_confidence_filtering', True),
        'confidence_threshold': logic_params.get('confidence_threshold', 0.5),
        'iou_match_threshold': logic_params.get('iou_match_threshold', 0.3),
        'max_self_correction_iterations': logic_params.get('max_self_correction_iterations', 15),
        'use_predictive_completion': logic_params.get('use_predictive_completion', True),
        'use_polyline_refinement': logic_params.get('use_polyline_refinement', True),
        'use_cv_bbox_refinement': logic_params.get('use_cv_bbox_refinement', True),
        'use_cv_text_detection': logic_params.get('use_cv_text_detection', True),
        'use_legend_matching': logic_params.get('use_legend_matching', True),
        'use_llm_id_correction': logic_params.get('use_llm_id_correction', True),
        'use_context_type_inference': logic_params.get('use_context_type_inference', True),
        'use_cot_reasoning': logic_params.get('use_cot_reasoning', True),
        'llm_executor_workers': logic_params.get('llm_executor_workers', 6),
    }


def run_analysis_with_strategy(
    image_path: Path,
    truth_path: Optional[Path],
    strategy_name: str,
    output_base_dir: Path
) -> dict:
    """Run analysis with specific strategy."""
    logger.info(f"\n{'='*80}")
    logger.info(f"=== STRATEGY: {strategy_name} ===")
    logger.info(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Create output directory with strategy name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"strategy_{strategy_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize services
    config_service = ConfigService(project_root / "config.yaml")
    
    # Load strategy
    config = config_service.get_config()
    if strategy_name not in config.strategies:
        logger.error(f"Strategy '{strategy_name}' not found in config!")
        return {"error": f"Strategy '{strategy_name}' not found"}
    
    strategy = config.strategies[strategy_name]
    
    # Convert Pydantic model to dict for LLMClient and KnowledgeManager
    config_dict = config.model_dump()
    
    llm_client = LLMClient(
        project_id=os.getenv("GCP_PROJECT_ID"),
        default_location=os.getenv("GCP_LOCATION"),
        config=config_dict
    )
    
    knowledge_manager = KnowledgeManager(
        element_type_list_path=project_root / "element_type_list.json",
        learning_db_path=project_root / "learning_db.json",
        llm_handler=llm_client,
        config=config_dict
    )
    
    pipeline = PipelineCoordinator(
        llm_client=llm_client,
        knowledge_manager=knowledge_manager,
        config_service=config_service,
        model_strategy=strategy,
        progress_callback=DummyProgressCallback()
    )
    
    try:
        # Run analysis
        result = pipeline.process(
            image_path=str(image_path),
            output_dir=str(output_dir),
            params_override={'truth_data_path': str(truth_path) if truth_path else None}
        )
        
        elapsed_time = time.time() - start_time
        
        # Get active settings
        active_settings = get_active_settings(config_service, strategy_name)
        
        # Compile results
        test_result = {
            'strategy': strategy_name,
            'timestamp': timestamp,
            'elapsed_time_seconds': round(elapsed_time, 2),
            'quality_score': round(result.quality_score, 2) if hasattr(result, 'quality_score') else None,
            'active_settings': active_settings,
            'result': result.model_dump() if hasattr(result, 'model_dump') else result,
            'output_directory': str(output_dir)
        }
        
        # Save test result
        result_file = output_dir / f"test_result_{strategy_name}_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Strategy '{strategy_name}' completed in {elapsed_time:.2f}s")
        logger.info(f"  Quality Score: {test_result['quality_score']}%")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"{'='*80}\n")
        
        return test_result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"✗ Strategy '{strategy_name}' failed after {elapsed_time:.2f}s: {e}", exc_info=True)
        return {
            'strategy': strategy_name,
            'error': str(e),
            'elapsed_time_seconds': round(elapsed_time, 2)
        }


def main():
    """Main test function."""
    logger.info("\n" + "="*80)
    logger.info("STRATEGY COMPARISON TEST")
    logger.info("="*80 + "\n")
    
    # Setup paths
    simple_pid_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
    simple_pid_truth_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
    
    if not simple_pid_path.exists():
        logger.error(f"Simple PID image not found: {simple_pid_path}")
        return
    
    if not simple_pid_truth_path.exists():
        logger.warning(f"Simple PID truth data not found: {simple_pid_truth_path}")
        simple_pid_truth_path = None
    
    # Create output directory
    output_base_dir = project_root / "outputs" / "strategy_comparison"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Test strategies
    strategies = [
        "all_flash",
        "optimal_swarm_monolith",
        "optimal_swarm_monolith_lite"
    ]
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n>>> Starting test with strategy: {strategy} <<<\n")
        result = run_analysis_with_strategy(
            image_path=simple_pid_path,
            truth_path=simple_pid_truth_path,
            strategy_name=strategy,
            output_base_dir=output_base_dir
        )
        results[strategy] = result
    
    # Save comparison report
    comparison_file = output_base_dir / f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("STRATEGY COMPARISON SUMMARY")
    logger.info("="*80 + "\n")
    
    for strategy, result in results.items():
        if 'error' in result:
            logger.error(f"{strategy}: FAILED - {result['error']}")
        else:
            logger.info(f"{strategy}:")
            logger.info(f"  Time: {result.get('elapsed_time_seconds', 'N/A')}s")
            logger.info(f"  Quality: {result.get('quality_score', 'N/A')}%")
            logger.info(f"  Output: {result.get('output_directory', 'N/A')}")
    
    logger.info(f"\nComparison report saved: {comparison_file}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()

