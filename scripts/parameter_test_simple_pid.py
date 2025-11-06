#!/usr/bin/env python3
"""
Parameter-Test für Simple P&ID mit wichtigsten Strategien:

1. Baseline (Phase 3 aktiviert)
2. Kritiker-Bypass (Phase 3 überspringen)
3. Ablation Study (Phase 3 + Phase 4.7 CV BBox Refinement deaktiviert)

Alle Tests mit Gemini 2.5 Flash (schnell + guter Qualitäts-Durchschnitt)
Validierung gegen Einfaches P&I_truth.json (Gold Standard: 10 Elements, 8 Connections)
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

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
    """Progress callback for testing."""
    
    def update_progress(self, progress: int, message: str):
        logger.info(f"[Progress {progress}%] {message}")
    
    def update_status_label(self, text: str):
        logger.info(f"[Status] {text}")
    
    def report_truth_mode(self, active: bool):
        logger.info(f"[Truth Mode] {'ACTIVE' if active else 'INACTIVE'}")
    
    def report_correction(self, correction_text: str):
        logger.info(f"[Correction] {correction_text}")


def load_truth_data(truth_path: Path) -> Dict[str, Any]:
    """Load truth data from JSON file."""
    with open(truth_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_against_truth(result: Any, truth_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate result against truth data."""
    # Extract elements and connections from result
    if hasattr(result, 'elements'):
        result_elements = result.elements
    elif isinstance(result, dict):
        result_elements = result.get('elements', [])
    else:
        result_elements = []
    
    if hasattr(result, 'connections'):
        result_connections = result.connections
    elif isinstance(result, dict):
        result_connections = result.get('connections', [])
    else:
        result_connections = []
    
    # Extract truth data
    truth_elements = truth_data.get('elements', [])
    truth_connections = truth_data.get('connections', [])
    
    # Count matches
    truth_element_ids = {el.get('id') for el in truth_elements if el.get('id')}
    result_element_ids = {el.get('id') for el in result_elements if el.get('id')}
    
    matched_elements = truth_element_ids & result_element_ids
    missed_elements = truth_element_ids - result_element_ids
    hallucinated_elements = result_element_ids - truth_element_ids
    
    # Connection matching (simplified: match by from_id and to_id)
    truth_conn_keys = {(conn.get('from_id'), conn.get('to_id')) for conn in truth_connections if conn.get('from_id') and conn.get('to_id')}
    result_conn_keys = {(conn.get('from_id'), conn.get('to_id')) for conn in result_connections if conn.get('from_id') and conn.get('to_id')}
    
    matched_connections = truth_conn_keys & result_conn_keys
    missed_connections = truth_conn_keys - result_conn_keys
    hallucinated_connections = result_conn_keys - truth_conn_keys
    
    # Calculate metrics
    element_precision = len(matched_elements) / len(result_element_ids) if result_element_ids else 0.0
    element_recall = len(matched_elements) / len(truth_element_ids) if truth_element_ids else 0.0
    element_f1 = 2 * (element_precision * element_recall) / (element_precision + element_recall) if (element_precision + element_recall) > 0 else 0.0
    
    connection_precision = len(matched_connections) / len(result_conn_keys) if result_conn_keys else 0.0
    connection_recall = len(matched_connections) / len(truth_conn_keys) if truth_conn_keys else 0.0
    connection_f1 = 2 * (connection_precision * connection_recall) / (connection_precision + connection_recall) if (connection_precision + connection_recall) > 0 else 0.0
    
    return {
        'truth_elements': len(truth_elements),
        'truth_connections': len(truth_connections),
        'result_elements': len(result_elements),
        'result_connections': len(result_connections),
        'matched_elements': len(matched_elements),
        'missed_elements': len(missed_elements),
        'hallucinated_elements': len(hallucinated_elements),
        'matched_connections': len(matched_connections),
        'missed_connections': len(missed_connections),
        'hallucinated_connections': len(hallucinated_connections),
        'element_precision': element_precision,
        'element_recall': element_recall,
        'element_f1': element_f1,
        'connection_precision': connection_precision,
        'connection_recall': connection_recall,
        'connection_f1': connection_f1,
        'missed_element_ids': list(missed_elements),
        'hallucinated_element_ids': list(hallucinated_elements)
    }


def run_test_with_config(
    config_service: ConfigService,
    llm_client: LLMClient,
    knowledge_manager: KnowledgeManager,
    image_path: Path,
    truth_path: Path,
    output_dir: Path,
    test_name: str,
    config_overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Run test with specific configuration overrides."""
    logger.info("\n" + "="*80)
    logger.info(f"TEST: {test_name}")
    logger.info(f"Config: {config_overrides}")
    logger.info("="*80)
    
    # Apply config overrides
    config = config_service.get_config()
    config_dict = config.model_dump()
    
    if 'logic_parameters' not in config_dict:
        config_dict['logic_parameters'] = {}
    
    # Store original values
    original_values = {}
    for key, value in config_overrides.items():
        if key in config_dict['logic_parameters']:
            original_values[key] = config_dict['logic_parameters'][key]
        config_dict['logic_parameters'][key] = value
    
    # Update config service
    config_service.update_config(config_dict)
    
    # Use simple_pid_strategy for all tests (Gemini 2.5 Flash)
    strategy = config_dict.get('strategies', {}).get('simple_pid_strategy', {})
    if not strategy:
        # Fallback to all_flash strategy
        strategy = config_dict.get('strategies', {}).get('all_flash', {})
    
    try:
        pipeline = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service,
            model_strategy=strategy,
            progress_callback=DummyProgressCallback()
        )
        
        result = pipeline.process(
            image_path=str(image_path),
            output_dir=str(output_dir / test_name),
            params_override={'truth_data_path': str(truth_path)}
        )
        
        truth_data = load_truth_data(truth_path)
        validation = validate_against_truth(result, truth_data)
        
        return {
            'test': test_name,
            'config': config_overrides,
            'result': result.model_dump() if hasattr(result, 'model_dump') else result,
            'validation': validation,
            'quality_score': result.quality_score if hasattr(result, 'quality_score') else 0.0
        }
    except Exception as e:
        logger.error(f"Test '{test_name}' failed: {e}", exc_info=True)
        return {'test': test_name, 'error': str(e)}
    finally:
        # Restore original values
        for key, value in original_values.items():
            config_dict['logic_parameters'][key] = value
        config_service.update_config(config_dict)


def main():
    """Main function."""
    output_base_dir = project_root / "outputs" / "parameter_test_simple_pid"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Test image and truth data
    simple_pid_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
    simple_pid_truth_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
    
    if not simple_pid_path.exists():
        logger.error(f"Image not found: {simple_pid_path}")
        return
    
    if not simple_pid_truth_path.exists():
        logger.error(f"Truth data not found: {simple_pid_truth_path}")
        return
    
    # Load truth data to show expected values
    truth_data = load_truth_data(simple_pid_truth_path)
    logger.info(f"\n{'='*80}")
    logger.info("GOLD STANDARD (Truth Data):")
    logger.info(f"  Elements: {len(truth_data.get('elements', []))}")
    logger.info(f"  Connections: {len(truth_data.get('connections', []))}")
    logger.info(f"{'='*80}\n")
    
    # Initialize services
    config_service = ConfigService(project_root / "config.yaml")
    config_dict = config_service.get_config().model_dump()
    
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
    
    all_results = []
    
    # Test 1: Baseline (Phase 3 aktiviert, alle Phasen aktiv)
    baseline_result = run_test_with_config(
        config_service, llm_client, knowledge_manager,
        simple_pid_path, simple_pid_truth_path, output_base_dir,
        "baseline",
        {}  # Alle Standard-Einstellungen aktiviert
    )
    all_results.append(baseline_result)
    
    # Test 2: Kritiker-Bypass (Phase 3 überspringen)
    kritiker_bypass_result = run_test_with_config(
        config_service, llm_client, knowledge_manager,
        simple_pid_path, simple_pid_truth_path, output_base_dir,
        "kritiker_bypass",
        {'use_self_correction_loop': False}
    )
    all_results.append(kritiker_bypass_result)
    
    # Test 3: Ablation Study (Phase 3 + Phase 4.7 CV BBox Refinement deaktiviert)
    ablation_result = run_test_with_config(
        config_service, llm_client, knowledge_manager,
        simple_pid_path, simple_pid_truth_path, output_base_dir,
        "ablation_no_phase3_no_cv_bbox",
        {
            'use_self_correction_loop': False,
            'use_cv_bbox_refinement': False
        }
    )
    all_results.append(ablation_result)
    
    # Save results
    results_file = output_base_dir / "parameter_test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PARAMETER TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"\nGold Standard: {len(truth_data.get('elements', []))} Elements, {len(truth_data.get('connections', []))} Connections\n")
    
    for result in all_results:
        if 'error' in result:
            logger.error(f"{result['test']}: FAILED - {result['error']}")
        else:
            validation = result.get('validation', {})
            logger.info(f"\n{result['test'].upper()}:")
            logger.info(f"  Elements: {validation.get('result_elements', 0)} (Expected: {validation.get('truth_elements', 0)}) - Matched: {validation.get('matched_elements', 0)}")
            logger.info(f"  Connections: {validation.get('result_connections', 0)} (Expected: {validation.get('truth_connections', 0)}) - Matched: {validation.get('matched_connections', 0)}")
            logger.info(f"  Element F1: {validation.get('element_f1', 0.0):.3f}")
            logger.info(f"  Connection F1: {validation.get('connection_f1', 0.0):.3f}")
            logger.info(f"  Quality Score: {result.get('quality_score', 0.0):.2f}%")
            if validation.get('missed_element_ids'):
                logger.info(f"  Missed Elements: {validation.get('missed_element_ids')}")
            if validation.get('hallucinated_element_ids'):
                logger.info(f"  Hallucinated Elements: {validation.get('hallucinated_element_ids')}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()

