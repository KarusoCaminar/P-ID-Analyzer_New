#!/usr/bin/env python3
"""
Phase 1: Stabilitäts- & Ablationstests (Einfaches P&ID)

Ziel: Finden der besten Basis-Konfiguration (Phase 2) und sicherstellen,
dass die nachfolgenden Phasen 3 und 4 die Daten nicht zerstören.

WICHTIG: Für alle Tests in dieser Phase gilt:
- Bild: Einfaches P&I.png
- Active Learner (AL): AUS und learning_db.json zurückgesetzt
- Strategie: simple_pid_strategy (Gemini 2.5 Flash)
- Phase 3 & 4: AUS (nur Phase 2 wird getestet)

Tests:
- T1a (Neu): Monolith Pro Raw (use_swarm_analysis=false, use_monolith_analysis=true, use_fusion=false, use_phase4=false)
- T4 (Neu): Monolith Pro + P4 (use_swarm_analysis=false, use_monolith_analysis=true, use_fusion=false, use_phase4=true)
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator

# Setup logging with live output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Live output to console
        logging.FileHandler('outputs/phase1_tests.log', encoding='utf-8')  # Also save to file
    ]
)
logger = logging.getLogger(__name__)

# Also configure root logger for pipeline components
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))


class DummyProgressCallback:
    """Progress callback for testing."""
    
    def update_progress(self, progress: int, message: str):
        if progress % 10 == 0:  # Only log every 10%
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
    
    # Helper function to get ID from element (dict or Pydantic model)
    def get_element_id(el):
        if isinstance(el, dict):
            return el.get('id')
        elif hasattr(el, 'id'):
            return el.id
        return None
    
    # Count matches
    truth_element_ids = {get_element_id(el) for el in truth_elements if get_element_id(el)}
    result_element_ids = {get_element_id(el) for el in result_elements if get_element_id(el)}
    
    matched_elements = truth_element_ids & result_element_ids
    missed_elements = truth_element_ids - result_element_ids
    hallucinated_elements = result_element_ids - truth_element_ids
    
    # Connection matching - handle both formats:
    # 1. Truth format: from_converter_ports/to_converter_ports with unit_name
    # 2. Result format: from_id/to_id
    def extract_connection_endpoints(conn):
        """Extract connection endpoints from either format."""
        from_id = None
        to_id = None
        
        if isinstance(conn, dict):
            # Check for result format (from_id/to_id)
            if 'from_id' in conn and 'to_id' in conn:
                from_id = conn.get('from_id')
                to_id = conn.get('to_id')
            # Check for truth format (from_converter_ports/to_converter_ports)
            elif 'from_converter_ports' in conn and 'to_converter_ports' in conn:
                from_ports = conn.get('from_converter_ports', [])
                to_ports = conn.get('to_converter_ports', [])
                
                # Extract unit_name from from_converter_ports
                if from_ports and isinstance(from_ports, list) and len(from_ports) > 0:
                    from_port = from_ports[0] if isinstance(from_ports[0], dict) else from_ports[0]
                    if isinstance(from_port, dict):
                        from_id = from_port.get('unit_name')
                
                # Extract unit_name from to_converter_ports
                if to_ports and isinstance(to_ports, list) and len(to_ports) > 0:
                    to_port = to_ports[0] if isinstance(to_ports[0], dict) else to_ports[0]
                    if isinstance(to_port, dict):
                        to_id = to_port.get('unit_name')
        elif hasattr(conn, 'from_id') and hasattr(conn, 'to_id'):
            # Pydantic model with from_id/to_id
            from_id = conn.from_id
            to_id = conn.to_id
        
        return from_id, to_id
    
    # Extract truth connection keys
    truth_conn_keys = set()
    for conn in truth_connections:
        from_id, to_id = extract_connection_endpoints(conn)
        if from_id and to_id:
            truth_conn_keys.add((from_id, to_id))
    
    # Extract result connection keys
    result_conn_keys = set()
    for conn in result_connections:
        from_id, to_id = extract_connection_endpoints(conn)
        if from_id and to_id:
            result_conn_keys.add((from_id, to_id))
    
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
        'elements': {
            'truth_count': len(truth_element_ids),
            'result_count': len(result_element_ids),
            'matched': len(matched_elements),
            'missed': len(missed_elements),
            'hallucinated': len(hallucinated_elements),
            'precision': element_precision,
            'recall': element_recall,
            'f1': element_f1
        },
        'connections': {
            'truth_count': len(truth_conn_keys),
            'result_count': len(result_conn_keys),
            'matched': len(matched_connections),
            'missed': len(missed_connections),
            'hallucinated': len(hallucinated_connections),
            'precision': connection_precision,
            'recall': connection_recall,
            'f1': connection_f1
        },
        'matched_element_ids': list(matched_elements),
        'missed_element_ids': list(missed_elements),
        'hallucinated_element_ids': list(hallucinated_elements),
        'matched_connection_keys': [list(k) for k in matched_connections],
        'missed_connection_keys': [list(k) for k in missed_connections],
        'hallucinated_connection_keys': [list(k) for k in hallucinated_connections]
    }


def run_test(
    test_name: str,
    test_id: str,
    image_path: Path,
    truth_path: Path,
    config_service: ConfigService,
    use_swarm: bool,
    use_monolith: bool,
    use_fusion: bool,
    use_phase3: bool = False,
    use_phase4: bool = False
) -> Dict[str, Any]:
    """Run a single test configuration."""
    logger.info("=" * 80)
    logger.info(f"TEST: {test_id} - {test_name}")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - use_swarm_analysis: {use_swarm}")
    logger.info(f"  - use_monolith_analysis: {use_monolith}")
    logger.info(f"  - use_fusion: {use_fusion}")
    logger.info(f"  - use_phase3 (Self-Correction Loop): {use_phase3}")
    logger.info(f"  - use_phase4 (Post-Processing): {use_phase4}")
    logger.info("")
    
    try:
        # Load truth data
        truth_data = load_truth_data(truth_path)
        logger.info(f"Truth data loaded: {len(truth_data.get('elements', []))} elements, {len(truth_data.get('connections', []))} connections")
        
        # Initialize components
        config_dict = config_service.get_raw_config()
        llm_client = LLMClient(
            project_id=os.getenv("GCP_PROJECT_ID"),
            default_location=os.getenv("GCP_LOCATION", "us-central1"),
            config=config_dict
        )
        knowledge_manager = KnowledgeManager(
            element_type_list_path=project_root / "element_type_list.json",
            learning_db_path=project_root / "learning_db.json",
            llm_handler=llm_client,
            config=config_dict
        )
        
        # Create pipeline coordinator with simple_pid_strategy
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service,
            model_strategy=None,  # Will use default from config
            progress_callback=DummyProgressCallback()
        )
        
        # Override model strategy to use simple_pid_strategy
        config_dict = config_service.get_raw_config()
        strategies = config_dict.get('strategies', {})
        simple_pid_strategy = strategies.get('simple_pid_strategy', {})
        
        if simple_pid_strategy:
            coordinator.model_strategy = {}
            for key, model_name in simple_pid_strategy.items():
                models = config_dict.get('models', {})
                if model_name in models:
                    model_info = models[model_name]
                    coordinator.model_strategy[key] = model_info
        
        # CRITICAL FIX 1.2: For T1a (Monolith Only), disable Swarm completely
        if test_id == 'T1a':
            use_swarm = False  # Force disable Swarm for Monolith-only test
            use_fusion = False  # Force disable Fusion for Monolith-only test
        
        # Prepare parameters override
        params_override = {
            'use_swarm_analysis': use_swarm,
            'use_monolith_analysis': use_monolith,
            'use_fusion': use_fusion,
            'use_self_correction_loop': use_phase3,
            'use_post_processing': use_phase4,  # FIXED: Use correct parameter name
            'use_active_learning': False,  # Always disabled for tests
            'iou_match_threshold': 0.5,  # CRITICAL FIX 3.1: Increase IoU threshold for better precision
        }
        
        # Create output directory for this test
        output_dir = project_root / "outputs" / "phase1_tests" / test_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        start_time = datetime.now()
        logger.info(f"Starting analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        result = coordinator.process(
            image_path=str(image_path),
            output_dir=str(output_dir),
            params_override=params_override
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Analysis completed in {duration:.2f} seconds")
        
        # Extract results
        if hasattr(result, 'elements'):
            elements = result.elements
            connections = result.connections
            quality_score = getattr(result, 'quality_score', 0.0)
        elif isinstance(result, dict):
            elements = result.get('elements', [])
            connections = result.get('connections', [])
            quality_score = result.get('quality_score', 0.0)
        else:
            elements = []
            connections = []
            quality_score = 0.0
        
        logger.info(f"Results: {len(elements)} elements, {len(connections)} connections, quality_score: {quality_score:.2f}")
        
        # Validate against truth
        validation = validate_against_truth(result, truth_data)
        
        # Save results
        result_file = output_dir / "test_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_id': test_id,
                'test_name': test_name,
                'configuration': {
                    'use_swarm_analysis': use_swarm,
                    'use_monolith_analysis': use_monolith,
                    'use_fusion': use_fusion,
                    'use_phase3': use_phase3,
                    'use_phase4': use_phase4
                },
                'results': {
                    'elements_count': len(elements),
                    'connections_count': len(connections),
                    'quality_score': quality_score
                },
                'validation': validation,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {result_file}")
        
        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"TEST SUMMARY: {test_id} - {test_name}")
        logger.info("=" * 80)
        logger.info(f"Elements: {validation['elements']['result_count']} detected, {validation['elements']['truth_count']} expected")
        logger.info(f"  - Matched: {validation['elements']['matched']}")
        logger.info(f"  - Missed: {validation['elements']['missed']}")
        logger.info(f"  - Hallucinated: {validation['elements']['hallucinated']}")
        logger.info(f"  - Precision: {validation['elements']['precision']:.2%}")
        logger.info(f"  - Recall: {validation['elements']['recall']:.2%}")
        logger.info(f"  - F1: {validation['elements']['f1']:.2%}")
        logger.info("")
        logger.info(f"Connections: {validation['connections']['result_count']} detected, {validation['connections']['truth_count']} expected")
        logger.info(f"  - Matched: {validation['connections']['matched']}")
        logger.info(f"  - Missed: {validation['connections']['missed']}")
        logger.info(f"  - Hallucinated: {validation['connections']['hallucinated']}")
        logger.info(f"  - Precision: {validation['connections']['precision']:.2%}")
        logger.info(f"  - Recall: {validation['connections']['recall']:.2%}")
        logger.info(f"  - F1: {validation['connections']['f1']:.2%}")
        logger.info("")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Quality Score: {quality_score:.2f}")
        logger.info("=" * 80)
        logger.info("")
        
        return {
            'test_id': test_id,
            'test_name': test_name,
            'success': True,
            'duration': duration,
            'quality_score': quality_score,
            'validation': validation,
            'result_file': str(result_file)
        }
        
    except Exception as e:
        logger.error(f"Test {test_id} failed: {e}", exc_info=True)
        return {
            'test_id': test_id,
            'test_name': test_name,
            'success': False,
            'error': str(e)
        }


def main():
    """Run all Phase 1 stability tests."""
    logger.info("=" * 80)
    logger.info("PHASE 1: STABILITÄTS- & ABLATIONSTESTS (Einfaches P&ID)")
    logger.info("=" * 80)
    logger.info("")
    
    # Find test image
    test_image = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
    if not test_image.exists():
        # Try alternative locations
        test_image = project_root / "training_data" / "organized_tests" / "simple_pids" / "Einfaches P&I.png"
        if not test_image.exists():
            test_image = project_root / "test_images" / "Einfaches P&I.png"
            if not test_image.exists():
                test_image = project_root / "Einfaches P&I.png"
                if not test_image.exists():
                    logger.error(f"Test image not found. Tried multiple locations.")
                    logger.error("Please place 'Einfaches P&I.png' in training_data/simple_pids/ or test_images/")
                    sys.exit(1)
    
    # Find truth data
    truth_file = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
    if not truth_file.exists():
        # Try alternative locations
        truth_file = project_root / "training_data" / "organized_tests" / "simple_pids" / "Einfaches P&I_truth.json"
        if not truth_file.exists():
            truth_file = project_root / "test_images" / "Einfaches P&I_truth.json"
            if not truth_file.exists():
                truth_file = project_root / "Einfaches P&I_truth.json"
                if not truth_file.exists():
                    logger.error(f"Truth data not found. Tried multiple locations.")
                    logger.error("Please place 'Einfaches P&I_truth.json' in training_data/simple_pids/ or test_images/")
                    sys.exit(1)
    
    logger.info(f"Test image: {test_image}")
    logger.info(f"Truth data: {truth_file}")
    logger.info("")
    
    # Initialize config service
    config_service = ConfigService()
    
    # Define tests - Baseline-Bestätigung für Monolith Pro + ID-Normalisierung
    tests = [
        {
            'test_id': 'T1a',
            'test_name': 'Monolith Pro Raw',
            'use_swarm': False,
            'use_monolith': True,
            'use_fusion': False,
            'use_phase3': False,
            'use_phase4': False  # Keine ID-Normalisierung
        },
        {
            'test_id': 'T4',
            'test_name': 'Monolith Pro + P4',
            'use_swarm': False,
            'use_monolith': True,
            'use_fusion': False,
            'use_phase3': False,
            'use_phase4': True  # Mit ID-Normalisierung (Phase 4)
        }
    ]
    
    # Run all tests
    results = []
    for test_config in tests:
        result = run_test(
            test_name=test_config['test_name'],
            test_id=test_config['test_id'],
            image_path=test_image,
            truth_path=truth_file,
            config_service=config_service,
            use_swarm=test_config['use_swarm'],
            use_monolith=test_config['use_monolith'],
            use_fusion=test_config['use_fusion'],
            use_phase3=test_config['use_phase3'],
            use_phase4=test_config['use_phase4']
        )
        results.append(result)
        
        # Small delay between tests
        import time
        time.sleep(2)
    
    # Generate summary report
    summary_file = project_root / "outputs" / "phase1_tests" / "summary_report.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'test_image': str(test_image),
        'truth_file': str(truth_file),
        'tests': results,
        'summary': {
            'total_tests': len(results),
            'successful_tests': len([r for r in results if r.get('success', False)]),
            'failed_tests': len([r for r in results if not r.get('success', False)])
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1 TESTS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Summary report saved to: {summary_file}")
    logger.info("")
    logger.info("Test Results:")
    for result in results:
        if result.get('success', False):
            validation = result.get('validation', {})
            logger.info(f"  {result['test_id']} ({result['test_name']}): "
                       f"Elements F1={validation['elements']['f1']:.2%}, "
                       f"Connections F1={validation['connections']['f1']:.2%}, "
                       f"Duration={result.get('duration', 0):.2f}s")
        else:
            logger.info(f"  {result['test_id']} ({result['test_name']}): FAILED - {result.get('error', 'Unknown error')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

