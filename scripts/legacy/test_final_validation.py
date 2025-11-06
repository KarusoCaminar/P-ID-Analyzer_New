#!/usr/bin/env python3
"""
Final Validation Test Script - Tests Simple PID and Uni-Bild WITHOUT Truth Data.

This script:
1. Tests Simple PID without truth data
2. Tests one Uni-Bild (page_1) without truth data
3. Evaluates results independently
4. Reports findings
"""

import os
import sys
import logging
import json
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DummyProgressCallback:
    """Dummy progress callback for testing."""
    def update_progress(self, progress: int, message: str):
        logger.info(f"[Progress {progress}%] {message}")
    
    def update_phase(self, phase: str, message: str):
        logger.info(f"[Phase: {phase}] {message}")
    
    def log_message(self, level: str, message: str):
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)


def run_analysis_no_truth(image_path: Path, output_dir: Path):
    """Run analysis on a single image WITHOUT truth data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"=== Analyzing (NO TRUTH MODE): {image_path.name} ===")
    logger.info(f"{'='*80}\n")
    
    try:
        # Initialize services
        config_service = ConfigService()
        config = config_service.get_raw_config()
        
        # Get GCP Project ID
        gcp_project_id = os.getenv("GCP_PROJECT_ID")
        if not gcp_project_id:
            logger.error("GCP_PROJECT_ID environment variable not set. Please set it in your .env file.")
            return None
        
        # Initialize LLM Client (korrekte Signatur)
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
        
        # Run analysis WITHOUT truth data
        result = coordinator.process(
            str(image_path),
            str(output_dir),
            params_override={
                'truth_data': None,  # Explicitly no truth data
                'iteration': 1
            }
        )
        
        if result:
            logger.info(f"\n✅ Analysis complete for {image_path.name}")
            return result
        else:
            logger.error(f"❌ Analysis failed for {image_path.name}")
            return None
            
    except Exception as e:
        logger.error(f"Error analyzing {image_path.name}: {e}", exc_info=True)
        return None


def evaluate_results(result: dict, image_name: str):
    """Evaluate analysis results independently."""
    logger.info(f"\n{'='*80}")
    logger.info(f"=== Evaluating Results: {image_name} ===")
    logger.info(f"{'='*80}\n")
    
    if not result:
        logger.error("No result to evaluate")
        return
    
    # Extract key metrics (handle both dict and object)
    if hasattr(result, 'final_ai_data'):
        elements = result.final_ai_data.get('elements', []) if result.final_ai_data else []
        connections = result.final_ai_data.get('connections', []) if result.final_ai_data else []
        quality_score = result.quality_score if hasattr(result, 'quality_score') else 0.0
    else:
        elements = result.get('final_ai_data', {}).get('elements', [])
        connections = result.get('final_ai_data', {}).get('connections', [])
        quality_score = result.get('quality_score', 0.0)
    
    logger.info(f"📊 QUALITY SCORE: {quality_score:.2f}%")
    logger.info(f"📊 ELEMENTS DETECTED: {len(elements)}")
    logger.info(f"📊 CONNECTIONS DETECTED: {len(connections)}")
    
    # Analyze element types
    type_counts = {}
    confidence_scores = []
    
    for el in elements:
        el_dict = el if isinstance(el, dict) else el.model_dump() if hasattr(el, 'model_dump') else el.__dict__ if hasattr(el, '__dict__') else {}
        el_type = el_dict.get('type', 'Unknown')
        confidence = el_dict.get('confidence', 0.0)
        
        type_counts[el_type] = type_counts.get(el_type, 0) + 1
        confidence_scores.append(confidence)
    
    logger.info(f"\n📈 ELEMENT TYPE DISTRIBUTION:")
    for el_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {el_type}: {count}")
    
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        min_confidence = min(confidence_scores)
        max_confidence = max(confidence_scores)
        logger.info(f"\n📈 CONFIDENCE STATISTICS:")
        logger.info(f"  - Average: {avg_confidence:.2f}")
        logger.info(f"  - Min: {min_confidence:.2f}")
        logger.info(f"  - Max: {max_confidence:.2f}")
    
    # Analyze connections
    connection_confidences = []
    for conn in connections:
        conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ if hasattr(conn, '__dict__') else {}
        confidence = conn_dict.get('confidence', 0.0)
        connection_confidences.append(confidence)
    
    if connection_confidences:
        avg_conn_confidence = sum(connection_confidences) / len(connection_confidences)
        logger.info(f"\n📈 CONNECTION CONFIDENCE:")
        logger.info(f"  - Average: {avg_conn_confidence:.2f}")
    
    # Check for issues
    issues = []
    
    # Low confidence elements
    low_confidence_elements = [el for el in elements 
                              if (el if isinstance(el, dict) else el.model_dump() if hasattr(el, 'model_dump') else el.__dict__ if hasattr(el, '__dict__') else {}).get('confidence', 0.0) < 0.7]
    if low_confidence_elements:
        issues.append(f"⚠️  {len(low_confidence_elements)} elements with confidence < 0.7")
    
    # Unknown types
    unknown_types = [el for el in elements 
                    if (el if isinstance(el, dict) else el.model_dump() if hasattr(el, 'model_dump') else el.__dict__ if hasattr(el, '__dict__') else {}).get('type', 'Unknown') == 'Unknown']
    if unknown_types:
        issues.append(f"⚠️  {len(unknown_types)} elements with type 'Unknown'")
    
    # Isolated elements (no connections)
    element_ids = set()
    for el in elements:
        el_dict = el if isinstance(el, dict) else el.model_dump() if hasattr(el, 'model_dump') else el.__dict__ if hasattr(el, '__dict__') else {}
        el_id = el_dict.get('id', '')
        if el_id:
            element_ids.add(el_id)
    
    connected_element_ids = set()
    for conn in connections:
        conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ if hasattr(conn, '__dict__') else {}
        connected_element_ids.add(conn_dict.get('from_id', ''))
        connected_element_ids.add(conn_dict.get('to_id', ''))
    
    isolated_elements = element_ids - connected_element_ids
    if isolated_elements:
        issues.append(f"⚠️  {len(isolated_elements)} isolated elements (no connections)")
    
    if issues:
        logger.info(f"\n⚠️  ISSUES FOUND:")
        for issue in issues:
            logger.info(f"  {issue}")
    else:
        logger.info(f"\n✅ NO ISSUES FOUND - All checks passed!")
    
    logger.info(f"\n{'='*80}\n")


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("FINAL VALIDATION TEST - Simple PID and Uni-Bild")
    logger.info("=" * 80)
    
    # Find test images
    project_root = Path(__file__).parent.parent
    
    # Simple PID
    simple_pid_paths = [
        project_root / "training_data" / "simple_pids" / "Einfaches P&I.png",
        project_root / "training_data" / "organized_tests" / "simple_pids" / "Einfaches P&I.png"
    ]
    simple_pid_path = None
    for path in simple_pid_paths:
        if path.exists():
            simple_pid_path = path
            break
    
    # Uni-Bild (page_1)
    uni_bild_paths = [
        project_root / "training_data" / "complex_pids" / "page_1_original.png",
        project_root / "training_data" / "organized_tests" / "complex_pids" / "page_1_original.png"
    ]
    uni_bild_path = None
    for path in uni_bild_paths:
        if path.exists():
            uni_bild_path = path
            break
    
    if not simple_pid_path:
        logger.error("Simple PID image not found!")
        return
    
    if not uni_bild_path:
        logger.error("Uni-Bild (page_1) image not found!")
        return
    
    # Create output directory
    output_dir = project_root / "outputs" / "final_validation_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test Simple PID
    simple_result = run_analysis_no_truth(simple_pid_path, output_dir / "simple_pid")
    if simple_result:
        evaluate_results(simple_result, "Simple PID")
    
    # Test Uni-Bild
    uni_result = run_analysis_no_truth(uni_bild_path, output_dir / "uni_bild")
    if uni_result:
        evaluate_results(uni_result, "Uni-Bild (page_1)")
    
    # Summary
    logger.info("=" * 80)
    logger.info("FINAL VALIDATION TEST COMPLETE")
    logger.info("=" * 80)
    
    if simple_result and uni_result:
        logger.info("✅ Both tests completed successfully!")
    elif simple_result:
        logger.warning("⚠️  Simple PID test completed, but Uni-Bild test failed")
    elif uni_result:
        logger.warning("⚠️  Uni-Bild test completed, but Simple PID test failed")
    else:
        logger.error("❌ Both tests failed")


if __name__ == "__main__":
    main()

