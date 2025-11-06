#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Skript für Uni-Bilder 2-4.
Analysiert die Bilder und wertet die Ergebnisse aus.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for stdout
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        elif hasattr(sys.stdout, 'buffer'):
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass  # Fallback to default encoding

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_truth_data(truth_path: Path) -> Dict[str, Any]:
    """Load truth data from JSON file."""
    try:
        with open(truth_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load truth data from {truth_path}: {e}")
        return {}


def analyze_uni_images():
    """Analyze Uni-Bilder 2-4 and evaluate results."""
    
    logger.info("="*80)
    logger.info("UNI-BILDER 2-4 TEST-SKRIPT")
    logger.info("="*80)
    
    # Initialize services
    config_service = ConfigService()
    config = config_service.get_config()
    
    # Get project ID and location from environment or config
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT_ID') or config.get('gcp_project_id', '')
    location = os.getenv('GCP_LOCATION') or config.get('gcp_location', 'us-central1')
    
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID environment variable not set!")
        logger.error("Please set it in your .env file or environment variables.")
        return
    
    logger.info(f"Using Project ID: {project_id}")
    logger.info(f"Using Location: {location}")
    
    try:
        llm_client = LLMClient(
            project_id=project_id,
            default_location=location,
            config=config_service.get_raw_config()
        )
        logger.info("LLMClient initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
        return
    
    try:
        # Get paths from config service
        element_type_list_path = config_service.get_path("element_type_list") or project_root / "element_type_list.json"
        learning_db_path = config_service.get_path("learning_db") or project_root / "learning_db.json"
        
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list_path),
            learning_db_path=str(learning_db_path),
            llm_handler=llm_client,
            config=config_service.get_raw_config()
        )
        logger.info("KnowledgeManager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize KnowledgeManager: {e}", exc_info=True)
        return
    
    try:
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service
        )
        logger.info("PipelineCoordinator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PipelineCoordinator: {e}", exc_info=True)
        return
    
    # Define test images
    base_path = project_root / "training_data" / "complex_pids"
    test_images = [
        {
            'image': base_path / "page_2_original.png",
            'truth': base_path / "page_2_original_truth_cgm.json",
            'name': 'Uni-Bild 2'
        },
        {
            'image': base_path / "page_3_original.png",
            'truth': base_path / "page_3_original_truth_cgm.json",
            'name': 'Uni-Bild 3'
        },
        {
            'image': base_path / "page_4_original.png",
            'truth': base_path / "page_4_original_truth_cgm.json",
            'name': 'Uni-Bild 4'
        }
    ]
    
    results = []
    
    for test_case in test_images:
        image_path = test_case['image']
        truth_path = test_case['truth']
        name = test_case['name']
        
        logger.info("\n" + "="*80)
        logger.info(f"=== Analyzing {name} ===")
        logger.info("="*80)
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            results.append({
                'name': name,
                'error': f"Image not found: {image_path}",
                'timestamp': datetime.now().isoformat()
            })
            continue
        
        # Load truth data if available
        truth_data = load_truth_data(truth_path) if truth_path.exists() else None
        if truth_data:
            logger.info(f"Truth data loaded: {len(truth_data.get('elements', []))} elements")
        else:
            logger.info("No truth data available - running without validation")
        
        # Run analysis
        try:
            result = coordinator.process(
                image_path=str(image_path),
                output_dir=None
            )
            
            # Extract quality metrics
            if hasattr(result, 'quality_score'):
                quality_score = result.quality_score
                elements = result.elements if hasattr(result, 'elements') else []
                connections = result.connections if hasattr(result, 'connections') else []
                kpis = getattr(result, 'final_kpis', {})
            elif isinstance(result, dict):
                quality_score = result.get('quality_score', 0.0)
                elements = result.get('elements', [])
                connections = result.get('connections', [])
                kpis = result.get('final_kpis', {})
            else:
                quality_score = 0.0
                elements = []
                connections = []
                kpis = {}
            
            test_result = {
                'name': name,
                'image_path': str(image_path),
                'quality_score': quality_score,
                'num_elements': len(elements),
                'num_connections': len(connections),
                'kpis': kpis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add truth comparison if available
            if truth_data:
                truth_elements = truth_data.get('elements', [])
                truth_connections = truth_data.get('connections', [])
                
                test_result['truth_comparison'] = {
                    'truth_elements': len(truth_elements),
                    'truth_connections': len(truth_connections),
                    'element_delta': len(elements) - len(truth_elements),
                    'connection_delta': len(connections) - len(truth_connections)
                }
            
            results.append(test_result)
            
            logger.info(f"✓ {name}: Score={quality_score:.2f}%, Elements={len(elements)}, Connections={len(connections)}")
            
        except Exception as e:
            logger.error(f"Error analyzing {name}: {e}", exc_info=True)
            results.append({
                'name': name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # Save results
    results_path = project_root / "outputs" / "uni_test_results.json"
    results_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST ZUSAMMENFASSUNG")
    print("="*80)
    for result in results:
        if 'error' in result:
            try:
                print(f"❌ {result['name']}: ERROR - {result['error']}")
            except UnicodeEncodeError:
                print(f"[ERROR] {result['name']}: ERROR - {result['error']}")
        else:
            try:
                print(f"✓ {result['name']}: Score={result['quality_score']:.2f}%, "
                      f"Elements={result['num_elements']}, Connections={result['num_connections']}")
            except UnicodeEncodeError:
                print(f"[OK] {result['name']}: Score={result['quality_score']:.2f}%, "
                      f"Elements={result['num_elements']}, Connections={result['num_connections']}")
            
            if 'truth_comparison' in result:
                tc = result['truth_comparison']
                try:
                    print(f"  → Truth: {tc['truth_elements']} elements, {tc['truth_connections']} connections")
                    print(f"  → Delta: {tc['element_delta']:+d} elements, {tc['connection_delta']:+d} connections")
                except UnicodeEncodeError:
                    print(f"  -> Truth: {tc['truth_elements']} elements, {tc['truth_connections']} connections")
                    print(f"  -> Delta: {tc['element_delta']:+d} elements, {tc['connection_delta']:+d} connections")
    
    print("="*80)
    
    # Calculate overall statistics
    successful = [r for r in results if 'error' not in r]
    if successful:
        avg_score = sum(r['quality_score'] for r in successful) / len(successful)
        total_elements = sum(r['num_elements'] for r in successful)
        total_connections = sum(r['num_connections'] for r in successful)
        
        print(f"\nGesamtstatistik:")
        print(f"  • Erfolgreiche Tests: {len(successful)}/{len(results)}")
        print(f"  • Durchschnittlicher Score: {avg_score:.2f}%")
        print(f"  • Gesamt Elemente: {total_elements}")
        print(f"  • Gesamt Verbindungen: {total_connections}")
    
    print("="*80)


if __name__ == "__main__":
    analyze_uni_images()

