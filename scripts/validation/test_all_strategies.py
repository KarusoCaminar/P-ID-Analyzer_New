"""
Test All Strategies - Comprehensive test of all strategies on multiple images.

Tests:
1. Simple PID diagram
2. Original picture 1 (Uni-1)
3. All strategies: simple_whole_image, hybrid_fusion
4. Internal KPIs (without ground truth)
5. Focus on fusion strategy validation
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file automatically
try:
    from src.utils.env_loader import load_env_automatically
    if load_env_automatically():
        print(f"[OK] .env Datei automatisch geladen")
    else:
        print(f"[WARNING] .env Datei nicht gefunden")
except (ImportError, Exception) as e:
    try:
        from dotenv import load_dotenv
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            print(f"[OK] .env Datei geladen: {env_file}")
    except ImportError:
        pass

from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.evaluation.kpi_calculator import KPICalculator
from src.utils.output_structure_manager import OutputStructureManager
from scripts.utils.live_log_monitor import LiveLogMonitor

# Test images
TEST_IMAGE_SIMPLE = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
TEST_IMAGE_UNI = project_root / "training_data" / "complex_pids" / "Verfahrensfließbild_Uni.png"

# Strategies to test
STRATEGIES_TO_TEST = [
    "simple_whole_image",
    "hybrid_fusion"
]

# Setup logging
LoggingService.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyTester:
    """Test all strategies on multiple images."""
    
    def __init__(self):
        """Initialize tester."""
        self.config_service = None
        self.llm_client = None
        self.knowledge_manager = None
        self.coordinator = None
        self.results = []
        
    def setup_services(self):
        """Setup all services."""
        logger.info("=" * 80)
        logger.info("SETTING UP SERVICES")
        logger.info("=" * 80)
        
        try:
            # Config Service
            logger.info("[1/4] Initializing Config Service...")
            self.config_service = ConfigService()
            logger.info("[OK] Config Service initialized")
            
            # LLM Client
            logger.info("[2/4] Initializing LLM Client...")
            import os
            config = self.config_service.get_raw_config()
            project_id = os.getenv("GCP_PROJECT_ID")
            location = os.getenv("GCP_LOCATION", "europe-west3")
            
            if not project_id:
                raise ValueError("GCP_PROJECT_ID nicht gesetzt!")
            
            self.llm_client = LLMClient(project_id, location, config)
            logger.info("[OK] LLM Client initialized")
            
            # Knowledge Manager
            logger.info("[3/4] Initializing Knowledge Manager...")
            element_type_list = self.config_service.get_path('element_type_list')
            learning_db = self.config_service.get_path('learning_db')
            
            self.knowledge_manager = KnowledgeManager(
                element_type_list_path=str(element_type_list),
                learning_db_path=str(learning_db),
                llm_handler=self.llm_client,
                config=config
            )
            logger.info("[OK] Knowledge Manager initialized")
            
            # Pipeline Coordinator
            logger.info("[4/4] Initializing Pipeline Coordinator...")
            self.coordinator = PipelineCoordinator(
                llm_client=self.llm_client,
                knowledge_manager=self.knowledge_manager,
                config_service=self.config_service
            )
            logger.info("[OK] Pipeline Coordinator initialized")
            
            logger.info("=" * 80)
            logger.info("ALL SERVICES INITIALIZED SUCCESSFULLY")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"[FAIL] Error initializing services: {e}", exc_info=True)
            raise
    
    def test_strategy_on_image(
        self,
        image_path: Path,
        strategy: str,
        image_name: str
    ) -> Dict[str, Any]:
        """Test a strategy on an image."""
        logger.info("=" * 80)
        logger.info(f"TESTING: {strategy} on {image_name}")
        logger.info("=" * 80)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return {
                'strategy': strategy,
                'image': image_name,
                'success': False,
                'error': f'Image not found: {image_path}'
            }
        
        # Get strategy config
        config = self.config_service.get_config()
        strategies = config.strategies
        
        if strategy not in strategies:
            logger.error(f"Strategy '{strategy}' not found in config")
            return {
                'strategy': strategy,
                'image': image_name,
                'success': False,
                'error': f'Strategy not found: {strategy}'
            }
        
        strategy_config = strategies[strategy]
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = project_root / "outputs" / "strategy_tests"
        output_dir = output_base / f"{strategy}_{image_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure structured output
        from src.utils.output_structure_manager import ensure_output_structure
        ensure_output_structure(output_dir)
        
        # Setup log file
        log_file = output_dir / "logs" / "test.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Start live log monitoring
        log_monitor = LiveLogMonitor(str(log_file))
        log_monitor.start()
        
        try:
            # Prepare parameters (strategy_config is already a dict from config)
            params_override = {
                **strategy_config,  # Include all strategy config parameters
                'test_name': f"{strategy}_{image_name}_test",
                'test_description': f"Strategy test: {strategy} on {image_name}"
            }
            
            # Run analysis
            start_time = time.time()
            logger.info(f"\nStarting analysis at {datetime.now().strftime('%H:%M:%S')}...")
            
            result = self.coordinator.process(
                image_path=str(image_path),
                output_dir=str(output_dir),
                params_override=params_override
            )
            
            end_time = time.time()
            duration_minutes = (end_time - start_time) / 60.0
            
            logger.info(f"\n[OK] Analysis completed in {duration_minutes:.2f} minutes")
            
            # Convert result
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            elif hasattr(result, 'dict'):
                result_dict = result.dict()
            else:
                result_dict = result if isinstance(result, dict) else {
                    'elements': getattr(result, 'elements', []),
                    'connections': getattr(result, 'connections', [])
                }
            
            elements = result_dict.get('elements', [])
            connections = result_dict.get('connections', [])
            
            logger.info(f"  Found Elements: {len(elements)}")
            logger.info(f"  Found Connections: {len(connections)}")
            
            # Calculate internal KPIs (without ground truth)
            logger.info("\nCalculating internal KPIs (without ground truth)...")
            kpi_calculator = KPICalculator()
            kpis = kpi_calculator.calculate_comprehensive_kpis(
                analysis_data=result_dict,
                truth_data=None  # No ground truth - use internal KPIs
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("INTERNAL KPIs (No Ground Truth)")
            logger.info("=" * 80)
            logger.info(f"Quality Score: {kpis.get('quality_score', 0.0):.2f}")
            logger.info(f"Total Elements: {kpis.get('total_elements', 0)}")
            logger.info(f"Total Connections: {kpis.get('total_connections', 0)}")
            logger.info(f"Graph Density: {kpis.get('graph_density', 0.0):.4f}")
            logger.info(f"Connected Elements: {kpis.get('connected_elements', 0)}")
            logger.info(f"Isolated Elements: {kpis.get('isolated_elements', 0)}")
            logger.info(f"Num Cycles: {kpis.get('num_cycles', 0)}")
            logger.info(f"Max Centrality: {kpis.get('max_centrality', 0.0):.4f}")
            logger.info(f"Avg Element Confidence: {kpis.get('avg_element_confidence', 0.0):.3f}")
            logger.info(f"Avg Connection Confidence: {kpis.get('avg_connection_confidence', 0.0):.3f}")
            logger.info("=" * 80)
            
            # Save results
            result_file = output_dir / "data" / "test_result.json"
            
            # CRITICAL FIX: Recursively convert datetime objects to strings
            def convert_datetime_to_string(obj):
                """Recursively convert datetime objects to ISO format strings."""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime_to_string(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime_to_string(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_datetime_to_string(item) for item in obj)
                else:
                    return obj
            
            test_result = {
                'strategy': strategy,
                'image': image_name,
                'image_path': str(image_path),
                'duration_minutes': duration_minutes,
                'timestamp': datetime.now().isoformat(),
                'result': result_dict,
                'kpis': kpis,
                'success': True
            }
            
            # Convert all datetime objects in test_result
            test_result = convert_datetime_to_string(test_result)
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n[OK] Results saved: {result_file}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error during test: {e}", exc_info=True)
            return {
                'strategy': strategy,
                'image': image_name,
                'success': False,
                'error': str(e)
            }
        finally:
            # Stop live log monitoring
            log_monitor.flush()
            log_monitor.stop()
    
    def run_all_tests(self):
        """Run all tests."""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE STRATEGY TEST")
        logger.info("=" * 80)
        logger.info(f"Strategies: {STRATEGIES_TO_TEST}")
        logger.info(f"Images: Simple PID, Uni-1")
        logger.info("=" * 80)
        
        # Setup services
        self.setup_services()
        
        # Test images
        test_images = [
            ("Simple PID", TEST_IMAGE_SIMPLE),
            ("Uni-1", TEST_IMAGE_UNI)
        ]
        
        # Run all tests
        for image_name, image_path in test_images:
            for strategy in STRATEGIES_TO_TEST:
                try:
                    result = self.test_strategy_on_image(
                        image_path=image_path,
                        strategy=strategy,
                        image_name=image_name
                    )
                    self.results.append(result)
                    
                    # Small delay between tests
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error testing {strategy} on {image_name}: {e}", exc_info=True)
                    self.results.append({
                        'strategy': strategy,
                        'image': image_name,
                        'success': False,
                        'error': str(e)
                    })
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate summary report of all tests."""
        logger.info("=" * 80)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("=" * 80)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'successful_tests': sum(1 for r in self.results if r.get('success', False)),
            'failed_tests': sum(1 for r in self.results if not r.get('success', False)),
            'results': self.results
        }
        
        # Save summary
        output_base = project_root / "outputs" / "strategy_tests"
        summary_file = output_base / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL FIX: Recursively convert datetime objects to strings
        def convert_datetime_to_string(obj):
            """Recursively convert datetime objects to ISO format strings."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime_to_string(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime_to_string(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_datetime_to_string(item) for item in obj)
            else:
                return obj
        
        # Convert all datetime objects in summary
        summary = convert_datetime_to_string(summary)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n[OK] Summary saved: {summary_file}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Successful: {summary['successful_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info("\nResults:")
        
        for result in self.results:
            strategy = result.get('strategy', 'unknown')
            image = result.get('image', 'unknown')
            success = result.get('success', False)
            
            if success:
                kpis = result.get('kpis', {})
                quality_score = kpis.get('quality_score', 0.0)
                elements = kpis.get('total_elements', 0)
                connections = kpis.get('total_connections', 0)
                duration = result.get('duration_minutes', 0.0)
                
                logger.info(f"  ✅ {strategy} on {image}:")
                logger.info(f"     Quality Score: {quality_score:.2f}")
                logger.info(f"     Elements: {elements}, Connections: {connections}")
                logger.info(f"     Duration: {duration:.2f} minutes")
            else:
                error = result.get('error', 'Unknown error')
                logger.info(f"  ❌ {strategy} on {image}: {error}")
        
        logger.info("=" * 80)
        
        # Print fusion strategy results specifically
        logger.info("\n" + "=" * 80)
        logger.info("FUSION STRATEGY RESULTS (Focus)")
        logger.info("=" * 80)
        
        fusion_results = [r for r in self.results if r.get('strategy') == 'hybrid_fusion']
        for result in fusion_results:
            image = result.get('image', 'unknown')
            success = result.get('success', False)
            
            if success:
                kpis = result.get('kpis', {})
                quality_score = kpis.get('quality_score', 0.0)
                elements = kpis.get('total_elements', 0)
                connections = kpis.get('total_connections', 0)
                graph_density = kpis.get('graph_density', 0.0)
                connected_elements = kpis.get('connected_elements', 0)
                duration = result.get('duration_minutes', 0.0)
                
                logger.info(f"\n{image}:")
                logger.info(f"  Quality Score: {quality_score:.2f}")
                logger.info(f"  Elements: {elements}")
                logger.info(f"  Connections: {connections}")
                logger.info(f"  Graph Density: {graph_density:.4f}")
                logger.info(f"  Connected Elements: {connected_elements}")
                logger.info(f"  Duration: {duration:.2f} minutes")
                logger.info(f"  ✅ Fusion strategy works!")
            else:
                error = result.get('error', 'Unknown error')
                logger.info(f"\n{image}:")
                logger.info(f"  ❌ Failed: {error}")
        
        logger.info("=" * 80)


def main():
    """Main function."""
    tester = StrategyTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        logger.info("\n[INTERRUPTED] Test interrupted by user")
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

