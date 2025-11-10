"""
Compare different fusion strategies to find the best approach.

Strategies to test:
1. Strategy A (Separation): Swarm = Elements only, Monolith = Connections only
2. Strategy B (Full Redundancy): Both find Elements + Connections
3. Option 1 (Local/Global): Swarm = Elements + Local Connections, Monolith = Global Connections
4. Option 2 (ID Override): Monolith can override IDs (Whole Image)
5. Option 3 (Current): Full Redundancy with ID Correction (already implemented)
"""

import sys
import json
import os
import logging
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Any, List, Optional
import copy

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.json_encoder import json_dump_safe
from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.evaluation.kpi_calculator import KPICalculator
from src.utils.output_structure_manager import OutputStructureManager

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

# Setup Logging
LoggingService.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)

# Test Configuration
TEST_IMAGE_COMPLEX = project_root / "training_data" / "complex_pids" / "page_1_original.png"
TEST_GROUND_TRUTH_COMPLEX = project_root / "training_data" / "complex_pids" / "page_1_original_truth_cgm.json"
OUTPUT_BASE_DIR = project_root / "outputs" / "strategy_comparison"


class StrategyComparison:
    """Compare different fusion strategies."""
    
    def __init__(self, test_image: Path, test_truth: Optional[Path] = None):
        """Initialize strategy comparison."""
        self.test_image = test_image
        self.test_truth = test_truth
        
        # Output manager
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_manager = OutputStructureManager(OUTPUT_BASE_DIR.parent, f"strategy_comparison_{timestamp}")
        self.output_dir = self.output_manager.get_output_dir()
        
        # Log file
        self.log_file = self.output_manager.get_log_file("comparison.log")
        LoggingService.setup_logging(
            log_level=logging.INFO,
            log_file=str(self.log_file)
        )
        self.logger = logging.getLogger(__name__)
        
        # Services
        self.config_service = None
        self.llm_client = None
        self.knowledge_manager = None
        self.coordinator = None
        
        # Results
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def setup_services(self):
        """Initialize all services."""
        self.logger.info("=" * 80)
        self.logger.info("INITIALIZING SERVICES")
        self.logger.info("=" * 80)
        
        try:
            # Config service
            self.logger.info("[1/4] Loading configuration...")
            self.config_service = ConfigService()
            config = self.config_service.get_raw_config()
            self.logger.info("[OK] Configuration loaded")
            
            # GCP credentials
            project_id = os.getenv("GCP_PROJECT_ID")
            location = os.getenv("GCP_LOCATION", "us-central1")
            
            if not project_id:
                raise ValueError("GCP_PROJECT_ID nicht gesetzt!")
            
            # LLM Client
            self.logger.info("[2/4] Initializing LLM Client...")
            self.llm_client = LLMClient(project_id, location, config)
            self.logger.info("[OK] LLM Client initialized")
            
            # Knowledge Manager
            self.logger.info("[3/4] Initializing Knowledge Manager...")
            element_type_list = self.config_service.get_path('element_type_list')
            learning_db = self.config_service.get_path('learning_db')
            
            start_km = time.time()
            self.knowledge_manager = KnowledgeManager(
                element_type_list_path=str(element_type_list),
                learning_db_path=str(learning_db),
                llm_handler=self.llm_client,
                config=config
            )
            km_time = time.time() - start_km
            self.logger.info(f"[OK] Knowledge Manager initialized in {km_time:.2f} seconds")
            
            # Pipeline Coordinator (will be recreated for each strategy)
            self.logger.info("[4/4] Services initialized")
            
            self.logger.info("=" * 80)
            self.logger.info("ALL SERVICES INITIALIZED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"[FAIL] Error initializing services: {e}", exc_info=True)
            raise
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get strategy configuration with custom overrides."""
        config = self.config_service.get_raw_config()
        strategies = config.get('strategies', {})
        base_strategy = strategies.get('hybrid_fusion', {})
        
        # Create custom strategy based on strategy_name
        strategy_config = copy.deepcopy(base_strategy)
        
        if strategy_name == "separation":
            # Strategy A: Separation (Swarm = Elements, Monolith = Connections)
            # Note: This requires prompt changes, so we'll use a modified config
            strategy_config.update({
                'use_swarm_analysis': True,
                'use_monolith_analysis': True,
                'use_fusion': True,
                'use_self_correction_loop': True,
                'monolith_whole_image': False,  # Use quadrants for speed
            })
            # Note: Prompts need to be modified (Swarm: elements only, Monolith: connections only)
            # This is a limitation - we'd need to modify prompts at runtime
            
        elif strategy_name == "full_redundancy":
            # Strategy B: Full Redundancy (Current implementation)
            strategy_config.update({
                'use_swarm_analysis': True,
                'use_monolith_analysis': True,
                'use_fusion': True,
                'use_self_correction_loop': True,
                'monolith_whole_image': True,  # Whole image for better connections
            })
            
        elif strategy_name == "local_global":
            # Option 1: Local/Global (Swarm = Elements + Local Connections, Monolith = Global Connections)
            strategy_config.update({
                'use_swarm_analysis': True,
                'use_monolith_analysis': True,
                'use_fusion': True,
                'use_self_correction_loop': True,
                'monolith_whole_image': True,  # Whole image for global connections
            })
            # Note: Prompts need to be modified (Swarm: local connections, Monolith: global connections)
            
        elif strategy_name == "id_override":
            # Option 2: ID Override (Monolith can override IDs)
            strategy_config.update({
                'use_swarm_analysis': True,
                'use_monolith_analysis': True,
                'use_fusion': True,
                'use_self_correction_loop': True,
                'monolith_whole_image': True,  # Whole image for ID correction
            })
            # Note: Prompts already support ID correction (current implementation)
            
        elif strategy_name == "current":
            # Option 3: Current (Full Redundancy with ID Correction)
            strategy_config.update({
                'use_swarm_analysis': True,
                'use_monolith_analysis': True,
                'use_fusion': True,
                'use_self_correction_loop': True,
                'monolith_whole_image': True,  # Whole image for better connections
            })
            # This is the current implementation
        
        return strategy_config
    
    def run_strategy_test(self, strategy_name: str) -> Dict[str, Any]:
        """Run a single strategy test."""
        self.logger.info("=" * 80)
        self.logger.info(f"TESTING STRATEGY: {strategy_name}")
        self.logger.info("=" * 80)
        
        # Get strategy config
        strategy_config = self.get_strategy_config(strategy_name)
        
        # Create coordinator for this strategy
        coordinator = PipelineCoordinator(
            llm_client=self.llm_client,
            knowledge_manager=self.knowledge_manager,
            config_service=self.config_service
        )
        
        # Prepare parameters
        params_override = {
            **strategy_config,
            'test_name': f"{strategy_name}_test",
            'test_description': f"Strategy comparison: {strategy_name}"
        }
        
        # Create output directory for this strategy
        strategy_output_dir = self.output_dir / strategy_name
        strategy_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        start_time = time.time()
        self.logger.info(f"\nStarting analysis at {datetime.now().strftime('%H:%M:%S')}...")
        
        try:
            result = coordinator.process(
                image_path=str(self.test_image),
                output_dir=str(strategy_output_dir),
                params_override=params_override
            )
            
            end_time = time.time()
            duration = (end_time - start_time) / 60.0
            
            self.logger.info(f"\n[OK] Analysis completed in {duration:.2f} minutes")
            
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
            
            self.logger.info(f"  Found Elements: {len(result_dict.get('elements', []))}")
            self.logger.info(f"  Found Connections: {len(result_dict.get('connections', []))}")
            
            # Calculate KPIs
            kpis = {}
            if self.test_truth and self.test_truth.exists():
                self.logger.info("\nCalculating KPIs...")
                with open(self.test_truth, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                
                kpi_calc = KPICalculator()
                kpis = kpi_calc.calculate_comprehensive_kpis(
                    analysis_data=result_dict,
                    truth_data=gt_data
                )
                
                self.logger.info("\n" + "=" * 80)
                self.logger.info(f"RESULTS - {strategy_name}")
                self.logger.info("=" * 80)
                self.logger.info(f"Element F1: {kpis.get('element_f1', 0.0):.4f}")
                self.logger.info(f"Element Precision: {kpis.get('element_precision', 0.0):.4f}")
                self.logger.info(f"Element Recall: {kpis.get('element_recall', 0.0):.4f}")
                self.logger.info(f"Connection F1: {kpis.get('connection_f1', 0.0):.4f}")
                self.logger.info(f"Quality Score: {kpis.get('quality_score', 0.0):.2f}")
                self.logger.info("=" * 80)
            
            # Save results
            result_file = self.output_manager.get_data_path(f"{strategy_name}_result.json")
            test_result = {
                'strategy': strategy_name,
                'image_path': str(self.test_image),
                'duration_minutes': duration,
                'timestamp': datetime.now().isoformat(),
                'result': result_dict,
                'kpis': kpis,
                'success': True
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json_dump_safe(test_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"\n[OK] Results saved: {result_file}")
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"\n[FAIL] TEST FAILED: {e}", exc_info=True)
            return {
                'strategy': strategy_name,
                'success': False,
                'error': str(e),
                'duration_minutes': (time.time() - start_time) / 60.0
            }
    
    def compare_strategies(self, strategy_names: List[str]):
        """Compare multiple strategies."""
        self.logger.info("=" * 80)
        self.logger.info("STRATEGY COMPARISON")
        self.logger.info("=" * 80)
        self.logger.info(f"Test Image: {self.test_image}")
        self.logger.info(f"Strategies to test: {', '.join(strategy_names)}")
        self.logger.info("=" * 80)
        
        # Test each strategy
        for strategy_name in strategy_names:
            try:
                result = self.run_strategy_test(strategy_name)
                self.results[strategy_name] = result
                
                # Wait between tests to avoid rate limits
                if strategy_name != strategy_names[-1]:
                    self.logger.info("\nWaiting 30 seconds before next test...")
                    time.sleep(30)
                    
            except Exception as e:
                self.logger.error(f"Error testing strategy {strategy_name}: {e}", exc_info=True)
                self.results[strategy_name] = {
                    'strategy': strategy_name,
                    'success': False,
                    'error': str(e)
                }
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Generate comparison report."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("GENERATING COMPARISON REPORT")
        self.logger.info("=" * 80)
        
        # Load ground truth
        gt_data = None
        if self.test_truth and self.test_truth.exists():
            with open(self.test_truth, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        
        # Create comparison data
        comparison = {
            'test_image': str(self.test_image),
            'test_truth': str(self.test_truth) if self.test_truth else None,
            'timestamp': datetime.now().isoformat(),
            'strategies': {}
        }
        
        for strategy_name, result in self.results.items():
            if result.get('success'):
                kpis = result.get('kpis', {})
                comparison['strategies'][strategy_name] = {
                    'duration_minutes': result.get('duration_minutes', 0.0),
                    'element_f1': kpis.get('element_f1', 0.0),
                    'element_precision': kpis.get('element_precision', 0.0),
                    'element_recall': kpis.get('element_recall', 0.0),
                    'connection_f1': kpis.get('connection_f1', 0.0),
                    'quality_score': kpis.get('quality_score', 0.0),
                    'elements_count': len(result.get('result', {}).get('elements', [])),
                    'connections_count': len(result.get('result', {}).get('connections', []))
                }
            else:
                comparison['strategies'][strategy_name] = {
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }
        
        # Save comparison report
        comparison_file = self.output_manager.get_data_path("comparison_report.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json_dump_safe(comparison, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPARISON SUMMARY")
        self.logger.info("=" * 80)
        
        for strategy_name, data in comparison['strategies'].items():
            if data.get('success') is False:
                self.logger.info(f"\n{strategy_name}: FAILED - {data.get('error', 'Unknown error')}")
            else:
                self.logger.info(f"\n{strategy_name}:")
                self.logger.info(f"  Duration: {data.get('duration_minutes', 0.0):.2f} minutes")
                self.logger.info(f"  Quality Score: {data.get('quality_score', 0.0):.2f}")
                self.logger.info(f"  Element F1: {data.get('element_f1', 0.0):.4f}")
                self.logger.info(f"  Connection F1: {data.get('connection_f1', 0.0):.4f}")
                self.logger.info(f"  Elements: {data.get('elements_count', 0)}")
                self.logger.info(f"  Connections: {data.get('connections_count', 0)}")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Comparison report saved: {comparison_file}")
        self.logger.info("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare fusion strategies")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["separation", "full_redundancy", "local_global", "id_override", "current"],
        default=["current"],  # Start with current implementation
        help="Strategies to test (default: current)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="complex",
        choices=["complex", "simple", "uni"],
        help="Test image to use (default: complex)"
    )
    
    args = parser.parse_args()
    
    # Select test image
    if args.image == "simple":
        test_image = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
        test_truth = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
    elif args.image == "complex":
        test_image = TEST_IMAGE_COMPLEX
        test_truth = TEST_GROUND_TRUTH_COMPLEX
    else:  # uni
        test_image = project_root / "training_data" / "complex_pids" / "Verfahrensfließbild_Uni.png"
        test_truth = project_root / "training_data" / "complex_pids" / "Verfahrensfließbild_Uni_truth.json"
    
    print("=" * 80)
    print("FUSION STRATEGY COMPARISON")
    print("=" * 80)
    print(f"Test Image: {test_image.name}")
    print(f"Strategies: {', '.join(args.strategies)}")
    print("=" * 80)
    print()
    
    comparison = StrategyComparison(test_image, test_truth)
    
    try:
        comparison.setup_services()
        comparison.compare_strategies(args.strategies)
    except KeyboardInterrupt:
        comparison.logger.warning("\nComparison interrupted by user")
    except Exception as e:
        comparison.logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

