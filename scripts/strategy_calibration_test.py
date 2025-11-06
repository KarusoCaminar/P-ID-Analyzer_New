#!/usr/bin/env python3
"""
Strategy Calibration Test Script - Phase B & C: Strategy Testing & Calibration

Tests 3 strategies to identify error amplification cascades:
1. Kritiker-Shadow-Mode (Phase 3): Critic identifies but no re-analysis
2. Fusion-Tuning (Phase 2c): IoU 0.1, 0.5, Confidence prioritization
3. Post-Processing-Ablation (Phase 4): Without CV BBox Refinement, without CoT Reasoning

Compares KPIs (F1-Score, Precision, Recall) to identify optimal configuration.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
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


class StrategyTestRunner:
    """Runs strategy tests and collects KPIs."""
    
    def __init__(self, config_service: ConfigService, llm_client: LLMClient, knowledge_manager: KnowledgeManager):
        self.config_service = config_service
        self.llm_client = llm_client
        self.knowledge_manager = knowledge_manager
        self.results = []
    
    def run_baseline(self, image_path: Path, truth_path: Optional[Path], output_dir: Path) -> Dict[str, Any]:
        """Run baseline (full pipeline)."""
        logger.info("\n" + "="*80)
        logger.info("BASELINE: Full Pipeline")
        logger.info("="*80)
        
        pipeline = PipelineCoordinator(
            llm_client=self.llm_client,
            knowledge_manager=self.knowledge_manager,
            config_service=self.config_service,
            progress_callback=DummyProgressCallback()
        )
        
        try:
            result = pipeline.process(
                image_path=str(image_path),
                output_dir=str(output_dir / "baseline"),
                params_override={'truth_data_path': str(truth_path) if truth_path else None}
            )
            
            kpis = self._extract_kpis(result, output_dir / "baseline")
            return {
                'strategy': 'baseline',
                'result': result.model_dump() if hasattr(result, 'model_dump') else result,
                'kpis': kpis
            }
        except Exception as e:
            logger.error(f"Baseline failed: {e}", exc_info=True)
            return {'strategy': 'baseline', 'error': str(e)}
    
    def run_kritiker_shadow_mode(self, image_path: Path, truth_path: Optional[Path], output_dir: Path) -> Dict[str, Any]:
        """Strategy 1: Kritiker-Shadow-Mode (Phase 3: Critic identifies but no re-analysis)."""
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 1: Kritiker-Shadow-Mode")
        logger.info("(Critic identifies uncertain zones, but NO re-analysis)")
        logger.info("="*80)
        
        # Override logic parameters to disable self-correction loop
        config = self.config_service.get_config()
        config_dict = config.model_dump()
        
        # Temporarily disable self-correction loop
        original_use_self_correction = config_dict.get('logic_parameters', {}).get('use_self_correction_loop', True)
        if 'logic_parameters' not in config_dict:
            config_dict['logic_parameters'] = {}
        config_dict['logic_parameters']['use_self_correction_loop'] = False
        
        # Update config service
        self.config_service.update_config(config_dict)
        
        try:
            pipeline = PipelineCoordinator(
                llm_client=self.llm_client,
                knowledge_manager=self.knowledge_manager,
                config_service=self.config_service,
                progress_callback=DummyProgressCallback()
            )
            
            result = pipeline.process(
                image_path=str(image_path),
                output_dir=str(output_dir / "kritiker_shadow"),
                params_override={'truth_data_path': str(truth_path) if truth_path else None}
            )
            
            kpis = self._extract_kpis(result, output_dir / "kritiker_shadow")
            return {
                'strategy': 'kritiker_shadow',
                'result': result.model_dump() if hasattr(result, 'model_dump') else result,
                'kpis': kpis
            }
        except Exception as e:
            logger.error(f"Kritiker-Shadow-Mode failed: {e}", exc_info=True)
            return {'strategy': 'kritiker_shadow', 'error': str(e)}
        finally:
            # Restore original config
            config_dict['logic_parameters']['use_self_correction_loop'] = original_use_self_correction
            self.config_service.update_config(config_dict)
    
    def run_fusion_tuning(self, image_path: Path, truth_path: Optional[Path], output_dir: Path, iou_threshold: float) -> Dict[str, Any]:
        """Strategy 2: Fusion-Tuning (IoU threshold parameter sweep)."""
        logger.info("\n" + "="*80)
        logger.info(f"STRATEGY 2: Fusion-Tuning (IoU={iou_threshold})")
        logger.info("="*80)
        
        # Override IoU threshold
        config = self.config_service.get_config()
        config_dict = config.model_dump()
        
        if 'logic_parameters' not in config_dict:
            config_dict['logic_parameters'] = {}
        original_iou = config_dict['logic_parameters'].get('iou_match_threshold', 0.3)
        config_dict['logic_parameters']['iou_match_threshold'] = iou_threshold
        
        # Update config service
        self.config_service.update_config(config_dict)
        
        try:
            pipeline = PipelineCoordinator(
                llm_client=self.llm_client,
                knowledge_manager=self.knowledge_manager,
                config_service=self.config_service,
                progress_callback=DummyProgressCallback()
            )
            
            result = pipeline.process(
                image_path=str(image_path),
                output_dir=str(output_dir / f"fusion_iou_{iou_threshold}"),
                params_override={'truth_data_path': str(truth_path) if truth_path else None}
            )
            
            kpis = self._extract_kpis(result, output_dir / f"fusion_iou_{iou_threshold}")
            return {
                'strategy': f'fusion_iou_{iou_threshold}',
                'result': result.model_dump() if hasattr(result, 'model_dump') else result,
                'kpis': kpis
            }
        except Exception as e:
            logger.error(f"Fusion-Tuning (IoU={iou_threshold}) failed: {e}", exc_info=True)
            return {'strategy': f'fusion_iou_{iou_threshold}', 'error': str(e)}
        finally:
            # Restore original config
            config_dict['logic_parameters']['iou_match_threshold'] = original_iou
            self.config_service.update_config(config_dict)
    
    def run_post_processing_ablation(
        self,
        image_path: Path,
        truth_path: Optional[Path],
        output_dir: Path,
        without_cv_bbox: bool = False,
        without_cot: bool = False
    ) -> Dict[str, Any]:
        """Strategy 3: Post-Processing-Ablation (remove CV BBox Refinement and/or CoT Reasoning)."""
        strategy_name = f"ablation_{'no_cv_bbox' if without_cv_bbox else ''}_{'no_cot' if without_cot else ''}".strip('_')
        logger.info("\n" + "="*80)
        logger.info(f"STRATEGY 3: Post-Processing-Ablation ({strategy_name})")
        logger.info("="*80)
        
        # Override logic parameters
        config = self.config_service.get_config()
        config_dict = config.model_dump()
        
        if 'logic_parameters' not in config_dict:
            config_dict['logic_parameters'] = {}
        
        original_cv_bbox = config_dict['logic_parameters'].get('use_cv_bbox_refinement', True)
        original_cot = config_dict['logic_parameters'].get('use_cot_reasoning', True)
        
        if without_cv_bbox:
            config_dict['logic_parameters']['use_cv_bbox_refinement'] = False
        if without_cot:
            config_dict['logic_parameters']['use_cot_reasoning'] = False
        
        # Update config service
        self.config_service.update_config(config_dict)
        
        try:
            pipeline = PipelineCoordinator(
                llm_client=self.llm_client,
                knowledge_manager=self.knowledge_manager,
                config_service=self.config_service,
                progress_callback=DummyProgressCallback()
            )
            
            result = pipeline.process(
                image_path=str(image_path),
                output_dir=str(output_dir / strategy_name),
                params_override={'truth_data_path': str(truth_path) if truth_path else None}
            )
            
            kpis = self._extract_kpis(result, output_dir / strategy_name)
            return {
                'strategy': strategy_name,
                'result': result.model_dump() if hasattr(result, 'model_dump') else result,
                'kpis': kpis
            }
        except Exception as e:
            logger.error(f"Post-Processing-Ablation ({strategy_name}) failed: {e}", exc_info=True)
            return {'strategy': strategy_name, 'error': str(e)}
        finally:
            # Restore original config
            config_dict['logic_parameters']['use_cv_bbox_refinement'] = original_cv_bbox
            config_dict['logic_parameters']['use_cot_reasoning'] = original_cot
            self.config_service.update_config(config_dict)
    
    def _extract_kpis(self, result: Any, output_dir: Path) -> Dict[str, Any]:
        """Extract KPIs from result."""
        try:
            if hasattr(result, 'quality_score'):
                quality_score = result.quality_score
            elif isinstance(result, dict):
                quality_score = result.get('quality_score', 0.0)
            else:
                quality_score = 0.0
            
            # Try to load KPIs from output directory
            kpi_file = output_dir / "kpis.json"
            if kpi_file.exists():
                with open(kpi_file, 'r', encoding='utf-8') as f:
                    kpis = json.load(f)
                    kpis['quality_score'] = quality_score
                    return kpis
            
            # Fallback: extract from result
            if hasattr(result, 'elements'):
                elements = result.elements
            elif isinstance(result, dict):
                elements = result.get('elements', [])
            else:
                elements = []
            
            if hasattr(result, 'connections'):
                connections = result.connections
            elif isinstance(result, dict):
                connections = result.get('connections', [])
            else:
                connections = []
            
            return {
                'quality_score': quality_score,
                'element_count': len(elements),
                'connection_count': len(connections),
                'avg_element_confidence': sum(el.get('confidence', 0.5) for el in elements) / len(elements) if elements else 0.0,
                'avg_connection_confidence': sum(conn.get('confidence', 0.5) for conn in connections) / len(connections) if connections else 0.0
            }
        except Exception as e:
            logger.warning(f"Error extracting KPIs: {e}")
            return {'quality_score': 0.0}


def main():
    """Main function."""
    output_base_dir = project_root / "outputs" / "strategy_calibration"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Use simple P&ID for testing
    simple_pid_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
    simple_pid_truth_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
    
    if not simple_pid_path.exists():
        logger.error(f"Image not found: {simple_pid_path}")
        return
    
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
    
    runner = StrategyTestRunner(config_service, llm_client, knowledge_manager)
    
    all_results = []
    
    # Test 1: Baseline
    baseline_result = runner.run_baseline(simple_pid_path, simple_pid_truth_path, output_base_dir)
    all_results.append(baseline_result)
    
    # Test 2: Kritiker-Shadow-Mode
    kritiker_result = runner.run_kritiker_shadow_mode(simple_pid_path, simple_pid_truth_path, output_base_dir)
    all_results.append(kritiker_result)
    
    # Test 3: Fusion-Tuning (IoU 0.1, 0.3, 0.5)
    for iou in [0.1, 0.3, 0.5]:
        fusion_result = runner.run_fusion_tuning(simple_pid_path, simple_pid_truth_path, output_base_dir, iou)
        all_results.append(fusion_result)
    
    # Test 4: Post-Processing-Ablation
    ablation_1 = runner.run_post_processing_ablation(simple_pid_path, simple_pid_truth_path, output_base_dir, without_cv_bbox=True)
    all_results.append(ablation_1)
    
    ablation_2 = runner.run_post_processing_ablation(simple_pid_path, simple_pid_truth_path, output_base_dir, without_cot=True)
    all_results.append(ablation_2)
    
    ablation_3 = runner.run_post_processing_ablation(simple_pid_path, simple_pid_truth_path, output_base_dir, without_cv_bbox=True, without_cot=True)
    all_results.append(ablation_3)
    
    # Save results
    results_file = output_base_dir / "strategy_calibration_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("STRATEGY CALIBRATION SUMMARY")
    logger.info("="*80)
    
    for result in all_results:
        if 'error' in result:
            logger.info(f"{result['strategy']}: FAILED - {result['error']}")
        else:
            kpis = result.get('kpis', {})
            logger.info(f"{result['strategy']}: Quality={kpis.get('quality_score', 0.0):.2f}%, "
                       f"Elements={kpis.get('element_count', 0)}, "
                       f"Connections={kpis.get('connection_count', 0)}")
    
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

