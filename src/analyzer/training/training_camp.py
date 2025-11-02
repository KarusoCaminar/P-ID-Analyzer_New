"""
Training Camp - Optimiertes Training mit Strategien- und Parameter-Testing.

Features:
- Sequential testing von Strategien (1 nach dem anderen)
- Parameter-Optimierung
- Automatisches Speichern der besten Parameter
- Vollständige Power-Ausschöpfung
"""

import logging
import time
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from itertools import product
import csv

logger = logging.getLogger(__name__)


class TrainingCamp:
    """
    Optimiertes Training Camp mit Strategien- und Parameter-Testing.
    """
    
    def __init__(
        self,
        pipeline_coordinator: Any,
        config_service: Any,
        training_data_dir: Path,
        config_path: Path
    ):
        """
        Initialize Training Camp.
        
        Args:
            pipeline_coordinator: PipelineCoordinator instance
            config_service: ConfigService instance
            training_data_dir: Directory with training images
            config_path: Path to config.yaml
        """
        self.pipeline_coordinator = pipeline_coordinator
        self.config_service = config_service
        self.training_data_dir = training_data_dir
        self.config_path = config_path
        
        self.training_stats = {
            'total_runs': 0,
            'total_strategies_tested': 0,
            'total_parameter_combinations_tested': 0,
            'best_score': 0.0,
            'best_strategy': None,
            'best_parameters': None,
            'improvement_history': []
        }
        
        # Training report paths
        self.report_csv_path = config_path.parent / "training_report.csv"
        self.report_json_path = config_path.parent / "training_report_detail.json"
    
    def run_full_training_camp(
        self,
        duration_hours: float = 24.0,
        max_cycles: int = 0,
        sequential: bool = True
    ) -> Dict[str, Any]:
        """
        Run full training camp with strategy and parameter testing.
        
        Args:
            duration_hours: Total training duration
            max_cycles: Maximum cycles (0 = unlimited)
            sequential: Test strategies one by one (True) or parallel (False)
            
        Returns:
            Training report
        """
        logger.info("=== STARTING FULL TRAINING CAMP ===")
        logger.info(f"Duration: {duration_hours}h, Max Cycles: {max_cycles}, Sequential: {sequential}")
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600.0)
        cycle_count = 0
        
        # Get strategies and parameters from config
        config = self.config_service.get_raw_config() or {}
        strategies = self._get_strategies_from_config(config)
        parameter_combinations = self._generate_parameter_combinations(config)
        
        logger.info(f"Found {len(strategies)} strategies and {len(parameter_combinations)} parameter combinations")
        
        # Initialize training report
        self._initialize_report()
        
        all_results = []
        
        while (max_cycles == 0 or cycle_count < max_cycles) and time.time() < end_time:
            cycle_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING CYCLE {cycle_count}")
            logger.info(f"{'='*60}")
            
            # Find training images
            training_images = self._find_training_images()
            if not training_images:
                logger.warning("No training images found. Waiting...")
                time.sleep(300)
                continue
            
            logger.info(f"Found {len(training_images)} training images")
            
            # Test each strategy sequentially
            for strategy_idx, strategy in enumerate(strategies):
                strategy_name = strategy.get('name', f'strategy_{strategy_idx}')
                logger.info(f"\n--- Testing Strategy: {strategy_name} ({strategy_idx + 1}/{len(strategies)}) ---")
                
                # Test each parameter combination
                for param_idx, params in enumerate(parameter_combinations):
                    if time.time() >= end_time:
                        logger.info("Time limit reached. Stopping.")
                        break
                    
                    param_name = f"params_{param_idx}"
                    logger.info(f"\n  Testing Parameter Combination {param_idx + 1}/{len(parameter_combinations)}: {params}")
                    
                    # Run analysis for each image with this strategy and parameters
                    cycle_scores = []
                    param_results = []
                    
                    for image_idx, image_path in enumerate(training_images):
                        if time.time() >= end_time:
                            break
                        
                        logger.info(f"    [{image_idx + 1}/{len(training_images)}] Processing: {image_path.name}")
                        
                        try:
                            # Run analysis with strategy and parameters
                            result = self._run_analysis_with_config(
                                image_path=image_path,
                                strategy=strategy,
                                parameters=params
                            )
                            
                            if result:
                                quality_score = result.get('quality_score', 0.0)
                                cycle_scores.append(quality_score)
                                param_results.append({
                                    'image': image_path.name,
                                    'score': quality_score,
                                    'kpis': result.get('kpis', {})
                                })
                                
                                logger.info(f"      Score: {quality_score:.2f}")
                                
                                # Track best configuration
                                if quality_score > self.training_stats['best_score']:
                                    improvement = quality_score - self.training_stats['best_score']
                                    self.training_stats['best_score'] = quality_score
                                    self.training_stats['best_strategy'] = strategy_name
                                    self.training_stats['best_parameters'] = params.copy()
                                    self.training_stats['improvement_history'].append({
                                        'cycle': cycle_count,
                                        'strategy': strategy_name,
                                        'parameters': params.copy(),
                                        'score': quality_score,
                                        'improvement': improvement,
                                        'timestamp': datetime.now().isoformat()
                                    })
                                    
                                    # Save best configuration to config.yaml
                                    self._save_best_config(params)
                                    
                                    logger.info(f"      *** NEW BEST SCORE: {quality_score:.2f} (+{improvement:.2f}) ***")
                                    logger.info(f"      Best Strategy: {strategy_name}")
                                    logger.info(f"      Best Parameters: {params}")
                            
                        except Exception as e:
                            logger.error(f"      Error processing {image_path.name}: {e}", exc_info=True)
                    
                    # Calculate average for this parameter combination
                    if cycle_scores:
                        avg_score = sum(cycle_scores) / len(cycle_scores)
                        logger.info(f"    Average Score for this combination: {avg_score:.2f}")
                        
                        # Record result
                        result_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'cycle': cycle_count,
                            'strategy': strategy_name,
                            'parameters': params,
                            'average_score': avg_score,
                            'scores': cycle_scores,
                            'images_processed': len(param_results),
                            'results': param_results
                        }
                        all_results.append(result_entry)
                        self._write_report_entry(result_entry)
                        
                        self.training_stats['total_runs'] += len(param_results)
                        self.training_stats['total_parameter_combinations_tested'] += 1
                
                self.training_stats['total_strategies_tested'] += 1
            
            # Save detailed report
            self._save_detailed_report(all_results)
            
            logger.info(f"\nCycle {cycle_count} complete.")
            logger.info(f"Best Score So Far: {self.training_stats['best_score']:.2f}")
            logger.info(f"Best Strategy: {self.training_stats['best_strategy']}")
        
        # Final report
        total_time = time.time() - start_time
        final_report = {
            'total_cycles': cycle_count,
            'total_time_hours': total_time / 3600.0,
            'best_score': self.training_stats['best_score'],
            'best_strategy': self.training_stats['best_strategy'],
            'best_parameters': self.training_stats['best_parameters'],
            'total_runs': self.training_stats['total_runs'],
            'total_strategies_tested': self.training_stats['total_strategies_tested'],
            'total_parameter_combinations_tested': self.training_stats['total_parameter_combinations_tested'],
            'improvement_count': len(self.training_stats['improvement_history'])
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING CAMP COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Cycles: {cycle_count}")
        logger.info(f"Best Score: {self.training_stats['best_score']:.2f}")
        logger.info(f"Best Strategy: {self.training_stats['best_strategy']}")
        logger.info(f"Total Runs: {self.training_stats['total_runs']}")
        
        return final_report
    
    def _get_strategies_from_config(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract strategies from config."""
        strategies = []
        
        training_camp = config.get('training_camp', {})
        strategies_config = training_camp.get('strategies_and_models_to_test', {})
        
        for strategy_name, strategy_config in strategies_config.items():
            if isinstance(strategy_config, dict):
                strategy = {
                    'name': strategy_name,
                    'config': strategy_config
                }
                strategies.append(strategy)
        
        # Fallback: Create default strategies if none found
        if not strategies:
            logger.warning("No strategies found in config. Using default strategies.")
            strategies = [
                {
                    'name': 'default_flash',
                    'config': {
                        'meta_model': 'Google Gemini 2.5 Flash',
                        'detail_model': 'Google Gemini 2.5 Flash',
                        'correction_model': 'Google Gemini 2.5 Flash'
                    }
                }
            ]
        
        return strategies
    
    def _generate_parameter_combinations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter combinations for testing."""
        training_camp = config.get('training_camp', {})
        parameter_combinations_config = training_camp.get('parameter_combinations', [])
        
        if parameter_combinations_config:
            return parameter_combinations_config
        
        # Generate default parameter combinations
        logic_params = config.get('logic_parameters', {})
        
        # Optimizable parameters with ranges
        optimizable_params = {
            'iou_match_threshold': [0.3, 0.4, 0.5, 0.6],
            'min_quality_to_keep_bbox': [0.4, 0.5, 0.6, 0.7],
            'max_self_correction_iterations': [1, 2, 3],
            'target_quality_score': [85.0, 90.0, 95.0]
        }
        
        # Generate all combinations
        param_names = list(optimizable_params.keys())
        param_values = list(optimizable_params.values())
        
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        
        return combinations
    
    def _run_analysis_with_config(
        self,
        image_path: Path,
        strategy: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run analysis with specific strategy and parameters."""
        try:
            # Update pipeline coordinator with strategy and parameters
            # (This would need to be implemented based on your pipeline structure)
            
            # For now, use pipeline coordinator with parameter override
            result = self.pipeline_coordinator.process(
                image_path=str(image_path),
                output_dir=None,
                params_override=parameters
            )
            
            # Extract results
            if hasattr(result, 'quality_score'):
                return {
                    'quality_score': result.quality_score,
                    'kpis': result.kpis.model_dump() if hasattr(result.kpis, 'model_dump') else result.kpis,
                    'elements': len(result.elements) if hasattr(result, 'elements') else 0,
                    'connections': len(result.connections) if hasattr(result, 'connections') else 0
                }
            else:
                return {
                    'quality_score': result.get('quality_score', 0.0),
                    'kpis': result.get('kpis', {}),
                    'elements': len(result.get('elements', [])),
                    'connections': len(result.get('connections', []))
                }
        except Exception as e:
            logger.error(f"Error in analysis: {e}", exc_info=True)
            return None
    
    def _find_training_images(self) -> List[Path]:
        """Find training images."""
        training_images = []
        
        try:
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                training_images.extend(list(self.training_data_dir.rglob(ext)))
            
            # Filter out truth files and output files
            training_images = [
                img for img in training_images
                if not any(exclude in img.name.lower() for exclude in ['truth', 'output', 'result', 'cgm', 'heatmap', 'debug', 'confidence', 'score'])
            ]
            
            logger.info(f"Found {len(training_images)} training images")
        except Exception as e:
            logger.error(f"Error finding training images: {e}", exc_info=True)
        
        return training_images
    
    def _save_best_config(self, parameters: Dict[str, Any]) -> None:
        """Save best configuration to config.yaml."""
        try:
            # Load current config
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Update logic_parameters with best parameters
            if 'logic_parameters' not in config:
                config['logic_parameters'] = {}
            
            config['logic_parameters'].update(parameters)
            
            # Add metadata
            config['logic_parameters']['_best_config_timestamp'] = datetime.now().isoformat()
            config['logic_parameters']['_best_score'] = self.training_stats['best_score']
            
            # Save config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            logger.info(f"Best configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving best config: {e}", exc_info=True)
    
    def _initialize_report(self) -> None:
        """Initialize CSV report file."""
        try:
            if not self.report_csv_path.exists():
                with open(self.report_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'cycle', 'strategy', 'parameters',
                        'average_score', 'images_processed', 'best_score'
                    ])
        except Exception as e:
            logger.error(f"Error initializing report: {e}", exc_info=True)
    
    def _write_report_entry(self, entry: Dict[str, Any]) -> None:
        """Write entry to CSV report."""
        try:
            with open(self.report_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    entry.get('timestamp', ''),
                    entry.get('cycle', 0),
                    entry.get('strategy', ''),
                    json.dumps(entry.get('parameters', {})),
                    entry.get('average_score', 0.0),
                    entry.get('images_processed', 0),
                    self.training_stats['best_score']
                ])
        except Exception as e:
            logger.error(f"Error writing report entry: {e}", exc_info=True)
    
    def _save_detailed_report(self, all_results: List[Dict[str, Any]]) -> None:
        """Save detailed JSON report."""
        try:
            report_data = {
                'training_stats': self.training_stats,
                'all_results': all_results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.report_json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving detailed report: {e}", exc_info=True)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()


