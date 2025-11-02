"""
Model Comparison Test Script
Tests different Gemini models (2.5 Flash, Preview, Pro) and compares results.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import statistics
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelComparisonTester:
    """Test different Gemini models and compare results."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.training_data_dir = project_root / "training_data" / "organized_tests"
        self.output_dir = project_root / "outputs" / "model_comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = [
            {
                'name': 'Gemini 2.5 Flash',
                'meta_model': 'Google Gemini 2.5 Flash',
                'hotspot_model': 'Google Gemini 2.5 Flash',
                'detail_model': 'Google Gemini 2.5 Flash',
                'coarse_model': 'Google Gemini 2.5 Flash',
                'correction_model': 'Google Gemini 2.5 Flash',
                'code_gen_model': 'Google Gemini 2.5 Flash',
                'critic_model_name': 'Google Gemini 2.5 Flash'
            },
            {
                'name': 'Gemini 2.5 Preview',
                'meta_model': 'Google Gemini 2.5 Preview',
                'hotspot_model': 'Google Gemini 2.5 Preview',
                'detail_model': 'Google Gemini 2.5 Preview',
                'coarse_model': 'Google Gemini 2.5 Preview',
                'correction_model': 'Google Gemini 2.5 Preview',
                'code_gen_model': 'Google Gemini 2.5 Preview',
                'critic_model_name': 'Google Gemini 2.5 Preview'
            },
            {
                'name': 'Gemini 2.5 Pro',
                'meta_model': 'Google Gemini 2.5 Pro',
                'hotspot_model': 'Google Gemini 2.5 Pro',
                'detail_model': 'Google Gemini 2.5 Pro',
                'coarse_model': 'Google Gemini 2.5 Pro',
                'correction_model': 'Google Gemini 2.5 Pro',
                'code_gen_model': 'Google Gemini 2.5 Pro',
                'critic_model_name': 'Google Gemini 2.5 Pro'
            }
        ]
        
        self.test_results: Dict[str, List[Dict[str, Any]]] = {}
    
    def find_test_images(self) -> List[Path]:
        """Find test images with ground truth."""
        test_images = []
        
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            test_images.extend(list(self.training_data_dir.rglob(ext)))
        
        # Filter: Only images with truth files
        images_with_truth = []
        for img in test_images:
            if any(exclude in img.name.lower() for exclude in ['truth', 'output', 'result', 'cgm', 'temp']):
                continue
            
            base_name = img.stem
            truth_patterns = [
                img.parent / f"{base_name}_truth_cgm.json",
                img.parent / f"{base_name}_truth.json",
            ]
            
            # Also search recursively
            truth_files = list(self.training_data_dir.rglob(f"{base_name}*truth*.json"))
            if truth_files or any(p.exists() for p in truth_patterns):
                images_with_truth.append(img)
        
        logger.info(f"Found {len(images_with_truth)} test images with ground truth")
        return images_with_truth[:3]  # Limit to 3 for comparison
    
    def run_test_with_model(self, model_config: Dict[str, str], image_path: Path) -> Optional[Dict[str, Any]]:
        """Run test with specific model configuration."""
        try:
            from src.services.config_service import ConfigService
            from src.analyzer.ai.llm_client import LLMClient
            from src.analyzer.learning.knowledge_manager import KnowledgeManager
            from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
            
            gcp_project_id = os.getenv('GCP_PROJECT_ID')
            gcp_location = os.getenv('GCP_LOCATION', 'us-central1')
            
            if not gcp_project_id:
                logger.warning("GCP_PROJECT_ID not set - skipping test")
                return None
            
            # Initialize components
            config_path = self.project_root / "config.yaml"
            config_service = ConfigService(config_path=config_path if config_path.exists() else None)
            config = config_service.get_config()
            
            llm_client = LLMClient(
                project_id=gcp_project_id,
                default_location=gcp_location,
                config=config.model_dump()
            )
            
            # Reset circuit breaker
            if hasattr(llm_client, 'retry_handler') and hasattr(llm_client.retry_handler, 'circuit_breaker'):
                llm_client.retry_handler.circuit_breaker.reset()
            
            element_type_list_path = config_service.get_path("element_type_list") or self.project_root / "element_type_list.json"
            learning_db_path = config_service.get_path("learning_db") or self.project_root / "learning_db.json"
            
            knowledge_manager = KnowledgeManager(
                element_type_list_path=str(element_type_list_path),
                learning_db_path=str(learning_db_path),
                llm_handler=llm_client,
                config=config.model_dump()
            )
            
            # Use model-specific strategy
            model_strategy = model_config
            
            coordinator = PipelineCoordinator(
                llm_client=llm_client,
                knowledge_manager=knowledge_manager,
                config_service=config_service,
                model_strategy=model_strategy,
                progress_callback=None
            )
            
            # Set active_logic_parameters after initialization
            coordinator.active_logic_parameters = {
                'confidence_threshold': 0.7,
                'iou_match_threshold': 0.1
            }
            
            # Run analysis
            logger.info(f"Testing {model_config['name']} on {image_path.name}...")
            result = coordinator.process(
                image_path=str(image_path),
                output_dir=None,
                params_override=None
            )
            
            if result and hasattr(result, 'quality_score'):
                # Load KPIs from output
                output_base = self.project_root / "outputs"
                output_dirs = sorted(output_base.glob(f"{image_path.stem}_output_*"), reverse=True)
                
                if output_dirs:
                    kpi_file = output_dirs[0] / f"{image_path.stem}_kpis.json"
                    if kpi_file.exists():
                        with open(kpi_file, 'r') as f:
                            kpis = json.load(f)
                        
                        return {
                            'model': model_config['name'],
                            'image': image_path.name,
                            'quality_score': result.quality_score,
                            'elements': len(result.elements),
                            'connections': len(result.connections),
                            'kpis': kpis,
                            'timestamp': datetime.now().isoformat()
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error testing {model_config['name']} on {image_path.name}: {e}", exc_info=True)
            return None
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run comparison tests with all models."""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON TEST")
        logger.info("="*60)
        
        test_images = self.find_test_images()
        if not test_images:
            logger.error("No test images found!")
            return {'error': 'No test images'}
        
        logger.info(f"Testing {len(test_images)} images with {len(self.models)} models")
        
        # Run tests for each model
        for model_config in self.models:
            model_name = model_config['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing Model: {model_name}")
            logger.info(f"{'='*60}")
            
            self.test_results[model_name] = []
            
            for img in test_images:
                result = self.run_test_with_model(model_config, img)
                if result:
                    self.test_results[model_name].append(result)
                    logger.info(f"  {img.name}: Quality={result['quality_score']:.2f}, Elements={result['elements']}, Connections={result['connections']}")
        
        # Compare results
        comparison = self.analyze_comparison()
        
        # Save results
        results_file = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'test_results': self.test_results,
                'comparison': comparison,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        return comparison
    
    def analyze_comparison(self) -> Dict[str, Any]:
        """Analyze and compare model results."""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'model_scores': {},
            'model_metrics': {},
            'best_model': {},
            'recommendations': []
        }
        
        for model_name, results in self.test_results.items():
            if not results:
                continue
            
            quality_scores = [r.get('quality_score', 0) for r in results]
            precisions = [r.get('kpis', {}).get('element_precision', 0) for r in results if r.get('kpis')]
            recalls = [r.get('kpis', {}).get('element_recall', 0) for r in results if r.get('kpis')]
            f1_scores = [r.get('kpis', {}).get('element_f1', 0) for r in results if r.get('kpis')]
            element_counts = [r.get('elements', 0) for r in results]
            connection_counts = [r.get('connections', 0) for r in results]
            
            comparison['model_scores'][model_name] = {
                'quality_score_mean': statistics.mean(quality_scores) if quality_scores else 0,
                'quality_score_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                'quality_score_min': min(quality_scores) if quality_scores else 0,
                'quality_score_max': max(quality_scores) if quality_scores else 0,
            }
            
            comparison['model_metrics'][model_name] = {
                'precision_mean': statistics.mean(precisions) if precisions else 0,
                'recall_mean': statistics.mean(recalls) if recalls else 0,
                'f1_mean': statistics.mean(f1_scores) if f1_scores else 0,
                'elements_mean': statistics.mean(element_counts) if element_counts else 0,
                'connections_mean': statistics.mean(connection_counts) if connection_counts else 0,
            }
        
        # Find best model for each metric
        if comparison['model_scores']:
            best_quality = max(
                comparison['model_scores'].items(),
                key=lambda x: x[1]['quality_score_mean']
            )
            comparison['best_model']['quality'] = best_quality[0]
            
        if comparison['model_metrics']:
            best_precision = max(
                comparison['model_metrics'].items(),
                key=lambda x: x[1]['precision_mean']
            )
            comparison['best_model']['precision'] = best_precision[0]
            
            best_recall = max(
                comparison['model_metrics'].items(),
                key=lambda x: x[1]['recall_mean']
            )
            comparison['best_model']['recall'] = best_recall[0]
            
            best_f1 = max(
                comparison['model_metrics'].items(),
                key=lambda x: x[1]['f1_mean']
            )
            comparison['best_model']['f1'] = best_f1[0]
        
        # Generate recommendations
        if comparison['model_scores']:
            logger.info("\n" + "="*60)
            logger.info("MODEL COMPARISON RESULTS")
            logger.info("="*60)
            
            for model_name, scores in comparison['model_scores'].items():
                logger.info(f"\n{model_name}:")
                logger.info(f"  Quality Score: {scores['quality_score_mean']:.2f} ± {scores['quality_score_std']:.2f}")
                
                if model_name in comparison['model_metrics']:
                    metrics = comparison['model_metrics'][model_name]
                    logger.info(f"  Precision: {metrics['precision_mean']:.3f}")
                    logger.info(f"  Recall: {metrics['recall_mean']:.3f}")
                    logger.info(f"  F1-Score: {metrics['f1_mean']:.3f}")
                    logger.info(f"  Elements: {metrics['elements_mean']:.1f}")
                    logger.info(f"  Connections: {metrics['connections_mean']:.1f}")
            
            if comparison['best_model']:
                logger.info("\nBest Models:")
                for metric, model in comparison['best_model'].items():
                    logger.info(f"  {metric.capitalize()}: {model}")
                
                # Generate recommendations
                if 'quality' in comparison['best_model']:
                    best = comparison['best_model']['quality']
                    comparison['recommendations'].append({
                        'metric': 'overall_quality',
                        'recommendation': f"Use {best} for best overall quality",
                        'model': best
                    })
                
                if 'precision' in comparison['best_model']:
                    best = comparison['best_model']['precision']
                    comparison['recommendations'].append({
                        'metric': 'precision',
                        'recommendation': f"Use {best} for highest precision",
                        'model': best
                    })
                
                if 'recall' in comparison['best_model']:
                    best = comparison['best_model']['recall']
                    comparison['recommendations'].append({
                        'metric': 'recall',
                        'recommendation': f"Use {best} for highest recall",
                        'model': best
                    })
        
        return comparison


def main():
    """Main entry point for model comparison tests."""
    project_root = Path(__file__).parent
    
    tester = ModelComparisonTester(project_root)
    comparison = tester.run_comparison()
    
    if 'error' not in comparison:
        logger.info("\n" + "="*60)
        logger.info("COMPARISON COMPLETE")
        logger.info("="*60)
        
        # Save markdown report
        report_file = tester.output_dir / f"model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Model Scores\n\n")
            for model_name, scores in comparison.get('model_scores', {}).items():
                f.write(f"### {model_name}\n\n")
                f.write(f"- Quality Score: {scores['quality_score_mean']:.2f} ± {scores['quality_score_std']:.2f}\n")
                f.write(f"- Range: {scores['quality_score_min']:.2f} - {scores['quality_score_max']:.2f}\n\n")
            
            f.write("## Best Models by Metric\n\n")
            for metric, model in comparison.get('best_model', {}).items():
                f.write(f"- **{metric.capitalize()}**: {model}\n")
            
            f.write("\n## Recommendations\n\n")
            for rec in comparison.get('recommendations', []):
                f.write(f"- **{rec['metric']}**: {rec['recommendation']}\n")
        
        logger.info(f"Report saved to: {report_file}")


if __name__ == '__main__':
    main()

