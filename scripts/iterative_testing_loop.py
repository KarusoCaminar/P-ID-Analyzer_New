"""
Iterative Testing & Improvement Loop
Testen -> Ergebnisse lesen -> Auswertung -> Verbesserung -> Erneut testen
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IterativeTester:
    """Iterative Testing mit automatischer Verbesserung."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.training_data_dir = project_root / "training_data" / "organized_tests"
        self.output_dir = project_root / "outputs" / "iterative_tests"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_history: List[Dict[str, Any]] = []
        self.improvements: List[Dict[str, Any]] = []
        
    def find_test_images(self) -> List[Path]:
        """Finde Test-Bilder mit Ground Truth: Maximale Simple PIDs + 2 Uni Bilder (3-4)."""
        test_images = []
        
        # 1. Simple PIDs finden
        simple_pids_dir = self.training_data_dir / "simple_pids"
        simple_images = []
        if simple_pids_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                simple_images.extend(list(simple_pids_dir.glob(ext)))
        
        # 2. Uni Bilder finden (page_3_original, page_4_original)
        complex_pids_dir = self.training_data_dir / "complex_pids"
        uni_images = []
        if complex_pids_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                complex_images = list(complex_pids_dir.glob(ext))
                # Filter für Uni Bilder (page_3, page_4)
                for img in complex_images:
                    name_lower = img.name.lower()
                    if 'page_3' in name_lower or 'page_4' in name_lower:
                        if not any(exclude in name_lower for exclude in ['truth', 'output', 'result', 'cgm', 'temp']):
                            uni_images.append(img)
        
        # Filter: Nur Bilder mit Truth-Dateien
        images_with_truth = []
        
        # Simple PIDs: Maximal alle
        for img in simple_images:
            if any(exclude in img.name.lower() for exclude in ['truth', 'output', 'result', 'cgm', 'temp']):
                continue
            
            base_name = img.stem
            truth_patterns = [
                img.parent / f"{base_name}_truth_cgm.json",
                img.parent / f"{base_name}_truth.json",
            ]
            
            for pattern in truth_patterns:
                if pattern.exists():
                    images_with_truth.append(img)
                    break
            else:
                truth_files = list(self.training_data_dir.rglob(f"{base_name}*truth*.json"))
                if truth_files:
                    images_with_truth.append(img)
        
        # Uni Bilder: Maximal 2 (page_3, page_4)
        uni_count = 0
        for img in uni_images:
            if uni_count >= 2:
                break
            
            if any(exclude in img.name.lower() for exclude in ['truth', 'output', 'result', 'cgm', 'temp']):
                continue
            
            base_name = img.stem
            truth_patterns = [
                img.parent / f"{base_name}_truth_cgm.json",
                img.parent / f"{base_name}_truth.json",
            ]
            
            for pattern in truth_patterns:
                if pattern.exists():
                    images_with_truth.append(img)
                    uni_count += 1
                    break
            else:
                truth_files = list(self.training_data_dir.rglob(f"{base_name}*truth*.json"))
                if truth_files:
                    images_with_truth.append(img)
                    uni_count += 1
        
        logger.info(f"Found {len(images_with_truth)} test images: {len(images_with_truth) - uni_count} simple PIDs + {uni_count} Uni Bilder (3-4)")
        return images_with_truth
    
    def run_single_test(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Führe einen einzelnen Test durch."""
        try:
            import os
            from src.services.config_service import ConfigService
            from src.analyzer.ai.llm_client import LLMClient
            from src.analyzer.learning.knowledge_manager import KnowledgeManager
            from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
            
            gcp_project_id = os.getenv('GCP_PROJECT_ID')
            gcp_location = os.getenv('GCP_LOCATION', 'us-central1')
            
            if not gcp_project_id:
                logger.warning("GCP_PROJECT_ID not set - API calls will fail")
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
            
            coordinator = PipelineCoordinator(
                llm_client=llm_client,
                knowledge_manager=knowledge_manager,
                config_service=config_service
            )
            
            # Run analysis
            result = coordinator.process_image(str(image_path))
            
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
                            'image': image_path.name,
                            'quality_score': result.quality_score,
                            'elements': len(result.elements),
                            'connections': len(result.connections),
                            'kpis': kpis,
                            'timestamp': datetime.now().isoformat()
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in test for {image_path.name}: {e}", exc_info=True)
            return None
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analysiere Testergebnisse und identifiziere Verbesserungspotenziale."""
        
        if not results:
            return {'error': 'No results to analyze'}
        
        # Metriken sammeln
        quality_scores = [r.get('quality_score', 0) for r in results]
        precisions = [r.get('kpis', {}).get('element_precision', 0) for r in results if r.get('kpis')]
        recalls = [r.get('kpis', {}).get('element_recall', 0) for r in results if r.get('kpis')]
        f1_scores = [r.get('kpis', {}).get('element_f1', 0) for r in results if r.get('kpis')]
        
        hallucinated = [r.get('kpis', {}).get('hallucinated_elements', 0) for r in results if r.get('kpis')]
        missed = [r.get('kpis', {}).get('missed_elements', 0) for r in results if r.get('kpis')]
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'metrics': {
                'quality_score': {
                    'mean': statistics.mean(quality_scores) if quality_scores else 0,
                    'median': statistics.median(quality_scores) if quality_scores else 0,
                    'min': min(quality_scores) if quality_scores else 0,
                    'max': max(quality_scores) if quality_scores else 0,
                },
                'precision': {
                    'mean': statistics.mean(precisions) if precisions else 0,
                    'median': statistics.median(precisions) if precisions else 0,
                },
                'recall': {
                    'mean': statistics.mean(recalls) if recalls else 0,
                    'median': statistics.median(recalls) if recalls else 0,
                },
                'f1_score': {
                    'mean': statistics.mean(f1_scores) if f1_scores else 0,
                    'median': statistics.median(f1_scores) if f1_scores else 0,
                },
            },
            'problems': [],
            'recommendations': []
        }
        
        # Probleme identifizieren
        avg_precision = statistics.mean(precisions) if precisions else 0
        avg_recall = statistics.mean(recalls) if recalls else 0
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        if avg_precision < 0.3:
            analysis['problems'].append({
                'type': 'low_precision',
                'severity': 'high',
                'value': avg_precision,
                'description': f'Precision zu niedrig: {avg_precision*100:.1f}% (sollte >30%)'
            })
        
        if avg_recall < 0.3:
            analysis['problems'].append({
                'type': 'low_recall',
                'severity': 'high',
                'value': avg_recall,
                'description': f'Recall zu niedrig: {avg_recall*100:.1f}% (sollte >30%)'
            })
        
        if avg_quality < 30:
            analysis['problems'].append({
                'type': 'low_quality_score',
                'severity': 'high',
                'value': avg_quality,
                'description': f'Quality Score zu niedrig: {avg_quality:.1f} (sollte >30)'
            })
        
        avg_hallucinated = statistics.mean(hallucinated) if hallucinated else 0
        if avg_hallucinated > 10:
            analysis['problems'].append({
                'type': 'high_hallucinations',
                'severity': 'medium',
                'value': avg_hallucinated,
                'description': f'Zu viele Halluzinationen: {avg_hallucinated:.1f} pro Analyse'
            })
        
        # Empfehlungen generieren
        if avg_precision < 0.3:
            analysis['recommendations'].append({
                'priority': 'high',
                'action': 'Prompt Engineering - Anti-Halluzination Prompts',
                'reason': f'Precision nur {avg_precision*100:.1f}%'
            })
        
        if avg_hallucinated > 10:
            analysis['recommendations'].append({
                'priority': 'high',
                'action': 'Post-Processing Filter für Halluzinationen',
                'reason': f'{avg_hallucinated:.1f} Halluzinationen pro Analyse'
            })
        
        if avg_quality < 30:
            analysis['recommendations'].append({
                'priority': 'high',
                'action': 'Confidence Threshold erhöhen (>0.7)',
                'reason': f'Quality Score nur {avg_quality:.1f}'
            })
        
        return analysis
    
    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Führe eine Test-Iteration durch."""
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")
        
        # Finde Test-Bilder
        test_images = self.find_test_images()
        
        if not test_images:
            logger.error("No test images found!")
            return {'error': 'No test images'}
        
        # Führe Tests durch (max 3 für schnelle Iteration)
        results = []
        for img in test_images[:3]:
            logger.info(f"Testing: {img.name}")
            result = self.run_single_test(img)
            if result:
                results.append(result)
        
        # Analysiere Ergebnisse
        analysis = self.analyze_results(results)
        
        # Speichere Ergebnisse
        iteration_file = self.output_dir / f"iteration_{iteration:03d}.json"
        with open(iteration_file, 'w') as f:
            json.dump({
                'iteration': iteration,
                'results': results,
                'analysis': analysis
            }, f, indent=2)
        
        # Vergleich mit vorheriger Iteration
        if self.test_history:
            last_analysis = self.test_history[-1].get('analysis', {})
            last_quality = last_analysis.get('metrics', {}).get('quality_score', {}).get('mean', 0)
            current_quality = analysis.get('metrics', {}).get('quality_score', {}).get('mean', 0)
            
            improvement = current_quality - last_quality
            logger.info(f"Quality Score: {last_quality:.2f} -> {current_quality:.2f} ({improvement:+.2f})")
            
            if improvement > 0:
                logger.info(f"✅ Verbesserung um {improvement:.2f} Punkte!")
            elif improvement < -1:
                logger.warning(f"⚠️  Verschlechterung um {abs(improvement):.2f} Punkte")
        
        self.test_history.append({
            'iteration': iteration,
            'results': results,
            'analysis': analysis
        })
        
        return analysis
    
    def should_continue(self, analysis: Dict[str, Any]) -> bool:
        """Prüfe ob weitere Iterationen sinnvoll sind."""
        metrics = analysis.get('metrics', {})
        quality = metrics.get('quality_score', {}).get('mean', 0)
        
        # Stoppe wenn Quality Score gut genug ist
        if quality > 70:
            logger.info("✅ Ziel erreicht! Quality Score > 70")
            return False
        
        # Stoppe wenn zu viele Probleme
        problems = analysis.get('problems', [])
        critical_problems = [p for p in problems if p.get('severity') == 'high']
        
        if len(critical_problems) == 0:
            logger.info("✅ Keine kritischen Probleme mehr!")
            return False
        
        return True


def main():
    """Hauptfunktion für iterative Tests."""
    project_root = Path(__file__).parent
    
    tester = IterativeTester(project_root)
    
    iteration = 1
    max_iterations = 10
    
    while iteration <= max_iterations:
        analysis = tester.run_iteration(iteration)
        
        if 'error' in analysis:
            logger.error(f"Error in iteration {iteration}: {analysis['error']}")
            break
        
        # Zeige Empfehlungen
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            logger.info("\n📋 EMPFEHLUNGEN:")
            for rec in recommendations:
                logger.info(f"  [{rec['priority'].upper()}] {rec['action']}")
        
        # Prüfe ob weitermachen
        if not tester.should_continue(analysis):
            logger.info("\n✅ Ziel erreicht oder keine kritischen Probleme mehr. Stoppe.")
            break
        
        iteration += 1
    
    # Final Report
    logger.info(f"\n{'='*60}")
    logger.info("FINALE ZUSAMMENFASSUNG")
    logger.info(f"{'='*60}")
    logger.info(f"Gesamt Iterationen: {len(tester.test_history)}")
    
    if tester.test_history:
        first_quality = tester.test_history[0].get('analysis', {}).get('metrics', {}).get('quality_score', {}).get('mean', 0)
        last_quality = tester.test_history[-1].get('analysis', {}).get('metrics', {}).get('quality_score', {}).get('mean', 0)
        
        logger.info(f"Quality Score Entwicklung: {first_quality:.2f} -> {last_quality:.2f} ({last_quality-first_quality:+.2f})")


if __name__ == '__main__':
    main()

