"""
Auto Trainer - Automatische Trainingsläufe für kontinuierliche Verbesserung.

Führt automatische Trainingszyklen durch und verbessert das System kontinuierlich.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AutoTrainer:
    """
    Automatischer Trainer für kontinuierliche Verbesserung.
    
    Features:
    - Automatische Trainingszyklen
    - Kontinuierliche Verbesserung
    - Statistiken-Tracking
    - Best-Score-Persistierung
    """
    
    def __init__(
        self,
        pipeline_coordinator: Any,
        training_data_dir: Path,
        config: Dict[str, Any]
    ):
        """
        Initialize Auto Trainer.
        
        Args:
            pipeline_coordinator: PipelineCoordinator instance
            training_data_dir: Directory with training images
            config: Configuration dictionary
        """
        self.pipeline_coordinator = pipeline_coordinator
        self.training_data_dir = training_data_dir
        self.config = config
        
        self.training_stats = {
            'total_cycles': 0,
            'total_images_processed': 0,
            'best_score': 0.0,
            'improvement_history': [],
            'last_training_time': None
        }
    
    def run_continuous_training(
        self,
        max_cycles: int = 0,
        duration_hours: float = 24.0,
        cycle_delay_seconds: float = 3600.0
    ) -> Dict[str, Any]:
        """
        Run continuous training cycles.
        
        Args:
            max_cycles: Maximum number of cycles (0 = unlimited)
            duration_hours: Total training duration in hours
            cycle_delay_seconds: Delay between cycles in seconds
            
        Returns:
            Training report
        """
        logger.info("=== Starting Continuous Training ===")
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600.0)
        cycle_count = 0
        
        while (max_cycles == 0 or cycle_count < max_cycles) and time.time() < end_time:
            cycle_count += 1
            logger.info(f"\n=== Training Cycle {cycle_count} ===")
            
            try:
                # Find training images
                training_images = self._find_training_images()
                if not training_images:
                    logger.warning("No training images found. Waiting for images...")
                    time.sleep(cycle_delay_seconds)
                    continue
                
                # Run training cycle
                cycle_report = self._run_training_cycle(training_images, cycle_count)
                
                # Update statistics
                self.training_stats['total_cycles'] += 1
                self.training_stats['total_images_processed'] += cycle_report.get('images_processed', 0)
                
                # Track improvement
                avg_score = cycle_report.get('average_score', 0.0)
                if avg_score > self.training_stats['best_score']:
                    improvement = avg_score - self.training_stats['best_score']
                    self.training_stats['best_score'] = avg_score
                    self.training_stats['improvement_history'].append({
                        'cycle': cycle_count,
                        'score': avg_score,
                        'improvement': improvement,
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.info(f"*** NEW BEST SCORE: {avg_score:.2f} (+{improvement:.2f}) ***")
                
                logger.info(f"Cycle {cycle_count} complete. Average score: {avg_score:.2f}")
                
                # Wait before next cycle
                if time.time() < end_time:
                    remaining_time = end_time - time.time()
                    delay = min(cycle_delay_seconds, remaining_time)
                    logger.info(f"Waiting {delay:.0f}s before next cycle...")
                    time.sleep(delay)
            
            except KeyboardInterrupt:
                logger.info("Training interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Error in training cycle {cycle_count}: {e}", exc_info=True)
                time.sleep(60)  # Wait before retry
        
        # Final report
        total_time = time.time() - start_time
        self.training_stats['last_training_time'] = datetime.now().isoformat()
        
        report = {
            'total_cycles': cycle_count,
            'total_time_hours': total_time / 3600.0,
            'best_score': self.training_stats['best_score'],
            'total_images_processed': self.training_stats['total_images_processed'],
            'improvement_count': len(self.training_stats['improvement_history'])
        }
        
        logger.info(f"\n=== Training Complete ===")
        logger.info(f"Total cycles: {cycle_count}")
        logger.info(f"Best score: {self.training_stats['best_score']:.2f}")
        logger.info(f"Total images processed: {self.training_stats['total_images_processed']}")
        
        return report
    
    def _find_training_images(self) -> List[Path]:
        """Find training images in training data directory."""
        training_images = []
        
        try:
            # Search for images in training data directory
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                training_images.extend(list(self.training_data_dir.rglob(ext)))
            
            # Filter out truth files and output files
            training_images = [
                img for img in training_images
                if not any(exclude in img.name.lower() for exclude in ['truth', 'output', 'result', 'cgm'])
            ]
            
            logger.info(f"Found {len(training_images)} training images")
        except Exception as e:
            logger.error(f"Error finding training images: {e}", exc_info=True)
        
        return training_images
    
    def _run_training_cycle(
        self,
        training_images: List[Path],
        cycle_number: int
    ) -> Dict[str, Any]:
        """Run a single training cycle."""
        logger.info(f"Running training cycle {cycle_number} with {len(training_images)} images...")
        
        scores = []
        processed = 0
        
        for image_path in training_images:
            try:
                # Run analysis
                result = self.pipeline_coordinator.process(
                    image_path=str(image_path),
                    output_dir=None
                )
                
                # Extract quality score
                quality_score = result.quality_score if hasattr(result, 'quality_score') else result.get('quality_score', 0.0)
                scores.append(quality_score)
                processed += 1
                
                logger.info(f"Processed {image_path.name}: Score = {quality_score:.2f}")
            
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'images_processed': processed,
            'average_score': avg_score,
            'scores': scores,
            'cycle_number': cycle_number
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()


