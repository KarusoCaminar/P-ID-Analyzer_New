#!/usr/bin/env python3
"""
Visual Trace Debug Script - Phase A: Diagnostic & Audit

Creates visual debug images at critical checkpoints:
- Checkpoint 1: After Phase 2c (Fusion Engine)
- Checkpoint 2: After Phase 3 (After Iteration 1 of Critic/Refinement)
- Checkpoint 3: After Phase 3 (After final iteration/Early Stop)
- Checkpoint 4: After Phase 4 (After final Post-Processing)
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.visualization.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualTraceProgressCallback:
    """Progress callback that saves debug images at checkpoints."""
    
    def __init__(self, output_dir: Path, image_path: str):
        self.output_dir = output_dir
        self.image_path = image_path
        self.visualizer = Visualizer(image_path)
        self.checkpoint_counter = 0
        
    def update_progress(self, progress: int, message: str):
        logger.info(f"[Progress {progress}%] {message}")
        
        # Check for checkpoint triggers
        if "Phase 2c: Fusion" in message and "completed" in message.lower():
            self._save_checkpoint("checkpoint_1_after_fusion")
        elif "Phase 3" in message and "iteration 1" in message.lower():
            self._save_checkpoint("checkpoint_2_after_iteration_1")
        elif "Phase 3" in message and ("completed" in message.lower() or "early stop" in message.lower()):
            self._save_checkpoint("checkpoint_3_after_phase_3_final")
        elif "Phase 4" in message and "completed" in message.lower():
            self._save_checkpoint("checkpoint_4_after_post_processing")
    
    def update_status_label(self, text: str):
        logger.info(f"[Status] {text}")
    
    def report_truth_mode(self, active: bool):
        logger.info(f"[Truth Mode] {'ACTIVE' if active else 'INACTIVE'}")
    
    def report_correction(self, correction_text: str):
        logger.info(f"[Correction] {correction_text}")
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save debug image at checkpoint."""
        try:
            # Get current state from pipeline (we'll need to modify pipeline_coordinator to expose this)
            # For now, we'll save after the fact using the coordinator's state
            logger.info(f"📸 Checkpoint: {checkpoint_name} - Saving debug image...")
            # This will be called from within pipeline_coordinator at the right moments
        except Exception as e:
            logger.error(f"Error saving checkpoint {checkpoint_name}: {e}")


class VisualTracePipelineCoordinator(PipelineCoordinator):
    """Extended PipelineCoordinator that saves debug images at checkpoints."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visualizer = None
        self.checkpoint_output_dir = None
        
    def set_visual_trace_mode(self, output_dir: Path, image_path: str):
        """Enable visual trace mode."""
        self.checkpoint_output_dir = output_dir
        self.visualizer = Visualizer(image_path)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_checkpoint_image(self, checkpoint_name: str, elements: list, connections: list):
        """Save debug image at checkpoint."""
        if not self.checkpoint_output_dir or not self.visualizer:
            return
        
        try:
            checkpoint_path = self.checkpoint_output_dir / f"{checkpoint_name}.png"
            self.visualizer.draw_debug_map(
                image_path=self.current_image_path,
                elements=elements,
                connections=connections,
                output_path=str(checkpoint_path)
            )
            logger.info(f"📸 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint {checkpoint_name}: {e}")
    
    def _run_phase_2c_fusion(self, swarm_result: Dict[str, Any], monolith_result: Dict[str, Any]) -> Dict[str, Any]:
        """Override to save checkpoint after fusion."""
        result = super()._run_phase_2c_fusion(swarm_result, monolith_result)
        
        # Save checkpoint 1: After Fusion
        fused_elements = result.get("elements", [])
        fused_connections = result.get("connections", [])
        self._save_checkpoint_image("checkpoint_1_after_fusion", fused_elements, fused_connections)
        
        return result
    
    def _run_phase_3_self_correction_loop(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Override to save checkpoints during self-correction loop."""
        # Store initial state for iteration tracking
        iteration_count = 0
        
        # Call parent method but intercept at key points
        result = super()._run_phase_3_self_correction_loop(image_path, output_dir)
        
        # Extract iteration data from result
        score_history = result.get("final_ai_data", {}).get("score_history", [])
        if len(score_history) >= 1:
            # Checkpoint 2: After iteration 1
            elements = result.get("final_ai_data", {}).get("elements", [])
            connections = result.get("final_ai_data", {}).get("connections", [])
            self._save_checkpoint_image("checkpoint_2_after_iteration_1", elements, connections)
        
        # Checkpoint 3: After final iteration
        final_elements = result.get("final_ai_data", {}).get("elements", [])
        final_connections = result.get("final_ai_data", {}).get("connections", [])
        self._save_checkpoint_image("checkpoint_3_after_phase_3_final", final_elements, final_connections)
        
        return result
    
    def _run_phase_4_post_processing(self, best_result: Dict[str, Any], image_path: str, output_dir: str, truth_data: Optional[Dict[str, Any]]) -> Any:
        """Override to save checkpoint after post-processing."""
        result = super()._run_phase_4_post_processing(best_result, image_path, output_dir, truth_data)
        
        # Checkpoint 4: After Post-Processing
        final_data = best_result.get("final_ai_data", {})
        elements = final_data.get("elements", [])
        connections = final_data.get("connections", [])
        self._save_checkpoint_image("checkpoint_4_after_post_processing", elements, connections)
        
        return result


def run_visual_trace(image_path: Path, truth_path: Optional[Path], output_dir: Path):
    """Run visual trace analysis."""
    logger.info(f"\n{'='*80}\n"
                f"VISUAL TRACE DEBUG - Phase A: Diagnostic & Audit\n"
                f"{'='*80}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create visual trace coordinator
    pipeline = VisualTracePipelineCoordinator(
        llm_client=llm_client,
        knowledge_manager=knowledge_manager,
        config_service=config_service,
        progress_callback=VisualTraceProgressCallback(checkpoint_dir, str(image_path))
    )
    
    # Enable visual trace mode
    pipeline.set_visual_trace_mode(checkpoint_dir, str(image_path))
    
    try:
        result = pipeline.process(
            image_path=str(image_path),
            output_dir=str(output_dir),
            params_override={'truth_data_path': str(truth_path) if truth_path else None}
        )
        
        logger.info(f"\n{'='*80}\n"
                   f"VISUAL TRACE COMPLETE\n"
                   f"Checkpoints saved to: {checkpoint_dir}\n"
                   f"Quality Score: {result.quality_score:.2f}%\n"
                   f"{'='*80}\n")
        
        return result.model_dump()
    except Exception as e:
        logger.error(f"Visual trace failed: {e}", exc_info=True)
        return {"error": str(e)}


def main():
    """Main function."""
    output_base_dir = project_root / "outputs" / "visual_trace_debug"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Use simple P&ID for visual trace
    simple_pid_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
    simple_pid_truth_path = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
    
    if not simple_pid_path.exists():
        logger.error(f"Image not found: {simple_pid_path}")
        return
    
    result = run_visual_trace(
        image_path=simple_pid_path,
        truth_path=simple_pid_truth_path,
        output_dir=output_base_dir / "simple_pid"
    )
    
    # Save results
    with open(output_base_dir / "visual_trace_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Visual trace results saved to: {output_base_dir / 'visual_trace_results.json'}")


if __name__ == "__main__":
    main()

