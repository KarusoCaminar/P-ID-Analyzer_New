"""
Pretraining Orchestrator - Handles automatic pretraining and active learning.

Integrates with ActiveLearner to automatically learn from pretraining symbols
and continuously improve the system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from src.analyzer.learning.active_learner import ActiveLearner
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.ai.llm_client import LLMClient

logger = logging.getLogger(__name__)


class PretrainingOrchestrator:
    """
    Orchestrates pretraining and active learning from symbols.
    """
    
    def __init__(
        self,
        active_learner: ActiveLearner,
        llm_client: LLMClient,
        config: Dict[str, Any]
    ):
        """
        Initialize pretraining orchestrator.
        
        Args:
            active_learner: ActiveLearner instance
            llm_client: LLMClient instance
            config: Configuration dictionary
        """
        self.active_learner = active_learner
        self.llm_client = llm_client
        self.config = config
    
    def run_pretraining(
        self,
        pretraining_path: Path,
        model_info: Dict[str, Any],
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run pretraining workflow using active learning.
        
        Args:
            pretraining_path: Path to pretraining symbols directory
            model_info: Model configuration
            progress_callback: Optional progress callback
            
        Returns:
            Pretraining report
        """
        logger.info("=== Starting Pretraining with Active Learning ===")
        
        if progress_callback:
            progress_callback.update_status_label("Starting pretraining...")
        
        # Use active learner to learn from pretraining symbols
        learning_report = self.active_learner.learn_from_pretraining_symbols(
            pretraining_path=pretraining_path,
            model_info=model_info
        )
        
        logger.info(f"Pretraining complete: {learning_report.get('symbols_learned', 0)} symbols learned")
        
        if progress_callback:
            progress_callback.update_status_label(
                f"Pretraining complete: {learning_report.get('symbols_learned', 0)} symbols learned"
            )
        
        return learning_report


