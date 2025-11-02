"""
Correction Learner - Learns from corrections and improves analysis over time.

Provides functionality for:
- Learning from error corrections
- Storing correction patterns
- Retrieving similar past corrections
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class CorrectionLearner:
    """
    Learns from corrections and stores patterns for future reference.
    """
    
    def __init__(self, learning_db_path: Path, llm_client: Any):
        """
        Initialize Correction Learner.
        
        Args:
            learning_db_path: Path to learning database JSON file
            llm_client: LLM client for generating embeddings
        """
        self.learning_db_path = learning_db_path
        self.llm_client = llm_client
        self.corrections: Dict[str, Dict[str, Any]] = {}
        self._load_corrections()
    
    def _load_corrections(self) -> None:
        """Load corrections from database."""
        try:
            if self.learning_db_path.exists():
                with open(self.learning_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.corrections = data.get('learned_corrections', {})
                    logger.info(f"Loaded {len(self.corrections)} corrections")
        except Exception as e:
            logger.error(f"Error loading corrections: {e}", exc_info=True)
            self.corrections = {}
    
    def _save_corrections(self) -> None:
        """Save corrections to database."""
        try:
            # Load existing data
            data = {}
            if self.learning_db_path.exists():
                with open(self.learning_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Update corrections
            data['learned_corrections'] = self.corrections
            
            # Save
            self.learning_db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(self.corrections)} corrections")
        except Exception as e:
            logger.error(f"Error saving corrections: {e}", exc_info=True)
    
    def learn_from_correction(
        self,
        problem: Dict[str, Any],
        correction: Dict[str, Any],
        image_name: Optional[str] = None
    ) -> str:
        """
        Learn from a correction.
        
        Args:
            problem: Problem description dictionary
            correction: Correction data dictionary
            image_name: Optional image name for reference
            
        Returns:
            Hash ID of the stored correction
        """
        try:
            # Generate hash from problem description
            problem_str = json.dumps(problem, sort_keys=True)
            correction_hash = hashlib.sha256(problem_str.encode()).hexdigest()[:16]
            
            # Store correction
            self.corrections[correction_hash] = {
                'problem': problem,
                'correction': correction,
                'image_name': image_name,
                'learned_timestamp': logging.time.time() if hasattr(logging, 'time') else None,
                'usage_count': 0
            }
            
            self._save_corrections()
            logger.info(f"Learned correction {correction_hash} for problem: {problem.get('type', 'unknown')}")
            
            return correction_hash
        except Exception as e:
            logger.error(f"Error learning from correction: {e}", exc_info=True)
            return ""
    
    def find_similar_corrections(
        self,
        problem: Dict[str, Any],
        top_k: int = 3
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find similar past corrections for a given problem.
        
        Args:
            problem: Problem description dictionary
            top_k: Number of top results to return
            
        Returns:
            List of tuples (correction_hash, correction_data)
        """
        if not self.corrections:
            return []
        
        try:
            # Simple matching based on problem type and description
            problem_type = problem.get('type', '')
            problem_desc = problem.get('description', '')
            
            matches = []
            for correction_hash, correction_data in self.corrections.items():
                stored_problem = correction_data.get('problem', {})
                stored_type = stored_problem.get('type', '')
                stored_desc = stored_problem.get('description', '')
                
                # Score based on type match and description similarity
                score = 0.0
                if problem_type == stored_type:
                    score += 1.0
                
                # Simple keyword matching
                problem_keywords = set(problem_desc.lower().split())
                stored_keywords = set(stored_desc.lower().split())
                if problem_keywords and stored_keywords:
                    overlap = len(problem_keywords & stored_keywords)
                    score += overlap / max(len(problem_keywords), len(stored_keywords))
                
                if score > 0:
                    matches.append((score, correction_hash, correction_data))
            
            # Sort by score and return top k
            matches.sort(reverse=True, key=lambda x: x[0])
            return [(hash_id, data) for score, hash_id, data in matches[:top_k]]
        
        except Exception as e:
            logger.error(f"Error finding similar corrections: {e}", exc_info=True)
            return []
    
    def apply_correction(
        self,
        correction_hash: str,
        current_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Apply a learned correction to current data.
        
        Args:
            correction_hash: Hash of the correction to apply
            current_data: Current analysis data
            
        Returns:
            Corrected data or None if correction not found
        """
        if correction_hash not in self.corrections:
            return None
        
        try:
            correction = self.corrections[correction_hash]
            correction_data = correction.get('correction', {})
            corrected_data = correction_data.get('corrected_data', {})
            
            # Update usage count
            self.corrections[correction_hash]['usage_count'] = \
                self.corrections[correction_hash].get('usage_count', 0) + 1
            self._save_corrections()
            
            # Merge correction with current data
            # This is a simple merge - can be enhanced with smarter merging logic
            result = current_data.copy()
            result.update(corrected_data)
            
            logger.info(f"Applied correction {correction_hash}")
            return result
        except Exception as e:
            logger.error(f"Error applying correction: {e}", exc_info=True)
            return None
    
    def get_correction(self, correction_hash: str) -> Optional[Dict[str, Any]]:
        """Get correction data by hash."""
        return self.corrections.get(correction_hash)
    
    def get_correction_count(self) -> int:
        """Get the number of learned corrections."""
        return len(self.corrections)


