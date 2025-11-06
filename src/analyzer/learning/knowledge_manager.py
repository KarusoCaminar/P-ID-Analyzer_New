"""
Knowledge Manager - Refactored from knowledge_bases.py.

Manages static and dynamic knowledge including:
- Element type library
- Type aliases
- Learning database
- Symbol library
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import threading
from filelock import FileLock, Timeout
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    Manages static and dynamic knowledge for P&ID analysis.
    
    This is a refactored version of the original knowledge_bases.py with:
    - Cleaner interfaces
    - Thread-safe operations
    - Better type hints
    - Improved error handling
    """
    
    def __init__(
        self,
        element_type_list_path: str,
        learning_db_path: str,
        llm_handler: Any,  # Will be properly typed when LLM client is created
        config: Dict[str, Any]
    ):
        self.element_type_list_path = Path(element_type_list_path)
        self.learning_db_path = Path(learning_db_path)
        self.llm_handler = llm_handler
        self.config = config
        
        # Static knowledge
        self.config_library: List[Dict[str, Any]] = []
        self._type_name_to_id: Dict[str, str] = {}
        self._type_id_to_data: Dict[str, Dict] = {}
        
        # Dynamic knowledge (learning database)
        self.learning_database: Dict[str, Any] = {}
        
        # Vector indices for fast similarity search
        self._symbol_vector_index: Optional[np.ndarray] = None
        self._symbol_vector_data: List[Dict[str, Any]] = []
        self._symbol_vector_ids: List[str] = []
        
        self._solution_vector_index: Optional[np.ndarray] = None
        self._solution_vector_keys: List[str] = []
        self._solution_vector_solutions: List[Dict[str, Any]] = []
        
        # Thread safety
        file_lock_timeout = config.get('logic_parameters', {}).get('db_file_lock_timeout', 15)
        self.lock_path = str(self.learning_db_path) + ".lock"
        self.db_process_lock = FileLock(self.lock_path, timeout=file_lock_timeout)
        self.db_thread_lock = threading.Lock()
        
        # Load knowledge
        self._load_config_library()
        self._load_learning_database()
    
    def _load_config_library(self) -> None:
        """Load base type library from element_type_list.json."""
        logger.info(f"Loading base knowledge from: {self.element_type_list_path}")
        try:
            with open(self.element_type_list_path, 'r', encoding='utf-8') as f:
                self.config_library = json.load(f)
            
            for element_type in self.config_library:
                type_id = element_type.get('id')
                type_name = element_type.get('name', '')
                if type_id and type_name:
                    normalized_name = self._normalize_label(type_name)
                    self._type_name_to_id[normalized_name] = type_id
                    self._type_id_to_data[type_id] = element_type
            
            logger.info(f"Successfully loaded {len(self.config_library)} base types")
        except Exception as e:
            logger.error(f"FATAL: Could not load base knowledge: {e}", exc_info=True)
            self.config_library = []
    
    def _load_learning_database(self) -> None:
        """Load learning database and build vector indices."""
        logger.info(f"Loading learning database from: {self.learning_db_path}")
        try:
            if self.learning_db_path.exists():
                with open(self.learning_db_path, 'r', encoding='utf-8') as f:
                    self.learning_database = json.load(f)
                logger.info("Learning database loaded successfully")
            else:
                logger.warning("Learning database not found. Creating new database")
                self._init_empty_database()
        except Exception as e:
            logger.error(f"Error loading learning database: {e}", exc_info=True)
            self._init_empty_database()
        
        # Ensure all required keys exist
        self._ensure_database_structure()
        
        # Build vector indices
        self._build_vector_index()
    
    def _init_empty_database(self) -> None:
        """Initialize empty learning database structure."""
        self.learning_database = {
            "knowledge_extensions": {"type_aliases": {}},
            "successful_patterns": {},
            "error_stats": {},
            "learned_solutions": {},
            "symbol_library": {},
            "learned_visual_corrections": {}
        }
    
    def _ensure_database_structure(self) -> None:
        """Ensure all required database keys exist."""
        self.learning_database.setdefault("knowledge_extensions", {"type_aliases": {}})
        self.learning_database["knowledge_extensions"].setdefault("type_aliases", {})
        self.learning_database.setdefault("successful_patterns", {})
        self.learning_database.setdefault("error_stats", {})
        self.learning_database.setdefault("learned_solutions", {})
        self.learning_database.setdefault("symbol_library", {})
        self.learning_database.setdefault("learned_visual_corrections", {})
    
    def _build_vector_index(self) -> None:
        """Build in-memory vector indices for fast similarity search."""
        logger.info("Building vector indices...")
        
        # Build solution index (text-based embeddings)
        learned_solutions = self.learning_database.get("learned_solutions", {})
        vectors_as_lists = []
        solution_keys_filtered = []
        solution_values_filtered = []
        
        if learned_solutions:
            for key, solution_data in learned_solutions.items():
                embedding = solution_data.get("problem_embedding")
                if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                    vectors_as_lists.append(embedding)
                    solution_keys_filtered.append(key)
                    solution_values_filtered.append(solution_data)
            
            if vectors_as_lists:
                self._solution_vector_index = np.array(vectors_as_lists, dtype=np.float32)
                self._solution_vector_keys = solution_keys_filtered
                self._solution_vector_solutions = solution_values_filtered
                logger.info(f"Solution vector index built with {len(self._solution_vector_index)} entries")
            else:
                self._solution_vector_index = None
                self._solution_vector_keys = []
                self._solution_vector_solutions = []
        
        # Build symbol index (visual embeddings)
        symbol_library = self.learning_database.get("symbol_library", {})
        if symbol_library:
            valid_symbols_data = [
                data for data in symbol_library.values()
                if isinstance(data, dict) and isinstance(data.get("visual_embedding"), list)
            ]
            if valid_symbols_data:
                self._symbol_vector_ids = [
                    sym_id for sym_id, data in symbol_library.items()
                    if isinstance(data, dict) and isinstance(data.get("visual_embedding"), list)
                ]
                self._symbol_vector_data = valid_symbols_data
                vectors_as_lists_symbols = [data["visual_embedding"] for data in valid_symbols_data]
                self._symbol_vector_index = np.array(vectors_as_lists_symbols, dtype=np.float32)
                logger.info(f"Symbol vector index built with {len(self._symbol_vector_index)} entries")
            else:
                self._symbol_vector_index = None
                self._symbol_vector_ids = []
                self._symbol_vector_data = []
    
    def _convert_bbox_to_dict(self, obj: Any) -> Any:
        """
        Convert BBox objects to dictionaries for JSON serialization.
        
        Args:
            obj: Object that might contain BBox objects
            
        Returns:
            Object with BBox objects converted to dicts
        """
        # Handle BBox Pydantic models (check class name first)
        if type(obj).__name__ == 'BBox' or (hasattr(obj, '__class__') and 'BBox' in str(obj.__class__)):
            # BBox Pydantic model
            if hasattr(obj, 'model_dump'):
                # Pydantic v2
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                # Pydantic v1
                return obj.dict()
            elif hasattr(obj, '__dict__'):
                # Fallback: use __dict__
                return {
                    'x': float(getattr(obj, 'x', 0)),
                    'y': float(getattr(obj, 'y', 0)),
                    'width': float(getattr(obj, 'width', 0)),
                    'height': float(getattr(obj, 'height', 0))
                }
        
        # Handle other Pydantic models
        if hasattr(obj, 'model_dump'):
            # Pydantic v2 model
            try:
                return obj.model_dump()
            except Exception:
                # Fallback to dict conversion
                pass
        elif hasattr(obj, 'dict'):
            # Pydantic v1 style
            try:
                return obj.dict()
            except Exception:
                pass
        
        # Handle BBox-like objects with attributes
        if hasattr(obj, '__dict__') and hasattr(obj, 'x') and hasattr(obj, 'y') and hasattr(obj, 'width') and hasattr(obj, 'height'):
            return {
                'x': float(getattr(obj, 'x', 0)),
                'y': float(getattr(obj, 'y', 0)),
                'width': float(getattr(obj, 'width', 0)),
                'height': float(getattr(obj, 'height', 0))
            }
        elif isinstance(obj, dict):
            # Recursively process dictionary
            return {k: self._convert_bbox_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively process list/tuple
            return [self._convert_bbox_to_dict(item) for item in obj]
        else:
            # Return as-is if not a BBox object
            return obj
    
    def save_learning_database(self) -> None:
        """Save learning database thread-safe and process-safe."""
        logger.info(f"Saving learning database to: {self.learning_db_path}")
        try:
            with self.db_process_lock:
                with self.db_thread_lock:
                    # Convert BBox objects to dicts before serialization
                    serializable_db = self._convert_bbox_to_dict(self.learning_database)
                    
                    # Write to temporary string first for validation
                    temp_json_string = json.dumps(
                        serializable_db,
                        ensure_ascii=False,
                        indent=4
                    )
                    self.learning_db_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.learning_db_path, 'w', encoding='utf-8') as f:
                        f.write(temp_json_string)
                    logger.info("Learning database saved successfully")
        except Timeout:
            logger.error(f"Could not acquire lock for {self.lock_path}. Save skipped.")
        except Exception as e:
            logger.error(f"Error saving learning database: {e}", exc_info=True)
    
    def get_known_types(self) -> List[str]:
        """Get list of known element type names."""
        return list(self._type_name_to_id.keys())
    
    def find_element_type_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find element type by name (with alias support)."""
        normalized = self._normalize_label(name)
        
        # Try direct match
        type_id = self._type_name_to_id.get(normalized)
        if type_id:
            return self._type_id_to_data.get(type_id)
        
        # Try aliases
        aliases = self.get_all_aliases()
        for canonical, alias_list in aliases.items():
            if normalized in [self._normalize_label(a) for a in alias_list]:
                type_id = self._type_name_to_id.get(self._normalize_label(canonical))
                if type_id:
                    return self._type_id_to_data.get(type_id)
        
        return None
    
    def get_all_aliases(self) -> Dict[str, List[str]]:
        """Get all type aliases."""
        return self.learning_database.get("knowledge_extensions", {}).get("type_aliases", {})
    
    def add_type_alias(self, canonical_type: str, alias: str) -> None:
        """Add a type alias."""
        aliases = self.learning_database.setdefault("knowledge_extensions", {}).setdefault("type_aliases", {})
        if canonical_type not in aliases:
            aliases[canonical_type] = []
        if alias not in aliases[canonical_type]:
            aliases[canonical_type].append(alias)
            self.save_learning_database()
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label for matching."""
        if not label:
            return ""
        import re
        s = label.lower().strip()
        s = re.sub(r"[\s\-_/]+", "", s)
        s = re.sub(r"[^\w\d]", "", s)
        return s
    
    def clean_learning_database(self, max_age_days: int = 90, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Clean learning database to remove outdated, low-quality, or duplicate entries.
        
        Args:
            max_age_days: Maximum age in days for analysis metadata (default: 90)
            min_confidence: Minimum confidence for storing patterns (default: 0.5)
            
        Returns:
            Cleaning report with statistics
        """
        logger.info("=== Cleaning learning database ===")
        
        report = {
            'entries_removed': 0,
            'entries_kept': 0,
            'duplicates_removed': 0,
            'low_quality_removed': 0,
            'outdated_removed': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            # Clean recent_analyses
            if 'recent_analyses' in self.learning_database:
                recent_analyses = self.learning_database.get('recent_analyses', [])
                original_count = len(recent_analyses)
                
                # Remove outdated entries
                cleaned_analyses = []
                for analysis in recent_analyses:
                    try:
                        timestamp_str = analysis.get('timestamp')
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if timestamp.replace(tzinfo=None) < cutoff_date:
                                report['outdated_removed'] += 1
                                continue
                        
                        # Keep high-quality analyses
                        quality_score = analysis.get('quality_score', 0.0)
                        avg_confidence = analysis.get('avg_confidence', 0.0)
                        if quality_score >= 30.0 or avg_confidence >= min_confidence:
                            cleaned_analyses.append(analysis)
                            report['entries_kept'] += 1
                        else:
                            report['low_quality_removed'] += 1
                    except Exception as e:
                        logger.warning(f"Error processing analysis entry: {e}")
                        report['low_quality_removed'] += 1
                
                # Keep only last 100 analyses
                if len(cleaned_analyses) > 100:
                    cleaned_analyses = cleaned_analyses[-100:]
                    report['entries_removed'] += len(recent_analyses) - 100
                
                self.learning_database['recent_analyses'] = cleaned_analyses
                report['entries_removed'] += original_count - len(cleaned_analyses)
            
            # Clean successful_patterns (remove duplicates and low-quality)
            if 'successful_patterns' in self.learning_database:
                patterns = self.learning_database.get('successful_patterns', {})
                cleaned_patterns = {}
                seen_patterns = set()
                
                for pattern_key, pattern_data in patterns.items():
                    # Check for duplicates
                    pattern_hash = json.dumps(pattern_data, sort_keys=True)
                    if pattern_hash in seen_patterns:
                        report['duplicates_removed'] += 1
                        continue
                    seen_patterns.add(pattern_hash)
                    
                    # Check quality
                    confidence = pattern_data.get('confidence', 0.0)
                    if confidence >= min_confidence:
                        cleaned_patterns[pattern_key] = pattern_data
                        report['entries_kept'] += 1
                    else:
                        report['low_quality_removed'] += 1
                
                self.learning_database['successful_patterns'] = cleaned_patterns
            
            # Clean learned_solutions (remove duplicates and outdated)
            if 'learned_solutions' in self.learning_database:
                solutions = self.learning_database.get('learned_solutions', {})
                cleaned_solutions = {}
                seen_solutions = set()
                
                for solution_key, solution_data in solutions.items():
                    # Check for duplicates
                    solution_hash = json.dumps(solution_data, sort_keys=True)
                    if solution_hash in seen_solutions:
                        report['duplicates_removed'] += 1
                        continue
                    seen_solutions.add(solution_hash)
                    
                    # Check timestamp
                    try:
                        timestamp_str = solution_data.get('timestamp')
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if timestamp.replace(tzinfo=None) < cutoff_date:
                                report['outdated_removed'] += 1
                                continue
                    except (ValueError, AttributeError, TypeError) as e:
                        logger.debug(f"Invalid timestamp in analysis entry: {e}")
                        pass  # Keep if timestamp is invalid
                    
                    cleaned_solutions[solution_key] = solution_data
                    report['entries_kept'] += 1
                
                self.learning_database['learned_solutions'] = cleaned_solutions
            
            # Save cleaned database
            self.save_learning_database()
            
            logger.info(f"Database cleaned: {report['entries_removed']} removed, "
                       f"{report['entries_kept']} kept, "
                       f"{report['duplicates_removed']} duplicates removed")
            
        except Exception as e:
            logger.error(f"Error cleaning learning database: {e}", exc_info=True)
        
        return report
    
    def extract_key_learnings(self) -> Dict[str, Any]:
        """
        Extract key learnings from recent analyses and store them for future use.
        
        Returns:
            Key learnings report with extracted patterns
        """
        logger.info("=== Extracting key learnings ===")
        
        key_learnings = {
            'critical_errors': [],
            'successful_patterns': [],
            'type_mappings': {},
            'confidence_calibration': {},
            'common_mistakes': [],
            'best_practices': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            from collections import Counter
            
            # Analyze recent analyses
            recent_analyses = self.learning_database.get('recent_analyses', [])
            
            if not recent_analyses:
                logger.warning("No recent analyses found for key learning extraction")
                return key_learnings
            
            # Extract common element types
            element_types = []
            high_confidence_types = []
            low_confidence_types = []
            
            for analysis in recent_analyses:
                types = analysis.get('element_types', [])
                element_types.extend(types)
                
                avg_confidence = analysis.get('avg_confidence', 0.0)
                quality_score = analysis.get('quality_score', 0.0)
                
                if avg_confidence >= 0.8 and quality_score >= 50.0:
                    high_confidence_types.extend(types)
                elif avg_confidence < 0.5 or quality_score < 30.0:
                    low_confidence_types.extend(types)
            
            # Most common types (successful patterns)
            type_counter = Counter(element_types)
            key_learnings['successful_patterns'] = [
                {'type': type_name, 'frequency': count}
                for type_name, count in type_counter.most_common(10)
            ]
            
            # Types with high confidence
            high_type_counter = Counter(high_confidence_types)
            for type_name, count in high_type_counter.most_common(5):
                key_learnings['type_mappings'][type_name] = {
                    'confidence': 'high',
                    'frequency': count
                }
            
            # Types with low confidence (common mistakes)
            low_type_counter = Counter(low_confidence_types)
            key_learnings['common_mistakes'] = [
                {'type': type_name, 'frequency': count, 'issue': 'low_confidence'}
                for type_name, count in low_type_counter.most_common(5)
            ]
            
            # Extract confidence calibration (works with or without truth data)
            confidence_scores = [a.get('avg_confidence', 0.0) for a in recent_analyses if a.get('avg_confidence')]
            quality_scores = [a.get('quality_score', 0.0) for a in recent_analyses if a.get('quality_score')]
            
            # If no quality scores from truth data, calculate internal quality scores
            if confidence_scores and not quality_scores:
                # Calculate internal quality score based on structural metrics
                internal_quality_scores = []
                for analysis in recent_analyses:
                    avg_conf = analysis.get('avg_confidence', 0.0)
                    num_elements = analysis.get('num_elements', 0)
                    num_connections = analysis.get('num_connections', 0)
                    graph_density = analysis.get('graph_density', 0.0)
                    isolated_elements = analysis.get('isolated_elements', 0)
                    
                    # Internal quality score: based on structural metrics
                    internal_score = 0.0
                    if avg_conf > 0:
                        internal_score = avg_conf * 50.0  # Base score from confidence
                    if num_elements > 0:
                        internal_score += min(num_elements * 0.5, 25.0)  # Max 25 points for elements
                    if num_connections > 0:
                        internal_score += min(num_connections * 0.3, 15.0)  # Max 15 points for connections
                    if graph_density > 0:
                        internal_score += min(graph_density * 10.0, 10.0)  # Max 10 points for density
                    # Penalty for isolated elements
                    if isolated_elements > 0:
                        internal_score -= min(isolated_elements * 0.5, 5.0)
                    
                    internal_quality_scores.append(max(0.0, min(100.0, internal_score)))
                
                quality_scores = internal_quality_scores
            
            if confidence_scores and quality_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                avg_quality = sum(quality_scores) / len(quality_scores)
                
                # Calibration offset: difference between expected (confidence * 100) and actual quality
                # Positive offset means confidence is too low, negative means too high
                calibration_offset_normalized = (avg_quality / 100.0) - avg_confidence
                
                key_learnings['confidence_calibration'] = {
                    'avg_confidence': avg_confidence,
                    'avg_quality': avg_quality,
                    'calibration_offset': calibration_offset_normalized,
                    'based_on_truth_data': any(a.get('has_truth_data', False) for a in recent_analyses)
                }
            
            # Store in learning database
            self.learning_database['key_learnings'] = key_learnings
            self.save_learning_database()
            
            logger.info(f"Key learnings extracted: {len(key_learnings['successful_patterns'])} patterns, "
                       f"{len(key_learnings['common_mistakes'])} common mistakes")
            
        except Exception as e:
            logger.error(f"Error extracting key learnings: {e}", exc_info=True)
        
        return key_learnings
    
    def get_confidence_calibration(self) -> float:
        """
        Get confidence calibration offset from learned data.
        
        Returns:
            Calibration offset to apply to confidence scores (normalized: -1.0 to 1.0)
            Returns 0.0 if no calibration data available
        """
        try:
            key_learnings = self.learning_database.get('key_learnings', {})
            calibration = key_learnings.get('confidence_calibration', {})
            offset = calibration.get('calibration_offset', 0.0)
            # Ensure offset is in reasonable range
            return max(-1.0, min(1.0, offset))
        except Exception as e:
            logger.warning(f"Error retrieving confidence calibration: {e}")
            return 0.0
    
    def track_critical_errors(
        self,
        errors: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track critical errors for future reference and improvement.
        
        Args:
            errors: List of critical errors
            context: Optional context (image_path, timestamp, etc.)
        """
        if not errors:
            return
        
        try:
            error_stats = self.learning_database.setdefault('error_stats', {})
            critical_errors = error_stats.setdefault('critical_errors', [])
            
            for error in errors:
                error_entry = {
                    'type': error.get('type', 'unknown'),
                    'message': error.get('message', ''),
                    'severity': error.get('severity', 'medium'),
                    'timestamp': datetime.now().isoformat(),
                    'context': context or {}
                }
                critical_errors.append(error_entry)
            
            # Keep only last 100 critical errors
            if len(critical_errors) > 100:
                critical_errors[:] = critical_errors[-100:]
            
            # Aggregate error statistics
            error_type_counter = {}
            for error in critical_errors:
                error_type = error.get('type', 'unknown')
                error_type_counter[error_type] = error_type_counter.get(error_type, 0) + 1
            
            error_stats['error_type_counts'] = error_type_counter
            error_stats['total_critical_errors'] = len(critical_errors)
            error_stats['last_updated'] = datetime.now().isoformat()
            
            self.save_learning_database()
            
            logger.info(f"Tracked {len(errors)} critical errors")
            
        except Exception as e:
            logger.error(f"Error tracking critical errors: {e}", exc_info=True)
    
    # Additional methods from original knowledge_bases.py can be migrated here
    # For brevity, I'm including the core structure. Full migration would include:
    # - learn_from_correction
    # - find_solution_for_problem
    # - integrate_symbol_library
    # - add_learning_pattern
    # - etc.

