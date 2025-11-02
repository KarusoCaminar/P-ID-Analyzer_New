"""
Active Learner - Self-training system that learns continuously from analysis results.

Provides:
- Automatic learning from pretraining symbols
- Continuous improvement from analysis feedback
- Adaptive learning from dataset patterns
- Online learning capabilities
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ActiveLearner:
    """
    Active learning system that continuously improves from experience.
    
    Features:
    - Automatic symbol learning from pretraining
    - Learning from analysis results
    - Pattern recognition and adaptation
    - Self-improvement loops
    """
    
    def __init__(
        self,
        knowledge_manager: Any,
        symbol_library: Any,
        llm_client: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize Active Learner.
        
        Args:
            knowledge_manager: KnowledgeManager instance
            symbol_library: SymbolLibrary instance
            llm_client: LLMClient instance
            config: Configuration dictionary
        """
        self.knowledge_manager = knowledge_manager
        self.symbol_library = symbol_library
        self.llm_client = llm_client
        self.config = config
        
        self.learning_stats = {
            'symbols_learned': 0,
            'patterns_learned': 0,
            'corrections_applied': 0,
            'last_learning_time': None
        }
    
    def learn_from_pretraining_symbols(
        self,
        pretraining_path: Path,
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Automatically learn from pretraining symbols.
        
        Args:
            pretraining_path: Path to pretraining symbols directory
            model_info: Model configuration for symbol extraction
            
        Returns:
            Learning report
        """
        logger.info("=== Active Learning: Learning from Pretraining Symbols ===")
        
        report = {
            'symbols_processed': 0,
            'symbols_learned': 0,
            'errors': []
        }
        
        try:
            # Find all symbol images
            image_paths = list(pretraining_path.glob("*.png")) + \
                         list(pretraining_path.glob("*.jpg")) + \
                         list(pretraining_path.glob("*.jpeg"))
            
            if not image_paths:
                logger.warning(f"No symbol images found in {pretraining_path}")
                return report
            
            logger.info(f"Processing {len(image_paths)} pretraining symbols...")
            
            for img_path in image_paths:
                try:
                    # Load and process symbol image
                    image = Image.open(img_path)
                    
                    # Extract symbol information using LLM
                    symbol_info = self._extract_symbol_info(image, model_info)
                    if not symbol_info:
                        continue
                    
                    # Add to symbol library
                    symbol_id = self._generate_symbol_id(img_path, symbol_info)
                    element_type = symbol_info.get('type', 'Unknown')
                    metadata = {
                        'source': 'pretraining',
                        'image_path': str(img_path),
                        'extracted_info': symbol_info,
                        'learned_timestamp': datetime.now().isoformat()
                    }
                    
                    success = self.symbol_library.add_symbol(
                        symbol_id=symbol_id,
                        image=image,
                        element_type=element_type,
                        metadata=metadata
                    )
                    
                    if success:
                        report['symbols_learned'] += 1
                        self.learning_stats['symbols_learned'] += 1
                        logger.info(f"Learned symbol: {symbol_id} ({element_type})")
                    
                    report['symbols_processed'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {img_path}: {e}"
                    logger.error(error_msg, exc_info=True)
                    report['errors'].append(error_msg)
            
            self.learning_stats['last_learning_time'] = datetime.now().isoformat()
            logger.info(f"Pretraining learning complete: {report['symbols_learned']} symbols learned")
            
        except Exception as e:
            logger.error(f"Error in learn_from_pretraining_symbols: {e}", exc_info=True)
            report['errors'].append(str(e))
        
        return report
    
    def learn_from_analysis_result(
        self,
        analysis_result: Dict[str, Any],
        truth_data: Optional[Dict[str, Any]] = None,
        quality_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Learn from analysis results and improve.
        
        Args:
            analysis_result: Analysis result dictionary
            truth_data: Optional ground truth for supervised learning
            quality_score: Quality score of the analysis
            
        Returns:
            Learning report
        """
        logger.info("=== Active Learning: Learning from Analysis Result ===")
        
        report = {
            'patterns_learned': 0,
            'corrections_learned': 0,
            'symbols_updated': 0,
            'errors': []
        }
        
        try:
            elements = analysis_result.get('elements', [])
            connections = analysis_result.get('connections', [])
            
            # LIVE LEARNING: Learn from ALL results immediately (not just high quality)
            # Lower threshold for live learning to learn from all runs
            learn_threshold = 0.5  # Learn from quality >= 50%
            
            if quality_score >= learn_threshold:
                # Extract patterns from current analysis
                patterns = self._extract_successful_patterns(elements, connections)
                for pattern in patterns:
                    self._store_pattern(pattern)
                    report['patterns_learned'] += 1
                    self.learning_stats['patterns_learned'] += 1
                logger.info(f"Live learning: Learned {len(patterns)} successful patterns")
            
            # LIVE LEARNING: Always learn from high confidence elements (even if overall score is low)
            high_confidence_elements = [el for el in elements if el.get('confidence', 0) >= 0.8]
            if high_confidence_elements:
                # Store these as positive examples for future reference
                for el in high_confidence_elements[:10]:  # Limit to 10 for performance
                    pattern = {
                        'type': 'high_confidence_element',
                        'element_type': el.get('type'),
                        'bbox': el.get('bbox'),
                        'confidence': el.get('confidence'),
                        'timestamp': datetime.now().isoformat()
                    }
                    self._store_pattern(pattern)
                    report['patterns_learned'] += 1
                logger.info(f"Live learning: Stored {len(high_confidence_elements)} high-confidence elements as examples")
            
            # LIVE LEARNING: Learn from corrections IMMEDIATELY (supervised learning with truth data)
            if truth_data:
                corrections = self._compare_with_truth(analysis_result, truth_data)
                for correction in corrections:
                    self._learn_correction(correction)
                    report['corrections_learned'] += 1
                    self.learning_stats['corrections_applied'] += 1
                logger.info(f"Live learning: Learned {len(corrections)} corrections from truth data")
                
                # Also store successful matches as positive examples
                matched_elements = [el for el in elements if el.get('id') in [t_el.get('id') for t_el in truth_data.get('elements', [])]]
                if matched_elements:
                    # Store matched elements for future reference
                    for el in matched_elements[:10]:  # Limit to 10
                        pattern = {
                            'type': 'truth_matched_element',
                            'element_type': el.get('type'),
                            'bbox': el.get('bbox'),
                            'confidence': el.get('confidence'),
                            'timestamp': datetime.now().isoformat()
                        }
                        self._store_pattern(pattern)
                        report['patterns_learned'] += 1
                    logger.info(f"Live learning: Stored {len(matched_elements)} truth-matched elements as positive examples")
            
            # LIVE LEARNING: Store analysis metadata for future reference
            analysis_metadata = {
                'timestamp': datetime.now().isoformat(),
                'quality_score': quality_score,
                'element_count': len(elements),
                'connection_count': len(connections),
                'element_types': list(set([el.get('type') for el in elements if el.get('type')])),
                'avg_confidence': sum([el.get('confidence', 0) for el in elements]) / len(elements) if elements else 0
            }
            
            # Store in learning database
            if hasattr(self.knowledge_manager, 'learning_database'):
                recent_analyses = self.knowledge_manager.learning_database.setdefault('recent_analyses', [])
                recent_analyses.append(analysis_metadata)
                
                # Keep only last 100 analyses for performance
                if len(recent_analyses) > 100:
                    recent_analyses = recent_analyses[-100:]
                    self.knowledge_manager.learning_database['recent_analyses'] = recent_analyses
                
                # Save learning database immediately
                if hasattr(self.knowledge_manager, 'save_learning_database'):
                    try:
                        self.knowledge_manager.save_learning_database()
                        logger.debug("Live learning: Saved analysis metadata to learning database")
                    except Exception as e:
                        logger.warning(f"Error saving learning database: {e}")
            
            # Update symbol library with new visual examples
            for element in elements:
                if element.get('bbox') and element.get('type'):
                    # Extract visual symbol from element bbox
                    symbol_image = self._extract_element_image(element, analysis_result.get('image_path'))
                    if symbol_image:
                        symbol_id = self._generate_symbol_id_from_element(element)
                        element_type = element.get('type')
                        metadata = {
                            'source': 'analysis_result',
                            'element_id': element.get('id'),
                            'learned_timestamp': datetime.now().isoformat()
                        }
                        
                        success = self.symbol_library.add_symbol(
                            symbol_id=symbol_id,
                            image=symbol_image,
                            element_type=element_type,
                            metadata=metadata
                        )
                        
                        if success:
                            report['symbols_updated'] += 1
            
            self.learning_stats['last_learning_time'] = datetime.now().isoformat()
            logger.info(f"Analysis learning complete: {report['patterns_learned']} patterns, {report['corrections_learned']} corrections")
            
        except Exception as e:
            logger.error(f"Error in learn_from_analysis_result: {e}", exc_info=True)
            report['errors'].append(str(e))
        
        return report
    
    def adapt_to_pid_type(
        self,
        pid_metadata: Dict[str, Any],
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt analysis approach based on P&ID type detection.
        
        Args:
            pid_metadata: Metadata about the P&ID (type, complexity, etc.)
            analysis_result: Current analysis result
            
        Returns:
            Adaptation report
        """
        logger.info("=== Active Learning: Adapting to P&ID Type ===")
        
        report = {
            'adaptations': [],
            'strategy_adjustments': {}
        }
        
        try:
            pid_type = pid_metadata.get('type', 'generic')
            complexity = pid_metadata.get('complexity', 'medium')
            
            # Adapt strategy based on P&ID type
            if pid_type == 'simple':
                # Use simpler, faster strategies
                report['strategy_adjustments'] = {
                    'use_swarm_only': True,
                    'tile_size': 2048,
                    'skip_polyline_refinement': True
                }
                report['adaptations'].append('Switched to swarm-only mode for simple diagram')
            
            elif pid_type == 'complex':
                # Use full pipeline with extra validation
                report['strategy_adjustments'] = {
                    'use_full_pipeline': True,
                    'enable_self_correction': True,
                    'max_correction_iterations': 3,
                    'tile_size': 1024
                }
                report['adaptations'].append('Enabled full pipeline for complex diagram')
            
            # Learn from successful adaptations
            if analysis_result.get('quality_score', 0) > 0.9:
                self._store_successful_adaptation(pid_type, complexity, report['strategy_adjustments'])
            
            logger.info(f"Adapted strategy for {pid_type} P&ID with complexity {complexity}")
            
        except Exception as e:
            logger.error(f"Error in adapt_to_pid_type: {e}", exc_info=True)
            report['errors'] = [str(e)]
        
        return report
    
    def _extract_symbol_info(self, image: Image.Image, model_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract symbol information from image using LLM."""
        try:
            prompt = """Analyze this P&ID symbol image and extract:
1. Element type (e.g., Pump, Valve, Tank)
2. Key visual features
3. Label if visible

Return as JSON with keys: type, features, label."""
            
            # Convert image to path if needed (LLMClient expects path)
            if hasattr(image, 'filename'):
                image_path = image.filename
            else:
                # Save temporarily
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                image.save(temp_file.name)
                image_path = temp_file.name
            
            response = self.llm_client.call_model(
                model_info,
                system_prompt="You are an expert in P&ID diagram analysis.",
                user_prompt=prompt,
                image_path=image_path
            )
            
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                try:
                    return json.loads(response)
                except:
                    return {'type': 'Unknown', 'features': [], 'label': ''}
            
            return None
        except Exception as e:
            logger.error(f"Error extracting symbol info: {e}", exc_info=True)
            return None
    
    def _generate_symbol_id(self, image_path: Path, symbol_info: Dict[str, Any]) -> str:
        """Generate unique ID for symbol."""
        content = f"{image_path.name}_{symbol_info.get('type', 'Unknown')}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_symbol_id_from_element(self, element: Dict[str, Any]) -> str:
        """Generate unique ID for symbol from element."""
        content = f"{element.get('id', '')}_{element.get('type', 'Unknown')}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _extract_successful_patterns(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract successful patterns from analysis."""
        patterns = []
        
        try:
            # Pattern: Common element type combinations
            element_types = [el.get('type') for el in elements if el.get('type')]
            type_counts = {}
            for el_type in element_types:
                type_counts[el_type] = type_counts.get(el_type, 0) + 1
            
            # Store frequent patterns
            for el_type, count in type_counts.items():
                if count > 2:  # Appears multiple times
                    patterns.append({
                        'type': 'element_type_frequency',
                        'element_type': el_type,
                        'frequency': count,
                        'confidence': min(1.0, count / 10.0)
                    })
            
            # Pattern: Connection patterns
            connection_types = {}
            for conn in connections:
                from_type = next((el.get('type') for el in elements if el.get('id') == conn.get('from_id')), None)
                to_type = next((el.get('type') for el in elements if el.get('id') == conn.get('to_id')), None)
                if from_type and to_type:
                    conn_key = f"{from_type}->{to_type}"
                    connection_types[conn_key] = connection_types.get(conn_key, 0) + 1
            
            for conn_key, count in connection_types.items():
                if count > 1:
                    patterns.append({
                        'type': 'connection_pattern',
                        'pattern': conn_key,
                        'frequency': count
                    })
        
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}", exc_info=True)
        
        return patterns
    
    def _compare_with_truth(
        self,
        analysis_result: Dict[str, Any],
        truth_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare analysis with truth data and generate corrections."""
        corrections = []
        
        try:
            analysis_elements = {el.get('id'): el for el in analysis_result.get('elements', [])}
            truth_elements = {el.get('id'): el for el in truth_data.get('elements', [])}
            
            # Find missing elements
            for truth_id, truth_el in truth_elements.items():
                if truth_id not in analysis_elements:
                    corrections.append({
                        'type': 'missing_element',
                        'element_id': truth_id,
                        'element_type': truth_el.get('type'),
                        'correction': truth_el
                    })
            
            # Find incorrect types
            for el_id, analysis_el in analysis_elements.items():
                if el_id in truth_elements:
                    truth_el = truth_elements[el_id]
                    if analysis_el.get('type') != truth_el.get('type'):
                        corrections.append({
                            'type': 'incorrect_type',
                            'element_id': el_id,
                            'incorrect_type': analysis_el.get('type'),
                            'correct_type': truth_el.get('type'),
                            'correction': truth_el
                        })
        
        except Exception as e:
            logger.error(f"Error comparing with truth: {e}", exc_info=True)
        
        return corrections
    
    def _learn_correction(self, correction: Dict[str, Any]) -> None:
        """Learn from a correction."""
        try:
            # Store correction directly in knowledge manager
            problem = {
                'type': correction.get('type'),
                'description': f"Correction for {correction.get('element_id')}"
            }
            correction_data = {'corrected_data': correction.get('correction')}
            
            # Use knowledge manager's learn_from_correction method if available
            if hasattr(self.knowledge_manager, 'learn_from_correction'):
                correction_hash = self.knowledge_manager.learn_from_correction(
                    problem=problem,
                    correction=correction_data,
                    image_name=None
                )
                logger.debug(f"Learned correction: {correction_hash}")
            else:
                # Fallback: Store directly in learning database
                learned_solutions = self.knowledge_manager.learning_database.setdefault('learned_solutions', {})
                correction_hash = hashlib.sha256(str(problem).encode()).hexdigest()[:16]
                learned_solutions[correction_hash] = {
                    'problem': problem,
                    'correction': correction_data,
                    'timestamp': datetime.now().isoformat()
                }
                self.knowledge_manager.save_learning_database()
                logger.debug(f"Learned correction: {correction_hash}")
        except Exception as e:
            logger.error(f"Error learning correction: {e}", exc_info=True)
    
    def _store_pattern(self, pattern: Dict[str, Any]) -> None:
        """Store a learned pattern."""
        try:
            patterns = self.knowledge_manager.learning_database.setdefault(
                'successful_patterns', {}
            )
            
            pattern_key = f"{pattern.get('type')}_{hashlib.sha256(str(pattern).encode()).hexdigest()[:8]}"
            patterns[pattern_key] = pattern
            
            self.knowledge_manager.save_learning_database()
        except Exception as e:
            logger.error(f"Error storing pattern: {e}", exc_info=True)
    
    def _extract_element_image(
        self,
        element: Dict[str, Any],
        image_path: Optional[str]
    ) -> Optional[Image.Image]:
        """Extract image snippet for element."""
        if not image_path or not element.get('bbox'):
            return None
        
        try:
            from src.utils.image_utils import crop_image_for_correction
            
            img_path = Path(image_path)
            if not img_path.exists():
                return None
            
            bbox = element['bbox']
            cropped_path = crop_image_for_correction(str(img_path), bbox, context_margin=0.0)
            
            if cropped_path:
                return Image.open(cropped_path)
        except Exception as e:
            logger.error(f"Error extracting element image: {e}", exc_info=True)
        
        return None
    
    def _store_successful_adaptation(
        self,
        pid_type: str,
        complexity: str,
        strategy: Dict[str, Any]
    ) -> None:
        """Store successful adaptation strategy."""
        try:
            adaptations = self.knowledge_manager.learning_database.setdefault(
                'successful_adaptations', {}
            )
            
            key = f"{pid_type}_{complexity}"
            if key not in adaptations:
                adaptations[key] = []
            
            adaptations[key].append({
                'strategy': strategy,
                'timestamp': datetime.now().isoformat(),
                'success_count': 1
            })
            
            self.knowledge_manager.save_learning_database()
        except Exception as e:
            logger.error(f"Error storing adaptation: {e}", exc_info=True)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return self.learning_stats.copy()

