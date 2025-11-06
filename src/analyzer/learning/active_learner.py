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
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
        
        Handles both:
        - Individual symbol images (single symbol per file)
        - Symbol collections (multiple symbols in one image, e.g., PDF collections)
        
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
            'symbols_updated': 0,
            'duplicates_found': 0,
            'collections_processed': 0,
            'individual_symbols_processed': 0,
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
            
            logger.info(f"Processing {len(image_paths)} pretraining files...")
            
            # Process each file
            symbols_to_process = []  # List of (label, image, source_name) tuples
            
            for img_path in image_paths:
                try:
                    image = Image.open(img_path)
                    img_width, img_height = image.size
                    
                    # Determine if this is a collection (large image) or individual symbol
                    # Collections are typically > 2000x2000 pixels
                    is_collection = img_width > 2000 or img_height > 2000
                    
                    if is_collection:
                        logger.info(f"Detected collection image: {img_path.name} ({img_width}x{img_height})")
                        report['collections_processed'] += 1
                        
                        # Extract symbols from collection
                        extracted_symbols = self._extract_symbols_from_collection(
                            image, img_path, model_info
                        )
                        symbols_to_process.extend(extracted_symbols)
                        logger.info(f"Extracted {len(extracted_symbols)} symbols from {img_path.name}")
                    else:
                        logger.info(f"Processing individual symbol: {img_path.name}")
                        report['individual_symbols_processed'] += 1
                        
                        # Extract symbol information
                        symbol_info = self._extract_symbol_info(image, model_info)
                        if symbol_info:
                            label = symbol_info.get('label', img_path.stem)
                            symbols_to_process.append((label, image, str(img_path)))
                    
                    report['symbols_processed'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {img_path}: {e}"
                    logger.error(error_msg, exc_info=True)
                    report['errors'].append(error_msg)
            
            # Now integrate all extracted symbols using the improved method
            if symbols_to_process:
                logger.info(f"Integrating {len(symbols_to_process)} symbols into library...")
                integration_report = self._integrate_symbols_batch(
                    symbols_to_process, model_info
                )
                
                report['symbols_learned'] = integration_report.get('new_symbols', 0)
                report['symbols_updated'] = integration_report.get('updated_symbols', 0)
                report['duplicates_found'] = integration_report.get('duplicates', 0)
            
            self.learning_stats['last_learning_time'] = datetime.now().isoformat()
            logger.info(f"Pretraining complete: {report['symbols_learned']} new, "
                       f"{report['symbols_updated']} updated, {report['duplicates_found']} duplicates")
            
            # AUTOMATIC: Generate viewshots from library after pretraining
            if report['symbols_learned'] > 0 or report['symbols_updated'] > 0:
                logger.info("Generating viewshots from symbol library...")
                viewshot_stats = self.generate_viewshots_from_library()
                logger.info(f"Viewshots generated: {viewshot_stats}")
            
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
        # Check if active learning is enabled
        use_active_learning = self.config.get('logic_parameters', {}).get('use_active_learning', False)
        if not use_active_learning:
            logger.debug("Active learning is disabled. Skipping learning from analysis result.")
            return {
                'patterns_learned': 0,
                'corrections_learned': 0,
                'symbols_updated': 0,
                'errors': []
            }
        
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
            
            # PERFORMANCE: Batch all learning operations and save only once at the end
            patterns_to_store = []
            high_confidence_patterns = []
            truth_matched_patterns = []
            
            if quality_score >= learn_threshold:
                # Extract patterns from current analysis
                patterns = self._extract_successful_patterns(elements, connections)
                patterns_to_store.extend(patterns)
                report['patterns_learned'] += len(patterns)
                self.learning_stats['patterns_learned'] += len(patterns)
            
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
                    high_confidence_patterns.append(pattern)
                    report['patterns_learned'] += 1
            
            # LIVE LEARNING: Learn from corrections IMMEDIATELY (supervised learning with truth data)
            corrections_to_learn = []
            if truth_data:
                corrections = self._compare_with_truth(analysis_result, truth_data)
                corrections_to_learn.extend(corrections)
                report['corrections_learned'] += len(corrections)
                self.learning_stats['corrections_applied'] += len(corrections)
                
                # Also store successful matches as positive examples (ID-based matching)
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
                        truth_matched_patterns.append(pattern)
                        report['patterns_learned'] += 1
            
            # PERFORMANCE: Store all patterns in batch (single database save at the end)
            if patterns_to_store or high_confidence_patterns or truth_matched_patterns:
                for pattern in patterns_to_store:
                    self._store_pattern(pattern, save_immediately=False)  # Don't save immediately
                for pattern in high_confidence_patterns:
                    self._store_pattern(pattern, save_immediately=False)  # Don't save immediately
                for pattern in truth_matched_patterns:
                    self._store_pattern(pattern, save_immediately=False)  # Don't save immediately
                
                # Log summary
                total_patterns = len(patterns_to_store) + len(high_confidence_patterns) + len(truth_matched_patterns)
                logger.info(f"Live learning: Stored {total_patterns} patterns (batch mode)")
            
            # Learn corrections in batch (before saving)
            for correction in corrections_to_learn:
                self._learn_correction(correction, save_immediately=False)  # Don't save immediately
            
            # Update symbol library with new visual examples (but don't save immediately)
            symbol_updates = []
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
                            metadata=metadata,
                            save_immediately=False  # Batch save
                        )
                        
                        if success:
                            symbol_updates.append(symbol_id)
                            report['symbols_updated'] += 1
            
            # PERFORMANCE: Save symbol library once at the end (if updates were made)
            if symbol_updates:
                try:
                    self.symbol_library.save_to_learning_db()
                    logger.debug(f"Saved {len(symbol_updates)} symbol updates to learning database")
                except Exception as e:
                    logger.warning(f"Error saving symbol library: {e}")
            
            # PERFORMANCE: Single database save at the end of all learning operations
            if (patterns_to_store or high_confidence_patterns or truth_matched_patterns or 
                corrections_to_learn or symbol_updates):
                if hasattr(self.knowledge_manager, 'save_learning_database'):
                    try:
                        self.knowledge_manager.save_learning_database()
                        logger.debug(f"Live learning: Saved all learning data (patterns, corrections, symbols) in batch")
                    except Exception as e:
                        logger.warning(f"Error saving learning database: {e}")
            
            # LIVE LEARNING: Store analysis metadata for future reference
            analysis_metadata = {
                'timestamp': datetime.now().isoformat(),
                'quality_score': quality_score,
                'element_count': len(elements),
                'connection_count': len(connections),
                'element_types': list(set([el.get('type') for el in elements if el.get('type')])),
                'avg_confidence': sum([el.get('confidence', 0) for el in elements]) / len(elements) if elements else 0
            }
            
            # Store in learning database (but don't save yet - will be saved once at end of iteration)
            if hasattr(self.knowledge_manager, 'learning_database'):
                recent_analyses = self.knowledge_manager.learning_database.setdefault('recent_analyses', [])
                recent_analyses.append(analysis_metadata)
                
                # Keep only last 100 analyses for performance
                if len(recent_analyses) > 100:
                    recent_analyses = recent_analyses[-100:]
                    self.knowledge_manager.learning_database['recent_analyses'] = recent_analyses
            
            # Update symbol library with new visual examples (but don't save immediately)
            symbol_updates = []
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
                            metadata=metadata,
                            save_immediately=False  # Batch save
                        )
                        
                        if success:
                            symbol_updates.append(symbol_id)
                            report['symbols_updated'] += 1
            
            # PERFORMANCE: Save symbol library once at the end (if updates were made)
            if symbol_updates:
                try:
                    self.symbol_library.save_to_learning_db()
                    logger.debug(f"Saved {len(symbol_updates)} symbol updates to learning database")
                except Exception as e:
                    logger.warning(f"Error saving symbol library: {e}")
            
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
    
    def _extract_symbols_from_collection(
        self,
        collection_image: Image.Image,
        collection_path: Path,
        model_info: Dict[str, Any]
    ) -> List[Tuple[str, Image.Image, str]]:
        """
        Extract individual symbols from a large collection image (e.g., PDF symbol collection).
        
        Uses COMBINED approach:
        1. Computer Vision (OpenCV): Contour detection, edge detection, text detection
        2. LLM refinement: Precise type detection and label extraction
        
        Args:
            collection_image: PIL Image of the collection
            collection_path: Path to the collection image file
            model_info: Model configuration
            
        Returns:
            List of (label, symbol_image, source_name) tuples
        """
        symbols = []
        
        try:
            # Save collection image temporarily
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            collection_image.save(temp_file.name)
            temp_path = temp_file.name
            
            # STEP 1: Computer Vision extraction (fast, precise bboxes)
            from src.utils.symbol_extraction import extract_symbols_with_cv, refine_symbol_bbox_with_cv
            
            cv_symbols = extract_symbols_with_cv(
                temp_path,
                min_symbol_size=30,
                max_symbol_size=800,
                text_padding=15
            )
            
            logger.info(f"CV extraction found {len(cv_symbols)} candidate regions in {collection_path.name}")
            
            # STEP 2: LLM refinement for type detection and label extraction
            # Process each CV-detected region with LLM
            img_width, img_height = collection_image.size
            
            for idx, cv_symbol in enumerate(cv_symbols):
                try:
                    # Get normalized bbox from CV
                    normalized_bbox = cv_symbol.get('normalized_bbox', {})
                    if not normalized_bbox:
                        continue
                    
                    # Convert to pixel coordinates
                    x = int(normalized_bbox.get('x', 0) * img_width)
                    y = int(normalized_bbox.get('y', 0) * img_height)
                    w = int(normalized_bbox.get('width', 0) * img_width)
                    h = int(normalized_bbox.get('height', 0) * img_height)
                    
                    # Validate and clip
                    if w <= 0 or h <= 0 or x < 0 or y < 0:
                        continue
                    if x + w > img_width:
                        w = img_width - x
                    if y + h > img_height:
                        h = img_height - y
                    
                    # Add padding for context
                    padding = 10
                    x_pad = max(0, x - padding)
                    y_pad = max(0, y - padding)
                    w_pad = min(img_width - x_pad, w + 2 * padding)
                    h_pad = min(img_height - y_pad, h + 2 * padding)
                    
                    # Crop symbol with padding
                    symbol_crop = collection_image.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))
                    
                    # Get text regions from CV extraction (for Pretraining)
                    text_regions = cv_symbol.get('text_regions', [])
                    # Convert text regions to crop-relative coordinates
                    crop_text_regions = []
                    if text_regions:
                        for text_region in text_regions:
                            # Convert to crop-relative coordinates
                            crop_text_region = {
                                'x': max(0, text_region.get('x', 0) - x_pad),
                                'y': max(0, text_region.get('y', 0) - y_pad),
                                'width': text_region.get('width', 0),
                                'height': text_region.get('height', 0)
                            }
                            crop_text_regions.append(crop_text_region)
                    
                    # Refine bbox with CV (remove white space) + include text for Pretraining
                    from src.utils.symbol_extraction import refine_symbol_bbox_with_cv
                    refined_bbox = refine_symbol_bbox_with_cv(
                        symbol_crop,
                        text_regions=crop_text_regions,
                        include_text=True  # PRETRAINING: Include text for OCR
                    )
                    
                    # Apply refinement to crop
                    crop_w, crop_h = symbol_crop.size
                    refine_x = int(refined_bbox['x'] * crop_w)
                    refine_y = int(refined_bbox['y'] * crop_h)
                    refine_w = int(refined_bbox['width'] * crop_w)
                    refine_h = int(refined_bbox['height'] * crop_h)
                    
                    if refine_w > 0 and refine_h > 0:
                        symbol_crop = symbol_crop.crop((refine_x, refine_y, refine_x + refine_w, refine_y + refine_h))
                    
                    # STEP 3: OPTIMIZED - OCR for label extraction + LLM for type detection
                    # Save refined crop temporarily
                    temp_crop = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    symbol_crop.save(temp_crop.name)
                    temp_crop_path = temp_crop.name
                    
                    # OPTIMIZATION: Use OCR for label extraction (more accurate than LLM for text)
                    label = f"{collection_path.stem}_sym_{idx}"
                    try:
                        # Try OCR first (Tesseract)
                        try:
                            import pytesseract
                            import cv2
                            import numpy as np
                            
                            # Preprocess for OCR
                            ocr_img = cv2.imread(temp_crop_path)
                            if ocr_img is not None:
                                gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)
                                # Apply thresholding for better OCR
                                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                
                                # Extract text using Tesseract
                                ocr_text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 7')
                                ocr_text = ocr_text.strip()
                                
                                # Filter valid labels (P&ID labels are typically short, alphanumeric)
                                if ocr_text and len(ocr_text) < 50 and any(c.isalnum() for c in ocr_text):
                                    # Clean up OCR text (remove special chars, keep alphanumeric and dashes)
                                    cleaned_label = ''.join(c for c in ocr_text if c.isalnum() or c in ['-', '_', '.'])
                                    if cleaned_label:
                                        label = cleaned_label
                                        logger.debug(f"OCR extracted label: {label}")
                        except ImportError:
                            logger.debug("pytesseract not available, skipping OCR")
                        except Exception as e:
                            logger.debug(f"OCR failed: {e}, falling back to LLM")
                    except Exception as e:
                        logger.debug(f"Error in OCR step: {e}")
                    
                    # LLM for type detection (more accurate than OCR for symbol recognition)
                    llm_prompt = """Analyze this P&ID symbol image and identify the element type.

CRITICAL: Use EXACT type names (Valve, Pump, Volume Flow Sensor, Mixer, Source, Sink, Tank, Storage, Heat Exchanger, Filter, Separator, Compressor, Turbine, Reactor, Sample Point) - CASE-SENSITIVE.

Return as JSON:
{
  "type": "Valve"
}

If you cannot identify the type, return {"type": "Unknown"}."""
                    
                    # Use call_llm (newer method) or call_model (older method) as fallback
                    if hasattr(self.llm_client, 'call_llm'):
                        llm_response = self.llm_client.call_llm(
                            model_info,
                            system_prompt="You are an expert in P&ID symbol identification.",
                            user_prompt=llm_prompt,
                            image_path=temp_crop_path,
                            expected_json_keys=["type"]
                        )
                    else:
                        # Fallback to older call_model method
                        llm_response = self.llm_client.call_model(
                            model_info,
                            system_prompt="You are an expert in P&ID symbol identification.",
                            user_prompt=llm_prompt,
                            image_path=temp_crop_path
                        )
                    
                    # Parse LLM response
                    element_type = 'Unknown'
                    
                    if isinstance(llm_response, dict):
                        element_type = llm_response.get('type', 'Unknown')
                    elif isinstance(llm_response, str):
                        try:
                            llm_data = json.loads(llm_response)
                            element_type = llm_data.get('type', 'Unknown')
                        except json.JSONDecodeError:
                            logger.debug(f"Could not parse LLM response for symbol {idx}")
                    
                    # Cleanup temp crop file
                    try:
                        os.unlink(temp_crop_path)
                    except Exception:
                        pass
                    
                    # Add symbol with refined info
                    symbols.append((label, symbol_crop, str(collection_path)))
                    logger.debug(f"Extracted symbol {idx}: {label} ({element_type}) from {collection_path.name} "
                               f"[CV confidence: {cv_symbol.get('confidence', 0.5):.2f}]")
                    
                except Exception as e:
                    logger.warning(f"Error extracting symbol {idx} from {collection_path.name}: {e}")
                    continue
            
            # If CV extraction found nothing, fallback to LLM-only extraction
            if not symbols:
                logger.info(f"CV extraction found no symbols, falling back to LLM-only extraction for {collection_path.name}")
                symbols = self._extract_symbols_with_llm_only(collection_image, collection_path, model_info, temp_path)
            
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error extracting symbols from collection {collection_path}: {e}", exc_info=True)
        
        return symbols
    
    def _extract_symbols_with_llm_only(
        self,
        collection_image: Image.Image,
        collection_path: Path,
        model_info: Dict[str, Any],
        temp_path: str
    ) -> List[Tuple[str, Image.Image, str]]:
        """Fallback: LLM-only extraction if CV fails."""
        symbols = []
        
        try:
            # Prompt for symbol detection and segmentation
            prompt = """Analyze this P&ID symbol collection image and extract ALL individual symbols.

For each symbol, provide:
1. Bounding box coordinates (x, y, width, height in normalized 0-1 coordinates)
2. Element type (e.g., Pump, Valve, Tank, Volume Flow Sensor)
3. Label if visible (e.g., "P-101", "V-42")

Return as JSON with this structure:
{
  "symbols": [
    {
      "bbox": {"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.05},
      "type": "Valve",
      "label": "V-101"
    },
    ...
  ]
}

CRITICAL: Extract ALL visible symbols. Use EXACT type names (Valve, Pump, Volume Flow Sensor, etc.)."""
            
            system_prompt = """You are an expert in P&ID symbol detection and segmentation.
Extract ALL individual symbols from this collection image with precise bounding boxes."""
            
            # Use call_llm (newer method) or call_model (older method) as fallback
            if hasattr(self.llm_client, 'call_llm'):
                response = self.llm_client.call_llm(
                    model_info,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    image_path=temp_path,
                    expected_json_keys=["symbols"]
                )
            else:
                # Fallback to older call_model method
                response = self.llm_client.call_model(
                    model_info,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    image_path=temp_path
                )
            
            # Parse response
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse symbol extraction response for {collection_path.name}")
                    return symbols
            
            if not isinstance(response, dict) or 'symbols' not in response:
                logger.warning(f"Invalid response format for {collection_path.name}")
                return symbols
            
            # Extract symbols from collection
            img_width, img_height = collection_image.size
            detected_symbols = response.get('symbols', [])
            
            logger.info(f"LLM detected {len(detected_symbols)} symbols in {collection_path.name}")
            
            for idx, symbol_data in enumerate(detected_symbols):
                try:
                    bbox = symbol_data.get('bbox', {})
                    if not bbox:
                        continue
                    
                    # Convert normalized bbox to pixel coordinates
                    x = int(bbox.get('x', 0) * img_width)
                    y = int(bbox.get('y', 0) * img_height)
                    w = int(bbox.get('width', 0) * img_width)
                    h = int(bbox.get('height', 0) * img_height)
                    
                    # Validate bbox
                    if w <= 0 or h <= 0 or x < 0 or y < 0:
                        continue
                    if x + w > img_width or y + h > img_height:
                        # Clip to image bounds
                        w = min(w, img_width - x)
                        h = min(h, img_height - y)
                    
                    # Crop symbol from collection
                    symbol_crop = collection_image.crop((x, y, x + w, y + h))
                    
                    # Get label and type
                    label = symbol_data.get('label', f"{collection_path.stem}_sym_{idx}")
                    element_type = symbol_data.get('type', 'Unknown')
                    
                    symbols.append((label, symbol_crop, str(collection_path)))
                    logger.debug(f"Extracted symbol {idx}: {label} ({element_type}) from {collection_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Error extracting symbol {idx} from {collection_path.name}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in LLM-only extraction: {e}", exc_info=True)
        
        return symbols
    
    def _extract_symbol_info(self, image: Image.Image, model_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract symbol information from image using LLM."""
        try:
            prompt = """Analyze this P&ID symbol image and extract:
1. Element type (e.g., Pump, Valve, Tank, Volume Flow Sensor)
2. Key visual features
3. Label if visible

CRITICAL: Use EXACT type names (Valve, Pump, Volume Flow Sensor, etc.) - CASE-SENSITIVE.

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
            
            # Use call_llm instead of call_model (call_model is deprecated)
            if hasattr(self.llm_client, 'call_llm'):
                response = self.llm_client.call_llm(
                    model_info,
                    system_prompt="You are an expert in P&ID diagram analysis.",
                    user_prompt=prompt,
                    image_path=image_path,
                    expected_json_keys=["type", "features", "label"]
                )
            else:
                # Fallback to older call_model method
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
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.debug(f"Error parsing symbol info JSON: {e}")
                    return {'type': 'Unknown', 'features': [], 'label': ''}
            
            return None
        except Exception as e:
            logger.error(f"Error extracting symbol info: {e}", exc_info=True)
            return None
    
    def _integrate_symbols_batch(
        self,
        symbols_to_process: List[Tuple[str, Image.Image, str]],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate symbols into library with duplicate checking via embedding similarity.
        
        Similar to the old version's integrate_symbol_library method.
        
        Args:
            symbols_to_process: List of (label, image, source_name) tuples
            model_info: Model configuration
            
        Returns:
            Integration report
        """
        report = {
            'new_symbols': 0,
            'updated_symbols': 0,
            'duplicates': 0,
            'errors': []
        }
        
        try:
            # Get similarity threshold from config
            similarity_threshold = self.config.get('logic_parameters', {}).get(
                'visual_symbol_similarity_threshold', 0.85
            )
            
            logger.info(f"Integrating {len(symbols_to_process)} symbols with threshold {similarity_threshold}")
            
            for label, symbol_image, source_name in symbols_to_process:
                try:
                    # Generate embedding for new symbol
                    embedding = self.llm_client.get_image_embedding(symbol_image)
                    if not embedding:
                        logger.warning(f"Could not generate embedding for symbol '{label}'")
                        continue
                    
                    embedding_array = np.array([embedding], dtype=np.float32)
                    
                    # Check for duplicates via similarity search
                    similar_symbols = self.symbol_library.find_similar_symbols(
                        symbol_image,
                        top_k=1,
                        threshold=similarity_threshold
                    )
                    
                    if similar_symbols and len(similar_symbols) > 0:
                        # Duplicate found - update existing symbol
                        symbol_id, similarity_score, existing_metadata = similar_symbols[0]
                        report['duplicates'] += 1
                        report['updated_symbols'] += 1
                        
                        logger.info(f"Symbol '{label}' is duplicate of {symbol_id} "
                                  f"(similarity: {similarity_score:.3f})")
                        
                        # Update metadata but keep same ID
                        updated_metadata = existing_metadata.copy()
                        updated_metadata['source'] = source_name
                        updated_metadata['updated_timestamp'] = datetime.now().isoformat()
                        updated_metadata['label'] = label
                        
                        # Note: We don't update the symbol here, just log it
                        # The duplicate check prevents adding duplicates
                        continue
                    
                    # New symbol - extract info and add
                    symbol_info = self._extract_symbol_info(symbol_image, model_info)
                    if not symbol_info:
                        # Fallback: use 'Unknown' instead of parsing label (label might be "Pid-symbols-PDF_sammlung_sym_0")
                        element_type = 'Unknown'
                        symbol_info = {'type': element_type, 'label': label}
                    else:
                        element_type = symbol_info.get('type', 'Unknown')
                    
                    # Generate unique ID with OCR label if available
                    # CRITICAL: Use OCR label in filename so AI knows what the symbol means
                    # Format: {ocr_label}_{type}_{short_uuid}.png
                    # Example: "P-201_Valve_abc123.png" or "FT-10_Volume_Flow_Sensor_def456.png"
                    # Check if label is generic (starts with collection name + "_sym_")
                    collection_stem = Path(source_name).stem if source_name else ""
                    if label and not (collection_stem and label.startswith(f"{collection_stem}_sym_")):
                        # OCR extracted a real label (e.g., "P-201", "FT-10", "V-101")
                        # Sanitize label for filename (remove special chars, limit length)
                        safe_label = ''.join(c for c in label if c.isalnum() or c in ['-', '_', '.'])[:20]
                        # Format: {label}_{type}_{short_uuid}
                        type_safe = element_type.lower().replace(' ', '_')
                        symbol_id = f"{safe_label}_{type_safe}_{uuid.uuid4().hex[:8]}"
                        logger.debug(f"Using OCR label in symbol ID: {symbol_id}")
                    else:
                        # Fallback: Use type + UUID (no OCR label available)
                        type_safe = element_type.lower().replace(' ', '_')
                        symbol_id = f"{type_safe}_{uuid.uuid4().hex[:12]}"
                        logger.debug(f"Using fallback symbol ID (no OCR label): {symbol_id}")
                    
                    metadata = {
                        'source': 'pretraining',
                        'source_file': source_name,
                        'label': label,
                        'extracted_info': symbol_info,
                        'learned_timestamp': datetime.now().isoformat()
                    }
                    
                    # Add to symbol library
                    success = self.symbol_library.add_symbol(
                        symbol_id=symbol_id,
                        image=symbol_image,
                        element_type=element_type,
                        metadata=metadata,
                        save_immediately=False  # Batch save at the end
                    )
                    
                    if success:
                        report['new_symbols'] += 1
                        logger.info(f"Added new symbol: {symbol_id} ({element_type}) - '{label}'")
                    else:
                        report['errors'].append(f"Failed to add symbol '{label}'")
                    
                except Exception as e:
                    error_msg = f"Error integrating symbol '{label}': {e}"
                    logger.error(error_msg, exc_info=True)
                    report['errors'].append(error_msg)
            
            # Save symbol library once at the end (batch save)
            try:
                self.symbol_library.save_to_learning_db()
                logger.info("Symbol library saved to database")
            except Exception as e:
                logger.warning(f"Error saving symbol library: {e}")
            
        except Exception as e:
            logger.error(f"Error in _integrate_symbols_batch: {e}", exc_info=True)
            report['errors'].append(str(e))
        
        return report
    
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
    
    def _learn_correction(self, correction: Dict[str, Any], save_immediately: bool = True) -> None:
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
                # Only save immediately if requested (for performance: batch saves)
                if save_immediately:
                    self.knowledge_manager.save_learning_database()
                logger.debug(f"Learned correction: {correction_hash}")
        except Exception as e:
            logger.error(f"Error learning correction: {e}", exc_info=True)
    
    def _store_pattern(self, pattern: Dict[str, Any], save_immediately: bool = True) -> None:
        """Store a learned pattern.
        
        Args:
            pattern: Pattern dictionary to store
            save_immediately: If True, save database immediately. If False, defer saving.
        """
        try:
            patterns = self.knowledge_manager.learning_database.setdefault(
                'successful_patterns', {}
            )
            
            pattern_key = f"{pattern.get('type')}_{hashlib.sha256(str(pattern).encode()).hexdigest()[:8]}"
            patterns[pattern_key] = pattern
            
            if save_immediately:
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
    
    def generate_viewshots_from_library(
        self,
        output_dir: Optional[Path] = None,
        max_per_type: int = 5,
        symbol_only: bool = True
    ) -> Dict[str, int]:
        """
        Generate viewshots from symbol library for LLM prompt integration.
        
        This method extracts the best examples from the symbol library and saves them
        as viewshots organized by type. These viewshots are used as visual references
        in LLM prompts to improve type recognition accuracy.
        
        STRATEGIC: For Viewshots, we want NUR Symbol (without text) because:
        - LLM should focus on Symbol-Form for Type-Recognition
        - Text is irrelevant and can distract from visual pattern matching
        
        Args:
            output_dir: Output directory for viewshots (default: training_data/viewshot_examples)
            max_per_type: Maximum number of viewshots per type (default: 5)
            symbol_only: If True, crop to symbol only (remove text) - for Viewshots
                        If False, keep symbol + text - for Pretraining
            
        Returns:
            Dictionary with viewshot generation statistics
        """
        stats = {}
        
        try:
            # Get output directory
            if output_dir is None:
                # Get from config or use default
                viewshot_dir = self.config.get('paths', {}).get('viewshot_examples_dir', 'training_data/viewshot_examples')
                output_dir = Path(viewshot_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all symbols from library
            all_symbols = self.symbol_library.get_all_symbols()
            
            if not all_symbols:
                logger.warning("Symbol library is empty. Cannot generate viewshots.")
                return stats
            
            # Group symbols by type
            symbols_by_type: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
            for symbol_id, symbol_data in all_symbols.items():
                element_type = symbol_data.get('element_type', 'Unknown')
                if element_type == 'Unknown':
                    continue
                
                if element_type not in symbols_by_type:
                    symbols_by_type[element_type] = []
                symbols_by_type[element_type].append((symbol_id, symbol_data))
            
            logger.info(f"Generating viewshots for {len(symbols_by_type)} types from {len(all_symbols)} symbols...")
            
            # Generate viewshots for each type
            for element_type, type_symbols in symbols_by_type.items():
                # Take first N symbols (or all if less than max_per_type)
                viewshot_symbols = type_symbols[:max_per_type]
                
                # Normalize type name for directory
                type_dir_name = element_type.lower().replace(' ', '_')
                type_dir = output_dir / type_dir_name
                type_dir.mkdir(exist_ok=True)
                
                # Copy or link symbol images to viewshot directory
                for idx, (symbol_id, symbol_data) in enumerate(viewshot_symbols):
                    try:
                        # Get image path from symbol data
                        image_path_str = symbol_data.get('image_path')
                        
                        if image_path_str and Path(image_path_str).exists():
                            # Load symbol image
                            source_path = Path(image_path_str)
                            symbol_image = Image.open(source_path)
                            
                            # STRATEGIC: For Viewshots, remove text and keep only symbol
                            if symbol_only:
                                from src.utils.symbol_extraction import refine_symbol_bbox_with_cv
                                
                                # Get symbol-only bbox (without text)
                                symbol_only_bbox = refine_symbol_bbox_with_cv(
                                    symbol_image,
                                    include_text=False  # NUR Symbol für Viewshots
                                )
                                
                                # Crop to symbol only
                                img_w, img_h = symbol_image.size
                                crop_x = int(symbol_only_bbox['x'] * img_w)
                                crop_y = int(symbol_only_bbox['y'] * img_h)
                                crop_w = int(symbol_only_bbox['width'] * img_w)
                                crop_h = int(symbol_only_bbox['height'] * img_h)
                                
                                # Ensure valid crop
                                if crop_w > 0 and crop_h > 0:
                                    symbol_image = symbol_image.crop((
                                        crop_x, crop_y, 
                                        crop_x + crop_w, 
                                        crop_y + crop_h
                                    ))
                                    logger.debug(f"Cropped viewshot to symbol-only: {crop_w}x{crop_h}")
                            
                            # Save viewshot with meaningful name
                            # Try to use OCR label from symbol metadata if available
                            metadata = symbol_data.get('metadata', {})
                            ocr_label = metadata.get('label', '')
                            
                            if ocr_label and not ocr_label.startswith('Pid-symbols-PDF_sammlung_sym_'):
                                # Use OCR label in viewshot filename
                                safe_label = ''.join(c for c in ocr_label if c.isalnum() or c in ['-', '_', '.'])[:15]
                                target_path = type_dir / f"{safe_label}_{type_dir_name}_{idx:04d}.png"
                            else:
                                # Fallback to generic name
                                target_path = type_dir / f"{type_dir_name}_{idx:04d}.png"
                            
                            symbol_image.save(target_path)
                            logger.debug(f"Saved viewshot: {target_path.name}")
                            stats[element_type] = stats.get(element_type, 0) + 1
                        else:
                            # Image not saved - need to load from learning_db or regenerate
                            # For now, skip (images should be saved during add_symbol)
                            logger.warning(f"Image path not found for symbol {symbol_id}, skipping viewshot")
                            continue
                        
                    except Exception as e:
                        logger.warning(f"Error generating viewshot for {symbol_id}: {e}")
                        continue
                
                logger.info(f"Generated {stats.get(element_type, 0)} viewshots for {element_type}")
            
            logger.info(f"Viewshot generation complete: {sum(stats.values())} viewshots generated for {len(stats)} types")
            
        except Exception as e:
            logger.error(f"Error generating viewshots from library: {e}", exc_info=True)
        
        return stats
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return self.learning_stats.copy()

