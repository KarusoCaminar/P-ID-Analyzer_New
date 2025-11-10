"""
ID Extractor - Robust, multi-layered ID extraction and correction.

This module provides a comprehensive solution for extracting correct element IDs
from P&ID images using multiple strategies:
1. OCR-based extraction (primary): Extract all text labels using Tesseract OCR
2. Bbox-based matching (secondary): Match element bboxes to nearest text labels
3. Pattern validation (tertiary): Validate P&ID tag patterns
4. LLM fallback (quaternary): Use LLM only if OCR fails

This is much more robust and reliable than pure LLM-based correction.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# P&ID Tag Patterns (common formats)
PID_TAG_PATTERNS = [
    r'^[A-Z]{1,3}-\d+$',  # P-201, FT-10, MV-101
    r'^[A-Z]{1,3}-\d+-\d+$',  # Fv-3-3040, MV-3121-101
    r'^[A-Z]{1,3}\d+[A-Z]?$',  # PU3121, MV3121A, CHP2
    r'^[A-Z]{2,4}_\d+$',  # ISA_2, HP_1
    r'^[A-Z]+\s+\d+$',  # HP 1, CHP 2
    r'^[A-Z]{1,3}\d+[A-Z]\d+[A-Z]?$',  # MV3121A, MV3131B
    r'^[A-Z]{2,4}-\d+[A-Z]?$',  # Buffer Storage, VSI Storage (simplified)
    r'^Sample Point$',  # Special case
    r'^ISA$',  # Special case (Instrument Air Supply)
]

# Common P&ID prefixes (for validation)
PID_PREFIXES = [
    'P', 'PU', 'PUMP', 'F', 'FT', 'FV', 'MV', 'V', 'VALVE',
    'CHP', 'HP', 'T', 'TANK', 'R', 'REACTOR', 'M', 'MIXER',
    'S', 'SINK', 'HX', 'HE', 'SEP', 'FILTER', 'COMP', 'TURB',
    'ISA', 'VSI', 'BUFFER', 'STORAGE'
]


class IDExtractor:
    """
    Robust ID extractor that uses OCR + CV + Pattern Matching + LLM fallback.
    
    This is the FINAL, DEFINITIVE solution for ID extraction.
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config_service: Optional[Any] = None
    ):
        """
        Initialize ID Extractor.
        
        Args:
            llm_client: LLM client (optional, for fallback)
            config_service: Configuration service (optional)
        """
        self.llm_client = llm_client
        self.config_service = config_service
        
        # Check if Tesseract is available
        self.tesseract_available = self._check_tesseract()
        
        if not self.tesseract_available:
            logger.warning("Tesseract OCR not available. ID extraction will use LLM fallback only.")
        
        # Load model info for LLM fallback
        self.model_info = None
        if config_service:
            try:
                config = config_service.get_raw_config()
                models_cfg = config.get('models', {})
                model_strategy = config.get('model_strategy', {})
                self.model_info = model_strategy.get('correction_model') or model_strategy.get('meta_model')
                if not self.model_info:
                    self.model_info = models_cfg.get('Google Gemini 2.5 Pro', {})
            except Exception as e:
                logger.debug(f"Could not load model info: {e}")
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            import pytesseract
            # Try to get version (quick test)
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, Exception):
            return False
    
    def extract_ids(
        self,
        image_path: str,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract correct IDs using multi-layered approach.
        
        Strategy:
        1. OCR-based extraction (primary): Extract all text labels
        2. Bbox-based matching: Match elements to nearest text labels
        3. Pattern validation: Validate P&ID tag patterns
        4. LLM fallback: Use LLM only if OCR fails
        
        Args:
            image_path: Path to P&ID image
            elements: List of elements with potentially incorrect IDs
            connections: List of connections with potentially incorrect IDs
            
        Returns:
            Dictionary with corrected 'elements' and 'connections'
        """
        logger.info("=== Starting Multi-Layered ID Extraction ===")
        logger.info(f"Extracting IDs for {len(elements)} elements")
        
        if not elements:
            logger.warning("No elements to extract IDs for")
            return {
                'elements': elements,
                'connections': connections
            }
        
        try:
            # Step 1: OCR-based extraction (primary)
            text_labels = self._extract_text_labels_ocr(image_path)
            
            if text_labels:
                logger.info(f"OCR extracted {len(text_labels)} text labels")
                # Step 2: Match elements to text labels
                corrected_elements = self._match_elements_to_labels(elements, text_labels, image_path)
                # Step 3: Update connections with corrected IDs
                corrected_connections = self._update_connections_with_new_ids(
                    connections,
                    elements,
                    corrected_elements
                )
                
                # Log changes
                self._log_id_changes(elements, corrected_elements)
                
                return {
                    'elements': corrected_elements,
                    'connections': corrected_connections
                }
            else:
                # OCR failed - use LLM fallback
                logger.warning("OCR extraction failed. Using LLM fallback.")
                return self._llm_fallback(image_path, elements, connections)
                
        except Exception as e:
            logger.error(f"Error in ID extraction: {e}", exc_info=True)
            # Fallback to LLM
            logger.warning("ID extraction failed. Using LLM fallback.")
            return self._llm_fallback(image_path, elements, connections)
    
    def _extract_text_labels_ocr(
        self,
        image_path: str
    ) -> List[Dict[str, Any]]:
        """
        Extract all text labels from image using OCR.
        
        Args:
            image_path: Path to image
            
        Returns:
            List of text labels with bboxes and text
        """
        if not self.tesseract_available:
            return []
        
        try:
            import pytesseract
            from pytesseract import Output
            
            # Read image
            # CRITICAL FIX: Handle Unicode paths (Windows encoding issue)
            # cv2.imread doesn't handle Unicode paths on Windows properly
            # Solution: Use numpy fromfile + cv2.imdecode for Unicode support
            try:
                # Try direct path first (works for ASCII paths)
                img = cv2.imread(image_path)
                if img is None:
                    # Fallback: Read file as bytes and decode (handles Unicode paths)
                    # CRITICAL FIX: np is already imported at module level (line 19), don't import again
                    with open(image_path, 'rb') as f:
                        img_data = np.frombuffer(f.read(), np.uint8)
                        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Could not load image: {image_path} (Error: {e})")
                return []
            
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            img_height, img_width = img.shape[:2]
            
            # Convert to grayscale for OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding for better OCR
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text with bounding boxes
            data = pytesseract.image_to_data(thresh, output_type=Output.DICT, lang='eng')
            
            text_labels = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != -1 else 0
                
                # Filter low-confidence and empty text
                if conf < 30 or not text or len(text) < 1:
                    continue
                
                # Filter very long text (likely not P&ID tags)
                if len(text) > 50:
                    continue
                
                # Get bounding box (absolute coordinates)
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Convert to normalized coordinates (0-1)
                bbox = {
                    'x': x / img_width,
                    'y': y / img_height,
                    'width': w / img_width,
                    'height': h / img_height
                }
                
                # Clean text (remove special chars, keep alphanumeric and dashes)
                cleaned_text = self._clean_text(text)
                
                # Validate P&ID tag pattern
                if self._validate_pid_tag(cleaned_text):
                    text_labels.append({
                        'text': cleaned_text,
                        'bbox': bbox,
                        'confidence': conf / 100.0,  # Normalize to 0-1
                        'original_text': text
                    })
            
            logger.info(f"Extracted {len(text_labels)} valid P&ID text labels")
            return text_labels
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}", exc_info=True)
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text for P&ID tag matching."""
        # Remove special chars, keep alphanumeric, dashes, underscores, spaces
        cleaned = ''.join(c for c in text if c.isalnum() or c in ['-', '_', ' ', '.'])
        # Remove multiple spaces
        cleaned = ' '.join(cleaned.split())
        # Remove leading/trailing spaces
        cleaned = cleaned.strip()
        return cleaned
    
    def _validate_pid_tag(self, text: str) -> bool:
        """
        Validate if text matches a P&ID tag pattern.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text matches a P&ID tag pattern
        """
        if not text or len(text) < 1:
            return False
        
        # Check against patterns
        for pattern in PID_TAG_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Check if text starts with a common P&ID prefix
        text_upper = text.upper()
        for prefix in PID_PREFIXES:
            if text_upper.startswith(prefix):
                return True
        
        # Special cases
        if text.upper() in ['SAMPLE POINT', 'ISA', 'S', 'M', 'R']:
            return True
        
        return False
    
    def _match_elements_to_labels(
        self,
        elements: List[Dict[str, Any]],
        text_labels: List[Dict[str, Any]],
        image_path: str
    ) -> List[Dict[str, Any]]:
        """
        Match elements to their nearest text labels.
        
        Args:
            elements: List of elements
            text_labels: List of text labels with bboxes
            image_path: Path to image (for size calculation)
            
        Returns:
            List of elements with corrected IDs
        """
        # Read image to get size
        # CRITICAL FIX: Handle Unicode paths (Windows encoding issue)
        # cv2.imread doesn't handle Unicode paths on Windows properly
        # Solution: Use numpy fromfile + cv2.imdecode for Unicode support
        try:
            # Try direct path first (works for ASCII paths)
            img = cv2.imread(image_path)
            if img is None:
                # Fallback: Read file as bytes and decode (handles Unicode paths)
                # CRITICAL FIX: np is already imported at module level (line 19), don't import again
                with open(image_path, 'rb') as f:
                    img_data = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Could not load image: {image_path} (Error: {e})")
            return elements
        
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return elements
        
        img_height, img_width = img.shape[:2]
        
        # Calculate image diagonal for distance threshold
        image_diagonal = np.sqrt(img_width ** 2 + img_height ** 2)
        max_distance_pixels = image_diagonal * 0.15  # 15% of diagonal (increased from 10%)
        
        corrected_elements = []
        
        for element in elements:
            element_bbox = element.get('bbox', {})
            element_id = element.get('id', '')
            
            # Get element bbox (normalized coordinates 0-1)
            el_x = element_bbox.get('x', 0)
            el_y = element_bbox.get('y', 0)
            el_w = element_bbox.get('width', 0)
            el_h = element_bbox.get('height', 0)
            
            # Calculate element center (normalized)
            el_center_x = el_x + el_w / 2
            el_center_y = el_y + el_h / 2
            
            # Convert to pixel coordinates for distance calculation
            el_center_x_px = el_center_x * img_width
            el_center_y_px = el_center_y * img_height
            el_w_px = el_w * img_width
            el_h_px = el_h * img_height
            
            # Find nearest text label
            best_match = None
            best_distance = float('inf')
            best_label = None
            
            for label in text_labels:
                label_bbox = label.get('bbox', {})
                label_text = label.get('text', '')
                
                # Get label bbox (normalized coordinates 0-1)
                label_x = label_bbox.get('x', 0)
                label_y = label_bbox.get('y', 0)
                label_w = label_bbox.get('width', 0)
                label_h = label_bbox.get('height', 0)
                
                # Calculate label center (normalized)
                label_center_x = label_x + label_w / 2
                label_center_y = label_y + label_h / 2
                
                # Convert to pixel coordinates for distance calculation
                label_center_x_px = label_center_x * img_width
                label_center_y_px = label_center_y * img_height
                
                # Calculate distance (Euclidean distance between centers in pixels)
                distance_px = np.sqrt(
                    (el_center_x_px - label_center_x_px) ** 2 +
                    (el_center_y_px - label_center_y_px) ** 2
                )
                
                # Check if label is within reasonable distance
                if distance_px < max_distance_pixels and distance_px < best_distance:
                    # Check if label is to the right, left, top, or bottom of element
                    # (P&ID labels are typically positioned near elements)
                    dx_px = abs(el_center_x_px - label_center_x_px)
                    dy_px = abs(el_center_y_px - label_center_y_px)
                    
                    # Prefer labels that are horizontally or vertically aligned
                    # (P&ID labels are usually positioned to the side or above/below)
                    # Allow labels within 3x element size in any direction
                    if dx_px < el_w_px * 3 or dy_px < el_h_px * 3:
                        best_distance = distance_px
                        best_match = label
                        best_label = label_text
            
            # Update element ID if match found
            element_copy = element.copy()
            if best_match:
                element_copy['id'] = best_label
                element_copy['label'] = best_label
                element_copy['id_source'] = 'ocr'  # Track source
                logger.debug(f"Matched element {element_id} -> {best_label} (distance: {best_distance:.1f}px)")
            else:
                # Keep original ID but mark as unmatched
                element_copy['id_source'] = 'original'  # Track source
                logger.debug(f"No match found for element {element_id}")
            
            corrected_elements.append(element_copy)
        
        return corrected_elements
    
    def _update_connections_with_new_ids(
        self,
        connections: List[Dict[str, Any]],
        original_elements: List[Dict[str, Any]],
        corrected_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Update connection IDs based on element ID changes.
        
        Args:
            connections: List of connections
            original_elements: Original elements (with old IDs)
            corrected_elements: Corrected elements (with new IDs)
            
        Returns:
            List of connections with updated IDs
        """
        # Build mapping: old_id -> new_id
        id_mapping = {}
        for orig_el, corr_el in zip(original_elements, corrected_elements):
            old_id = orig_el.get('id', '')
            new_id = corr_el.get('id', '')
            if old_id != new_id:
                id_mapping[old_id] = new_id
        
        if not id_mapping:
            # No ID changes - return original connections
            return connections
        
        # Update connections
        corrected_connections = []
        for conn in connections:
            conn_copy = conn.copy()
            from_id = conn.get('from_id', '')
            to_id = conn.get('to_id', '')
            
            # Update from_id if mapped
            if from_id in id_mapping:
                conn_copy['from_id'] = id_mapping[from_id]
            
            # Update to_id if mapped
            if to_id in id_mapping:
                conn_copy['to_id'] = id_mapping[to_id]
            
            corrected_connections.append(conn_copy)
        
        logger.info(f"Updated {len(id_mapping)} connection IDs")
        return corrected_connections
    
    def _llm_fallback(
        self,
        image_path: str,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        LLM fallback for ID extraction (when OCR fails).
        
        Args:
            image_path: Path to image
            elements: List of elements
            connections: List of connections
            
        Returns:
            Dictionary with corrected elements and connections
        """
        if not self.llm_client or not self.model_info:
            logger.warning("LLM fallback not available. Returning original IDs.")
            return {
                'elements': elements,
                'connections': connections
            }
        
        logger.info("Using LLM fallback for ID extraction")
        
        try:
            # Build prompt for LLM
            prompt = self._build_llm_prompt(elements, connections)
            
            # Call LLM
            response = self.llm_client.call_llm(
                model_info=self.model_info,
                system_prompt="You are a P&ID ID extraction specialist. Extract the CORRECT element IDs from the image text.",
                user_prompt=prompt,
                image_path=image_path,
                expected_json_keys=['elements', 'connections']
            )
            
            # Parse response
            if response and isinstance(response, dict):
                corrected_elements = response.get('elements', elements)
                corrected_connections = response.get('connections', connections)
                
                # Mark as LLM-extracted
                for el in corrected_elements:
                    el['id_source'] = 'llm'
                
                return {
                    'elements': corrected_elements,
                    'connections': corrected_connections
                }
            else:
                logger.warning("LLM fallback returned empty result. Using original IDs.")
                return {
                    'elements': elements,
                    'connections': connections
                }
                
        except Exception as e:
            logger.error(f"Error in LLM fallback: {e}", exc_info=True)
            return {
                'elements': elements,
                'connections': connections
            }
    
    def _build_llm_prompt(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for LLM fallback."""
        import json
        
        elements_json = json.dumps(elements, indent=2, ensure_ascii=False)
        connections_json = json.dumps(connections, indent=2, ensure_ascii=False)
        
        prompt = f"""**ROLE:** You are a P&ID ID extraction specialist.

**TASK:** Extract the CORRECT element IDs from the image text and correct the provided elements and connections.

**CURRENT DATA (may have incorrect IDs):**
Elements:
{elements_json}

Connections:
{connections_json}

**CRITICAL INSTRUCTIONS:**
1. Look at the image and find the ACTUAL text labels next to each element symbol
2. Extract the CORRECT P&ID Tag Names (IDs) from the image text (e.g., "P-201", "Fv-3-3040", "MV3121A", "PU3121")
3. For each element in the list, correct the "id" field to match what you see in the image
4. Update all connection "from_id" and "to_id" fields to use the corrected IDs
5. DO NOT change element types, bboxes, or other fields - ONLY correct the IDs
6. If you cannot find an ID in the image, keep the original ID

**OUTPUT FORMAT:**
Return a JSON object with:
- "elements": List of elements with CORRECTED IDs (all other fields unchanged)
- "connections": List of connections with CORRECTED IDs (from_id, to_id updated)

**RETURN ONLY VALID JSON, NO ADDITIONAL TEXT:**"""
        
        return prompt
    
    def _log_id_changes(
        self,
        original_elements: List[Dict[str, Any]],
        corrected_elements: List[Dict[str, Any]]
    ) -> None:
        """Log ID changes for debugging."""
        changes = []
        for orig_el, corr_el in zip(original_elements, corrected_elements):
            orig_id = orig_el.get('id', '')
            corr_id = corr_el.get('id', '')
            id_source = corr_el.get('id_source', 'unknown')
            
            if orig_id != corr_id:
                changes.append(f"{orig_id} -> {corr_id} (source: {id_source})")
        
        if changes:
            logger.info(f"ID corrections: {len(changes)} IDs changed")
            for change in changes[:10]:  # Log first 10 changes
                logger.info(f"  {change}")
            if len(changes) > 10:
                logger.info(f"  ... and {len(changes) - 10} more")
        else:
            logger.info("No ID changes detected")

