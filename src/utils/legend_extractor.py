"""
Legend Extractor with Fallbacks - Robust legend extraction for P&ID diagrams.

CRITICAL: The entire pipeline depends on legend extraction (Phase 1.2).
If legend extraction fails, the pipeline has no basis for analysis.

Fallbacks:
1. CV-based text extraction (Tesseract) + LLM text-to-structure conversion
2. Standard ISO/DIN legend database as emergency fallback
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image

logger = logging.getLogger(__name__)


class LegendExtractor:
    """
    Robust legend extractor with multiple fallback strategies.
    
    CRITICAL: Legend extraction is the foundation of the entire pipeline.
    Without legend data (symbol_map, line_map), the pipeline cannot function correctly.
    """
    
    def __init__(
        self,
        llm_client: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize legend extractor.
        
        Args:
            llm_client: LLM client for text-to-structure conversion
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config
        self.logic_parameters = config.get('logic_parameters', {})
        
        # Load standard legend database
        self._load_standard_legend_database()
    
    def _load_standard_legend_database(self) -> None:
        """
        Load standard ISO/DIN legend database as emergency fallback.
        
        This provides common P&ID symbol types and line styles when
        no legend is found in the diagram.
        """
        standard_legend_path = Path("training_data") / "standard_legend.json"
        
        if standard_legend_path.exists():
            try:
                with open(standard_legend_path, 'r', encoding='utf-8') as f:
                    self.standard_legend = json.load(f)
                logger.info(f"Loaded standard legend database: {len(self.standard_legend.get('symbol_map', {}))} symbols")
            except Exception as e:
                logger.warning(f"Failed to load standard legend database: {e}")
                self.standard_legend = self._create_default_standard_legend()
        else:
            logger.warning("Standard legend database not found. Creating default.")
            self.standard_legend = self._create_default_standard_legend()
    
    def _create_default_standard_legend(self) -> Dict[str, Any]:
        """
        Create default standard legend based on ISO/DIN standards.
        
        Returns:
            Dictionary with standard symbol_map and line_map
        """
        return {
            "symbol_map": {
                # Common P&ID symbols (ISO/DIN standard)
                "Valve": "Valve",
                "Pump": "Pump",
                "Flow Sensor": "Volume Flow Sensor",
                "Pressure Sensor": "Pressure Sensor",
                "Temperature Sensor": "Temperature Sensor",
                "Tank": "Tank",
                "Mixer": "Mixer",
                "Heat Exchanger": "Heat Exchanger",
                "Source": "Source",
                "Sink": "Sink",
                "Sample Point": "Sample Point",
                "Control Valve": "Control Valve",
                "Check Valve": "Check Valve",
                "Ball Valve": "Ball Valve",
                "Gate Valve": "Gate Valve",
                "Globe Valve": "Globe Valve",
                "Centrifugal Pump": "Pump",
                "Positive Displacement Pump": "Pump",
                "Flow Meter": "Volume Flow Sensor",
                "Pressure Gauge": "Pressure Sensor",
                "Temperature Gauge": "Temperature Sensor"
            },
            "line_map": {
                "Process Line": {"color": "black", "style": "solid"},
                "Instrument Line": {"color": "blue", "style": "dashed"},
                "Electrical Line": {"color": "red", "style": "dashed"},
                "Utility Line": {"color": "green", "style": "dotted"}
            },
            "metadata": {
                "source": "ISO/DIN Standard Fallback",
                "confidence": 0.5  # Low confidence for fallback
            }
        }
    
    def extract_legend_with_fallbacks(
        self,
        image_path: str,
        llm_legend_result: Optional[Dict[str, Any]] = None,
        legend_bbox: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Extract legend with multiple fallback strategies.
        
        Strategy:
        1. Use LLM result if available and valid
        2. Fallback 1: CV-based text extraction + LLM text-to-structure
        3. Fallback 2: Standard ISO/DIN legend database
        
        CRITICAL: Now also extracts bounding boxes for individual legend symbols
        to enable proper N-to-M visual matching in legend_matching.py.
        
        Args:
            image_path: Path to P&ID image
            llm_legend_result: LLM legend extraction result (if available)
            legend_bbox: Legend bounding box (if available)
            
        Returns:
            Dictionary with symbol_map, line_map, symbol_bboxes, and metadata
        """
        # Strategy 1: Use LLM result if valid
        if llm_legend_result and isinstance(llm_legend_result, dict):
            symbol_map = llm_legend_result.get("symbol_map", {})
            line_map = llm_legend_result.get("line_map", {})
            # NEW: Extract symbol bounding boxes from LLM result if available
            symbol_bboxes = llm_legend_result.get("symbol_bboxes", {})
            
            if symbol_map or line_map:
                logger.info(f"Using LLM legend result: {len(symbol_map)} symbols, {len(line_map)} line rules")
                
                # If LLM didn't provide bboxes, try to extract them from legend area
                if not symbol_bboxes and legend_bbox:
                    logger.info("LLM result missing symbol bboxes, attempting CV-based extraction...")
                    symbol_bboxes = self._extract_symbol_bboxes_from_legend(
                        image_path, legend_bbox, symbol_map
                    )
                
                return {
                    "symbol_map": symbol_map,
                    "line_map": line_map,
                    "symbol_bboxes": symbol_bboxes,  # NEW: Individual symbol bboxes
                    "legend_bbox": legend_bbox,
                    "metadata": {
                        "source": "LLM Extraction",
                        "confidence": 0.9
                    }
                }
            else:
                logger.warning("LLM legend result has no symbol_map or line_map. Trying fallbacks...")
        
        # Strategy 2: CV-based text extraction + LLM text-to-structure
        if legend_bbox:
            logger.info("Attempting Fallback 1: CV-based text extraction...")
            cv_result = self._extract_legend_with_cv(
                image_path, legend_bbox
            )
            
            if cv_result and (cv_result.get("symbol_map") or cv_result.get("line_map")):
                logger.info(f"Fallback 1 successful: {len(cv_result.get('symbol_map', {}))} symbols")
                
                # Extract symbol bboxes from legend area
                symbol_map = cv_result.get("symbol_map", {})
                if symbol_map:
                    symbol_bboxes = self._extract_symbol_bboxes_from_legend(
                        image_path, legend_bbox, symbol_map
                    )
                    cv_result["symbol_bboxes"] = symbol_bboxes
                
                return cv_result
        
        # Strategy 3: No legend found - return empty (will use Learning DB + Element Type List)
        logger.info("All legend extraction methods failed. No legend found in diagram.")
        logger.info("Pipeline will use Learning Database + Element Type List as fallback.")
        
        return {
            "symbol_map": {},  # Empty - will use base knowledge
            "line_map": {},  # Empty - will use base knowledge
            "symbol_bboxes": {},  # NEW: Empty symbol bboxes
            "legend_bbox": None,
            "metadata": {
                "source": "No Legend Found",
                "confidence": 0.0,
                "info": "No legend found in diagram. Using Learning Database + Element Type List."
            }
        }
    
    def _extract_legend_with_cv(
        self,
        image_path: str,
        legend_bbox: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Fallback 1: Extract legend using CV-based text extraction (Tesseract).
        
        Process:
        1. Crop legend area from image
        2. Extract text using Tesseract OCR
        3. Send raw text to LLM for text-to-structure conversion
        4. Return symbol_map and line_map
        
        Args:
            image_path: Path to P&ID image
            legend_bbox: Legend bounding box (normalized)
            
        Returns:
            Dictionary with symbol_map and line_map, or None if failed
        """
        try:
            # Check if Tesseract is available
            try:
                import pytesseract
                from PIL import Image
                import cv2
                import numpy as np
            except ImportError:
                logger.warning("Tesseract not available. Skipping CV-based legend extraction.")
                return None
            
            # Load image and crop legend area
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Convert normalized bbox to pixel coordinates
            x = int(legend_bbox['x'] * img_width)
            y = int(legend_bbox['y'] * img_height)
            w = int(legend_bbox['width'] * img_width)
            h = int(legend_bbox['height'] * img_height)
            
            # Crop legend area
            legend_crop = img.crop((x, y, x + w, y + h))
            
            # Preprocess image for OCR
            img_cv = cv2.cvtColor(np.array(legend_crop), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding for better OCR
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text using Tesseract
            try:
                raw_text = pytesseract.image_to_string(thresh, lang='eng')
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")
                return None
            
            if not raw_text or len(raw_text.strip()) < 10:
                logger.warning("OCR extracted too little text. Skipping CV fallback.")
                return None
            
            logger.info(f"OCR extracted {len(raw_text)} characters from legend area")
            
            # Convert text to structure using LLM
            structure_result = self._convert_text_to_structure(raw_text)
            
            if structure_result:
                symbol_map = structure_result.get("symbol_map", {})
                
                # Extract symbol bboxes from legend area
                symbol_bboxes = {}
                if symbol_map:
                    symbol_bboxes = self._extract_symbol_bboxes_from_legend(
                        image_path, legend_bbox, symbol_map
                    )
                
                return {
                    "symbol_map": symbol_map,
                    "line_map": structure_result.get("line_map", {}),
                    "symbol_bboxes": symbol_bboxes,  # NEW: Individual symbol bboxes
                    "legend_bbox": legend_bbox,
                    "metadata": {
                        "source": "CV + LLM Text-to-Structure",
                        "confidence": 0.7,
                        "raw_text_length": len(raw_text)
                    }
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"CV-based legend extraction failed: {e}")
            return None
    
    def _convert_text_to_structure(
        self,
        raw_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Convert raw text to structured legend format using LLM.
        
        This is easier than image-to-structure because text is already extracted.
        
        Args:
            raw_text: Raw text extracted from legend area
            
        Returns:
            Dictionary with symbol_map and line_map, or None if failed
        """
        try:
            # Get prompts and model
            prompts = self.config.get('prompts', {})
            if isinstance(prompts, dict):
                prompt_template = prompts.get('legend_text_to_structure_prompt')
            else:
                prompt_template = getattr(prompts, 'legend_text_to_structure_prompt', None)
            
            if not prompt_template:
                # Create default prompt
                prompt_template = """You are a P&ID legend parser. Extract symbol types and line styles from the following legend text.

The text is from a P&ID diagram legend. Extract:
1. Symbol mappings (symbol_key -> symbol_type)
2. Line mappings (line_key -> {color, style})

Return a JSON object with:
- symbol_map: {symbol_key: symbol_type}
- line_map: {line_key: {color: "color_name", style: "solid|dashed|dotted"}}

Legend text:
{raw_text}

Return ONLY valid JSON, no additional text.""".format(raw_text=raw_text[:2000])  # Limit text length
            
            # Use detail model for text-to-structure conversion
            model_strategy = self.config.get('model_strategy', {})
            model_info = model_strategy.get('detail_model') or model_strategy.get('swarm_model')
            
            if not model_info:
                logger.warning("No model available for text-to-structure conversion")
                return None
            
            # Call LLM
            response = self.llm_client.call_llm(
                model_info,
                "You are a P&ID legend parser. Extract structured data from legend text.",
                prompt_template,
                image_path=None,  # No image, only text
                expected_json_keys=["symbol_map", "line_map"]
            )
            
            if response and isinstance(response, dict):
                return {
                    "symbol_map": response.get("symbol_map", {}),
                    "line_map": response.get("line_map", {})
                }
            
            return None
             
        except Exception as e:
            logger.warning(f"Text-to-structure conversion failed: {e}")
            return None
    
    def _extract_symbol_bboxes_from_legend(
        self,
        image_path: str,
        legend_bbox: Dict[str, float],
        symbol_map: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract bounding boxes for individual symbols within the legend area.
        
        This is critical for proper N-to-M visual matching in legend_matching.py.
        Instead of comparing a single diagram element with the entire legend area
        (which is like comparing an apple with a fruit basket), we need to compare
        each diagram element with each individual legend symbol.
        
        Uses CV-based detection to find individual symbol regions within the legend.
        
        Args:
            image_path: Path to P&ID image
            legend_bbox: Legend bounding box (normalized)
            symbol_map: Symbol map {symbol_key: symbol_type}
            
        Returns:
            Dictionary mapping symbol_key to bbox {x, y, width, height} (normalized)
        """
        symbol_bboxes = {}
        
        try:
            # Check if OpenCV is available
            try:
                import cv2
                import numpy as np
            except ImportError:
                logger.warning("OpenCV not available. Cannot extract individual symbol bboxes.")
                return symbol_bboxes
            
            # Load image and crop legend area
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Convert normalized bbox to pixel coordinates
            x = int(legend_bbox['x'] * img_width)
            y = int(legend_bbox['y'] * img_height)
            w = int(legend_bbox['width'] * img_width)
            h = int(legend_bbox['height'] * img_height)
            
            # Crop legend area
            legend_crop = img.crop((x, y, x + w, y + h))
            legend_array = np.array(legend_crop.convert('RGB'))
            legend_gray = cv2.cvtColor(legend_array, cv2.COLOR_RGB2GRAY)
            
            # Detect individual symbol regions using contour detection
            # Symbols in legends are typically arranged in rows/columns
            _, binary = cv2.threshold(legend_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours (individual symbols)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (symbols should be reasonably sized)
            min_symbol_size = min(w, h) // 20  # At least 5% of legend size
            max_symbol_size = min(w, h) // 2    # At most 50% of legend size
            
            detected_regions = []
            for contour in contours:
                x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter by size
                if (min_symbol_size <= w_cont <= max_symbol_size and 
                    min_symbol_size <= h_cont <= max_symbol_size and
                    area > min_symbol_size * min_symbol_size * 0.3):  # At least 30% filled
                    
                    # Normalize bbox relative to legend area, then to full image
                    # First: normalize relative to legend crop (0-1)
                    rel_x = x_cont / w
                    rel_y = y_cont / h
                    rel_w = w_cont / w
                    rel_h = h_cont / h
                    
                    # Then: convert to full image coordinates (normalized)
                    full_x = legend_bbox['x'] + rel_x * legend_bbox['width']
                    full_y = legend_bbox['y'] + rel_y * legend_bbox['height']
                    full_w = rel_w * legend_bbox['width']
                    full_h = rel_h * legend_bbox['height']
                    
                    detected_regions.append({
                        'bbox': {'x': full_x, 'y': full_y, 'width': full_w, 'height': full_h},
                        'area': area,
                        'center_y': y_cont + h_cont // 2  # For sorting
                    })
            
            # Sort regions by vertical position (top to bottom, left to right)
            detected_regions.sort(key=lambda r: (r['center_y'], r['bbox']['x']))
            
            # Match detected regions to symbol_map keys
            # Simple heuristic: assume symbols are listed in order (top to bottom)
            symbol_keys = list(symbol_map.keys())
            
            for i, region in enumerate(detected_regions[:len(symbol_keys)]):
                if i < len(symbol_keys):
                    symbol_key = symbol_keys[i]
                    symbol_bboxes[symbol_key] = region['bbox']
                    logger.debug(f"Extracted bbox for legend symbol '{symbol_key}': {region['bbox']}")
            
            logger.info(f"Extracted {len(symbol_bboxes)} individual symbol bboxes from legend")
            
        except Exception as e:
            logger.warning(f"Error extracting symbol bboxes from legend: {e}")
        
        return symbol_bboxes

