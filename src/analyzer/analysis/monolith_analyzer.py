"""
Monolith Analyzer - Quadrant-based analysis for structure detection.

Analyzes P&ID diagrams by dividing them into 4 large overlapping quadrants
to capture structural relationships and connections.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from src.interfaces.analyzer import IAnalyzer
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService
from src.utils.image_utils import generate_raster_grid
from src.utils.graph_utils import GraphSynthesizer, SynthesizerConfig

logger = logging.getLogger(__name__)
llm_logger = logging.getLogger('llm_calls')  # Dedicated logger for LLM calls


class MonolithAnalyzer(IAnalyzer):
    """
    Monolith analyzer for quadrant-based structure detection.
    
    Analyzes the diagram using 4 large overlapping quadrants
    to capture structural relationships and connections.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_manager: KnowledgeManager,
        config_service: ConfigService,
        model_strategy: Dict[str, Any],
        logic_parameters: Dict[str, Any],
        symbol_library: Optional[Any] = None
    ):
        """
        Initialize monolith analyzer.
        
        Args:
            llm_client: LLM client for analysis
            knowledge_manager: Knowledge manager for type resolution
            config_service: Configuration service
            model_strategy: Model strategy configuration
            logic_parameters: Logic parameters for analysis
            symbol_library: Optional symbol library for pre-filtering
        """
        self.llm_client = llm_client
        self.knowledge_manager = knowledge_manager
        self.config_service = config_service
        self.model_strategy = model_strategy
        self.logic_parameters = logic_parameters
        self.symbol_library = symbol_library
        
        config = config_service.get_config()
        self.prompts = config.prompts
        
        # Instance attributes for context
        self.legend_context = None
        self.error_feedback = None
        self.element_list_json = ""  # JSON string of elements from Swarm for connection detection
    
    def analyze(
        self,
        image_path: str,
        output_dir: Optional[Path] = None,
        excluded_zones: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze image using monolith (quadrant) approach.
        
        Args:
            image_path: Path to image
            output_dir: Optional output directory
            excluded_zones: Zones to exclude from analysis
            
        Returns:
            Dictionary with 'elements' and 'connections' keys
        """
        logger.info("Starting monolith analysis...")
        
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            logger.error(f"Image not found or could not be opened: {e}")
            return {"elements": [], "connections": []}
        
        output_dir = output_dir or Path(os.path.dirname(image_path))
        temp_dir_path = output_dir / "temp_quadrants"
        temp_dir_path.mkdir(exist_ok=True, parents=True)
        
        excluded_zones = excluded_zones or []
        
        # ADAPTIVE QUADRANT STRATEGY: Calculate optimal number of quadrants based on image size
        # This replaces the fixed 3000px threshold with an adaptive approach
        # OPTIMIZATION: Use whole-image for small images to get full context (better connection detection)
        max_dimension = max(img_width, img_height)
        num_quadrants = self._calculate_optimal_quadrant_strategy(img_width, img_height)
        
        if num_quadrants == 0:
            # Very small image - use whole image analysis
            # OPTIMIZATION: Whole-image provides full context for better connection detection
            logger.info(f"Image is very small ({max_dimension}px), using whole-image analysis instead of quadrants "
                       f"(full context for optimal connection detection)")
            legend_context = getattr(self, 'legend_context', None)
            return self._analyze_whole_image(image_path, legend_context)
        
        logger.info(f"Adaptive quadrant strategy: Using {num_quadrants} quadrants for {img_width}x{img_height} image ({max_dimension}px)")
        
        # Adaptive tile size based on image dimension (not grid-based)
        # Strategy 1: 3000-6000px → 4 Quadranten, 60% Tile-Size, 25% Overlap
        # Strategy 2: 6000-10000px → 6 Quadranten, 50% Tile-Size, 30% Overlap
        # Strategy 3: >10000px → 8-9 Quadranten, 40% Tile-Size, 35% Overlap
        if max_dimension < 6000:
            # Strategy 1: 4 Quadranten (2x2 grid)
            grid_cols, grid_rows = 2, 2
            tile_size_percentage = 0.60  # 60% of max dimension
            overlap_percentage = 0.25  # 25% Overlap
        elif max_dimension < 10000:
            # Strategy 2: 6 Quadranten (2x3 grid)
            grid_cols, grid_rows = 2, 3
            tile_size_percentage = 0.50  # 50% of max dimension
            overlap_percentage = 0.30  # 30% Overlap
        else:
            # Strategy 3: 8-9 Quadranten (3x3 grid for very large)
            grid_cols, grid_rows = 3, 3
            tile_size_percentage = 0.40  # 40% of max dimension
            overlap_percentage = 0.35  # 35% Overlap
        
        # Calculate tile size as percentage of max dimension (adaptive, not grid-based)
        tile_size = int(max_dimension * tile_size_percentage)
        
        # Calculate overlap
        overlap = int(tile_size * overlap_percentage)
        
        logger.info(f"Adaptive strategy: {num_quadrants} quadrants ({grid_cols}x{grid_rows}), "
                   f"tile_size={tile_size}px ({tile_size_percentage*100:.0f}% of {max_dimension}px), "
                   f"overlap={overlap}px ({overlap_percentage*100:.0f}%)")
        
        # Generate quadrant tiles using adaptive grid
        quadrant_tiles = generate_raster_grid(
            image_path,
            tile_size,
            overlap,
            excluded_zones,
            temp_dir_path
        )
        
        if not quadrant_tiles:
            logger.warning("Could not generate quadrant tiles for monolith analysis.")
            if temp_dir_path.exists():
                import shutil
                shutil.rmtree(temp_dir_path)
            return {"elements": [], "connections": []}
        
        logger.info(f"Generated {len(quadrant_tiles)} quadrant tiles.")
        
        # Process quadrants in parallel
        # Get legend_context from instance attribute if set
        legend_context = getattr(self, 'legend_context', None)
        raw_results = self._process_quadrants_parallel(quadrant_tiles, legend_context)
        
        # Synthesize results
        synthesizer_config = SynthesizerConfig(
            iou_match_threshold=self.logic_parameters.get('iou_match_threshold', 0.5)
        )
        synthesizer = GraphSynthesizer(raw_results, img_width, img_height, config=synthesizer_config)
        monolith_graph = synthesizer.synthesize()
        
        # Clean up
        if temp_dir_path.exists():
            import shutil
            shutil.rmtree(temp_dir_path)
        
        logger.info(f"Monolith analysis complete: {len(monolith_graph['elements'])} elements, {len(monolith_graph['connections'])} connections")
        
        return monolith_graph
    
    def _process_quadrants_parallel(
        self,
        tiles: List[Dict[str, Any]],
        legend_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process quadrant tiles in parallel."""
        raw_results = []
        
        # Get prompts and model - Handle PromptsConfig (Pydantic model) or dict
        if isinstance(self.prompts, dict):
            monolith_prompt_template = self.prompts.get('monolithic_analysis_prompt_template')
            system_prompt = self.prompts.get('general_system_prompt')
        else:
            # PromptsConfig Pydantic model - use attribute access
            monolith_prompt_template = getattr(self.prompts, 'monolithic_analysis_prompt_template', None)
            system_prompt = getattr(self.prompts, 'general_system_prompt', None) or 'You are an expert in analyzing technical diagrams.'
        # IMPROVED: Prefer monolith_model, fallback to detail_model
        monolith_model = self.model_strategy.get('monolith_model') or self.model_strategy.get('detail_model')
        detail_model_info = monolith_model
        
        if not all([monolith_prompt_template, system_prompt, detail_model_info]):
            logger.error("Configuration for monolith analysis incomplete.")
            return []
        
        # Build component list
        known_types_list = self.knowledge_manager.get_known_types()
        component_list_str = ", ".join(f"'{name}'" for name in known_types_list)
        
        # Build legend context string using helper method (consistent with whole-image mode)
        legend_context_str = self._build_legend_context_str(legend_context)
        
        # Load viewshot examples
        viewshot_context = self._load_viewshot_examples()
        
        # Get element_list_json from instance attribute (set by pipeline_coordinator)
        element_list_json = getattr(self, 'element_list_json', '[]')
        
        # CRITICAL: For simple P&IDs, if element_list_json is empty, Monolith should recognize elements AND connections
        # Use different prompt template if element_list_json is empty (simple P&ID mode)
        if element_list_json == "[]" or element_list_json.strip() == "[]":
            # Simple P&ID mode: Monolith recognizes elements AND connections independently
            logger.info("CRITICAL: element_list_json is empty - Monolith will recognize elements AND connections independently")
            # Use a modified prompt that allows element detection
            monolith_prompt_template_simple = monolith_prompt_template.replace(
                "**TASK:** Your ONLY task is to find ALL connections (lines/pipes) between the elements provided in the knowledge base.",
                "**TASK:** Find ALL elements (components) AND ALL connections (lines/pipes) in the P&ID diagram."
            ).replace(
                "**CRITICAL KNOWLEDGE BASE (INPUT):**\nHere is a complete JSON list of all known elements on the diagram. You MUST use their exact IDs.\n`{element_list_json}`",
                "**CRITICAL:** You must recognize elements AND connections independently. No element list is provided - you must detect all elements yourself."
            ).replace(
                "**CRITICAL RULES (STRICTLY ENFORCE):**\n1. **ONLY DETECT CONNECTIONS.** Do NOT detect elements.",
                "**CRITICAL RULES (STRICTLY ENFORCE):**\n1. **DETECT BOTH ELEMENTS AND CONNECTIONS.** You must find all components and all connections between them."
            ).replace(
                "2. Use the EXACT element IDs from the knowledge base for `\"from_id\"` and `\"to_id\"`.",
                "2. Use the EXACT element IDs you detect for `\"from_id\"` and `\"to_id\"` in connections."
            ).replace(
                "5. IGNORE connections to elements NOT in the provided list.",
                "5. Find ALL connections between ALL elements you detect."
            ).replace(
                "**1. \"elements\" List:**\n- CRITICAL: Provide an EMPTY list. You MUST NOT detect elements.\n- ` \"elements\": [] `",
                "**1. \"elements\" List:**\n- Find ALL components (pumps, valves, sensors, etc.) in the diagram.\n- REQUIRED KEYS: `\"id\"`, `\"type\"`, `\"label\"`, `\"bbox\"`."
            )
            monolith_prompt = monolith_prompt_template_simple.replace(
                "{ignore_zones_str}", "[]"
            ).replace(
                "[{component_list_str}]", f"[{component_list_str}]"
            ).replace(
                "{element_list_json}", ""  # Empty - Monolith recognizes elements itself
            ).replace(
                "{legend_context}", legend_context_str
            ).replace(
                "{viewshot_valve_examples}", viewshot_context.get('valve', '')
            ).replace(
                "{viewshot_flow_sensor_examples}", viewshot_context.get('flow_sensor', '')
            ).replace(
                "{viewshot_mixer_examples}", viewshot_context.get('mixer', '')
            ).replace(
                "{viewshot_source_examples}", viewshot_context.get('source', '')
            ).replace(
                "{viewshot_sample_point_examples}", viewshot_context.get('sample_point', '')
            ).replace(
                "{viewshot_pump_examples}", viewshot_context.get('pump', '')
            ).replace(
                "{viewshot_sink_examples}", viewshot_context.get('sink', '')
            ).replace(
                "{viewshot_section}", viewshot_context.get('section', '')
            )
        else:
            # Complex P&ID mode: Monolith recognizes connections (with element list as input)
            # CRITICAL: Also allow Monolith to detect ADDITIONAL elements from legend that Swarm missed
            # Check if legend has symbol_map (indicates legend is present)
            has_legend_symbols = False
            if legend_context and legend_context.get('symbol_map'):
                has_legend_symbols = True
                symbol_map = legend_context.get('symbol_map', {})
                logger.info(f"Legend detected with {len(symbol_map)} symbols. Monolith can detect additional elements from legend.")
            
            # If legend is present, modify prompt to allow additional element detection
            if has_legend_symbols:
                # Enhanced prompt: Monolith can detect connections AND additional elements from legend
                monolith_prompt_enhanced = monolith_prompt_template.replace(
                    "**TASK:** Your ONLY task is to find ALL connections (lines/pipes) between the elements provided in the knowledge base.",
                    "**TASK:** Find ALL connections (lines/pipes) between the elements provided in the knowledge base. Additionally, if you find symbols from the legend that are NOT in the element list, also detect them."
                ).replace(
                    "**CRITICAL RULES (STRICTLY ENFORCE):**\n1. **ONLY DETECT CONNECTIONS.** Do NOT detect elements.",
                    "**CRITICAL RULES (STRICTLY ENFORCE):**\n1. **PRIMARY TASK: DETECT CONNECTIONS** between elements in the knowledge base.\n2. **SECONDARY TASK: DETECT ADDITIONAL ELEMENTS** from the legend that are NOT in the element list."
                ).replace(
                    "**1. \"elements\" List:**\n- CRITICAL: Provide an EMPTY list. You MUST NOT detect elements.\n- ` \"elements\": [] `",
                    "**1. \"elements\" List:**\n- PRIMARY: Provide an EMPTY list (connections are your main task).\n- SECONDARY: If you find symbols from the legend that are NOT in the element list above, add them here.\n- Only add elements that match symbols in the legend symbol map.\n- Use EXACT type names from the legend (case-sensitive, exact spacing)."
                )
                
                monolith_prompt = monolith_prompt_enhanced.replace(
                    "{ignore_zones_str}", "[]"
                ).replace(
                    "[{component_list_str}]", f"[{component_list_str}]"
                ).replace(
                    "{element_list_json}", element_list_json  # Use provided element list
                ).replace(
                    "{legend_context}", legend_context_str
                ).replace(
                    "{viewshot_valve_examples}", viewshot_context.get('valve', '')
                ).replace(
                    "{viewshot_flow_sensor_examples}", viewshot_context.get('flow_sensor', '')
                ).replace(
                    "{viewshot_mixer_examples}", viewshot_context.get('mixer', '')
                ).replace(
                    "{viewshot_source_examples}", viewshot_context.get('source', '')
                ).replace(
                    "{viewshot_sample_point_examples}", viewshot_context.get('sample_point', '')
                ).replace(
                    "{viewshot_pump_examples}", viewshot_context.get('pump', '')
                ).replace(
                    "{viewshot_sink_examples}", viewshot_context.get('sink', '')
                ).replace(
                    "{viewshot_section}", viewshot_context.get('section', '')
                )
            else:
                # No legend: Standard prompt (only connections, no additional elements)
                monolith_prompt = monolith_prompt_template.replace(
                    "{ignore_zones_str}", "[]"
                ).replace(
                    "[{component_list_str}]", f"[{component_list_str}]"
                ).replace(
                    "{element_list_json}", element_list_json  # Use provided element list
                ).replace(
                    "{legend_context}", legend_context_str
                ).replace(
                    "{viewshot_valve_examples}", viewshot_context.get('valve', '')
                ).replace(
                    "{viewshot_flow_sensor_examples}", viewshot_context.get('flow_sensor', '')
                ).replace(
                    "{viewshot_mixer_examples}", viewshot_context.get('mixer', '')
                ).replace(
                    "{viewshot_source_examples}", viewshot_context.get('source', '')
                ).replace(
                    "{viewshot_sample_point_examples}", viewshot_context.get('sample_point', '')
                ).replace(
                    "{viewshot_pump_examples}", viewshot_context.get('pump', '')
                ).replace(
                    "{viewshot_sink_examples}", viewshot_context.get('sink', '')
                ).replace(
                    "{viewshot_section}", viewshot_context.get('section', '')
                )
        
        # Add error feedback if available
        error_feedback_str = getattr(self, 'error_feedback', None) or ""
        if error_feedback_str:
            monolith_prompt += f"\n\n**ERROR FEEDBACK (FIX THESE ISSUES):**\n{error_feedback_str}\n"
        
        # Process quadrants with controlled concurrency
        max_workers = self.logic_parameters.get('llm_executor_workers', 4)
        timeout = self.logic_parameters.get('llm_default_timeout', 120)
        
        # Limit max_workers to prevent API overload (max 8 concurrent requests)
        max_workers = min(max_workers, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {}
            
            for tile in tiles:
                # Build tile-specific prompt with symbol library hints and CV text detection
                tile_prompt = monolith_prompt
                
                # Check symbol library before LLM call (if available)
                symbol_hints = ""
                if self.symbol_library:
                    try:
                        # Load tile image for symbol library matching
                        from PIL import Image
                        with Image.open(tile['path']) as tile_img:
                            # Check for similar symbols in library
                            similar_symbols = self.symbol_library.find_similar_symbols(
                                tile_img,
                                threshold=0.75,
                                max_results=3
                            )
                            
                            if similar_symbols:
                                symbol_hints = "\n\n**KNOWN SYMBOLS DETECTED (from symbol library):**\n"
                                symbol_hints += "**FEW-SHOT EXAMPLES (use these exact types):**\n"
                                for symbol_id, similarity, metadata in similar_symbols:
                                    symbol_type = metadata.get('element_type', 'Unknown')
                                    symbol_hints += f"\n**FEW-SHOT EXAMPLE {symbol_id}:**\n"
                                    symbol_hints += f"- Visual: Symbol similar to known '{symbol_type}' (similarity: {similarity:.2f})\n"
                                    symbol_hints += f"- Type: '{symbol_type}' (EXACT spelling, case-sensitive)\n"
                                    symbol_hints += f"- If you see a symbol similar to this, use EXACT type '{symbol_type}'.\n"
                                symbol_hints += "\n**IMPORTANT:** If you see symbols matching the above, use the exact type from the symbol library.\n"
                    except Exception as e:
                        logger.debug(f"Error checking symbol library for quadrant {tile['path']}: {e}")
                
                # CV-BASED TEXT DETECTION: Detect text regions for label hints
                text_hints = ""
                use_cv_text_detection = self.logic_parameters.get('use_cv_text_detection', True)
                if use_cv_text_detection:
                    try:
                        from src.utils.symbol_extraction import _detect_text_regions
                        import cv2
                        import numpy as np
                        
                        # Load tile image
                        tile_img_cv = cv2.imread(tile['path'])
                        if tile_img_cv is not None:
                            gray = cv2.cvtColor(tile_img_cv, cv2.COLOR_BGR2GRAY)
                            text_regions = _detect_text_regions(gray, padding=10)
                            
                            if text_regions:
                                text_hints = "\n\n**TEXT REGIONS DETECTED (CV-based):**\n"
                                text_hints += f"- Found {len(text_regions)} text regions in this quadrant.\n"
                                text_hints += "**IMPORTANT:** These regions likely contain labels (e.g., 'P-101', 'V-42'). "
                                text_hints += "Extract labels from these regions and associate them with nearby symbols.\n"
                    except Exception as e:
                        logger.debug(f"Error in CV text detection for quadrant {tile['path']}: {e}")
                
                # Combine hints
                if symbol_hints or text_hints:
                    tile_prompt = monolith_prompt + symbol_hints + text_hints
                
                future = executor.submit(
                    self.llm_client.call_llm,
                    detail_model_info,
                    system_prompt,
                    tile_prompt,
                    tile['path'],
                    expected_json_keys=["elements", "connections"]
                )
                future_to_tile[future] = tile
            
            # Collect results
            for future in as_completed(future_to_tile):
                tile = future_to_tile[future]
                try:
                    response_data = future.result(timeout=timeout)
                    if response_data:
                        raw_results.append({
                            'tile_coords': tile['coords'],
                            'tile_width': tile['tile_width'],
                            'tile_height': tile['tile_height'],
                            'data': response_data
                        })
                except Exception as exc:
                    logger.error(f"Error analyzing quadrant '{tile['path']}': {exc}", exc_info=True)
        
        return raw_results
    
    def _calculate_optimal_quadrant_strategy(self, img_width: int, img_height: int) -> int:
        """
        Calculate optimal number of quadrants based on image size.
        
        Adaptive strategy:
        - Very small images (<2000px): 0 (whole image)
        - Small images (2000-4000px): 4 quadrants (2x2)
        - Medium images (4000-8000px): 6 quadrants (2x3)
        - Large images (>8000px): 9 quadrants (3x3)
        
        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Number of quadrants to use (0 = whole image, 4/6/9 = quadrant count)
        """
        max_dimension = max(img_width, img_height)
        total_pixels = img_width * img_height
        
        # Very small images: use whole image
        if max_dimension < 2000 or total_pixels < 4_000_000:  # <4MP
            return 0
        
        # Small images: 4 quadrants (2x2)
        if max_dimension < 4000 or total_pixels < 16_000_000:  # <16MP
            return 4
        
        # Medium images: 6 quadrants (2x3)
        if max_dimension < 8000 or total_pixels < 64_000_000:  # <64MP
            return 6
        
        # Large images: 9 quadrants (3x3)
        return 9
    
    def _analyze_whole_image(
        self,
        image_path: str,
        legend_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze whole image as single context (for very small images).
        
        Args:
            image_path: Path to image
            legend_context: Optional legend context
            
        Returns:
            Dictionary with 'elements' and 'connections' keys
        """
        logger.info("Analyzing whole image as single context...")
        
        try:
            # Get prompts and model
            if isinstance(self.prompts, dict):
                monolith_prompt_template = self.prompts.get('monolithic_analysis_prompt_template')
                system_prompt = self.prompts.get('general_system_prompt')
            else:
                monolith_prompt_template = getattr(self.prompts, 'monolithic_analysis_prompt_template', None)
                system_prompt = getattr(self.prompts, 'general_system_prompt', None) or 'You are an expert in analyzing technical diagrams.'
            
            # IMPROVED: Prefer monolith_model, fallback to detail_model
            monolith_model = self.model_strategy.get('monolith_model') or self.model_strategy.get('detail_model')
            detail_model_info = monolith_model
            
            if not all([monolith_prompt_template, system_prompt, detail_model_info]):
                logger.error("Configuration for monolith analysis incomplete.")
                return {"elements": [], "connections": []}
            
            # Build component list
            known_types_list = self.knowledge_manager.get_known_types()
            component_list_str = ", ".join(f"'{name}'" for name in known_types_list)
            
            # Load viewshot examples
            viewshot_context = self._load_viewshot_examples()
            
            # Build legend context string
            legend_context_str = self._build_legend_context_str(legend_context)
            
            # Get element_list_json from instance attribute (set by pipeline_coordinator)
            element_list_json = getattr(self, 'element_list_json', '[]')
            
            # CRITICAL: For simple P&IDs, if element_list_json is empty, Monolith should recognize elements AND connections
            # Use different prompt template if element_list_json is empty (simple P&ID mode)
            if element_list_json == "[]" or element_list_json.strip() == "[]":
                # Simple P&ID mode: Monolith recognizes elements AND connections independently
                logger.info("CRITICAL: element_list_json is empty - Monolith will recognize elements AND connections independently")
                # Use a modified prompt that allows element detection
                monolith_prompt_template_simple = monolith_prompt_template.replace(
                    "**TASK:** Your ONLY task is to find ALL connections (lines/pipes) between the elements provided in the knowledge base.",
                    (
                        "**TASK:** Find ALL elements (components) AND ALL connections (lines/pipes) in the P&ID diagram.\n"
                        "**CRITICAL ID RULE:** The `id` you assign to an element MUST be the exact text label found on the diagram (e.g., 'P-201', 'FT-10'). This is essential for matching."
                    )
                ).replace(
                    "**CRITICAL KNOWLEDGE BASE (INPUT):**\nHere is a complete JSON list of all known elements on the diagram. You MUST use their exact IDs.\n`{element_list_json}`",
                    "**CRITICAL:** You must recognize elements AND connections independently. No element list is provided - you must detect all elements yourself."
                ).replace(
                    "**CRITICAL RULES (STRICTLY ENFORCE):**\n1. **ONLY DETECT CONNECTIONS.** Do NOT detect elements.",
                    "**CRITICAL RULES (STRICTLY ENFORCE):**\n1. **DETECT BOTH ELEMENTS AND CONNECTIONS.** You must find all components and all connections between them."
                ).replace(
                    "2. Use the EXACT element IDs from the knowledge base for `\"from_id\"` and `\"to_id\"`.",
                    "2. Use the EXACT element IDs you detect for `\"from_id\"` and `\"to_id\"` in connections."
                ).replace(
                    "5. IGNORE connections to elements NOT in the provided list.",
                    "5. Find ALL connections between ALL elements you detect."
                ).replace(
                    "**1. \"elements\" List:**\n- CRITICAL: Provide an EMPTY list. You MUST NOT detect elements.\n- ` \"elements\": [] `",
                    (
                        "**1. \"elements\" List:**\n"
                        "- Find ALL components (pumps, valves, sensors, etc.) in the diagram.\n"
                        "- REQUIRED KEYS: `\"id\"`, `\"type\"`, `\"label\"`, `\"bbox\"`, `\"confidence\"`.\n"
                        "- **`\"id\"`**: MUST be the text label visible on the diagram (e.g., 'P-201').\n"
                        "- **`\"label\"`**: The descriptive label (e.g., 'From Transfer Pump P-201').\n"
                        "- **`\"confidence\"`**: (float, 0.0-1.0) Your confidence (0.9+ = Very Sure, < 0.5 = Guessing)."
                    )
                ).replace(
                    "**2. \"connections\" List:**\n- Find ALL connections (lines/pipes) between ALL elements you detect.",
                    (
                        "**2. \"connections\" List:**\n"
                        "- Find ALL connections (lines/pipes) between ALL elements you detect.\n"
                        "- REQUIRED KEYS: `\"from_id\"`, `\"to_id\"`, `\"confidence\"`.\n"
                        "- **`\"confidence\"`**: (float, 0.0-1.0) Your confidence (0.9+ = Very Sure, < 0.5 = Guessing)."
                    )
                )
                monolith_prompt = monolith_prompt_template_simple.replace(
                    "{ignore_zones_str}", "[]"
                ).replace(
                    "[{component_list_str}]", f"[{component_list_str}]"
                ).replace(
                    "{element_list_json}", ""  # Empty - Monolith recognizes elements itself
                ).replace(
                    "{legend_context}", legend_context_str
                ).replace(
                    "{viewshot_valve_examples}", viewshot_context.get('valve', '')
                ).replace(
                    "{viewshot_flow_sensor_examples}", viewshot_context.get('flow_sensor', '')
                ).replace(
                    "{viewshot_mixer_examples}", viewshot_context.get('mixer', '')
                ).replace(
                    "{viewshot_source_examples}", viewshot_context.get('source', '')
                ).replace(
                    "{viewshot_sample_point_examples}", viewshot_context.get('sample_point', '')
                ).replace(
                    "{viewshot_pump_examples}", viewshot_context.get('pump', '')
                ).replace(
                    "{viewshot_sink_examples}", viewshot_context.get('sink', '')
                ).replace(
                    "{viewshot_section}", viewshot_context.get('section', '')
                )
            else:
                # Complex P&ID mode: Monolith recognizes connections (with element list as input)
                # CRITICAL: Also allow Monolith to detect ADDITIONAL elements from legend that Swarm missed
                # Check if legend has symbol_map (indicates legend is present)
                has_legend_symbols = False
                if legend_context and legend_context.get('symbol_map'):
                    has_legend_symbols = True
                    symbol_map = legend_context.get('symbol_map', {})
                    logger.info(f"Legend detected with {len(symbol_map)} symbols. Monolith can detect additional elements from legend.")
                
                # If legend is present, modify prompt to allow additional element detection
                if has_legend_symbols:
                    # Enhanced prompt: Monolith can detect connections AND additional elements from legend
                    monolith_prompt_template_enhanced = monolith_prompt_template.replace(
                        "**TASK:** Your ONLY task is to find ALL connections (lines/pipes) between the elements provided in the knowledge base.",
                        "**TASK:** Find ALL connections (lines/pipes) between the elements provided in the knowledge base. Additionally, if you find symbols from the legend that are NOT in the element list, also detect them."
                    ).replace(
                        "**CRITICAL RULES (STRICTLY ENFORCE):**\n1. **ONLY DETECT CONNECTIONS.** Do NOT detect elements.",
                        "**CRITICAL RULES (STRICTLY ENFORCE):**\n1. **PRIMARY TASK: DETECT CONNECTIONS** between elements in the knowledge base.\n2. **SECONDARY TASK: DETECT ADDITIONAL ELEMENTS** from the legend that are NOT in the element list."
                    ).replace(
                        "**1. \"elements\" List:**\n- CRITICAL: Provide an EMPTY list. You MUST NOT detect elements.\n- ` \"elements\": [] `",
                        "**1. \"elements\" List:**\n- PRIMARY: Provide an EMPTY list (connections are your main task).\n- SECONDARY: If you find symbols from the legend that are NOT in the element list above, add them here.\n- Only add elements that match symbols in the legend symbol map.\n- Use EXACT type names from the legend (case-sensitive, exact spacing)."
                    )
                    
                    monolith_prompt = monolith_prompt_template_enhanced.replace(
                        "{ignore_zones_str}", "[]"
                    ).replace(
                        "[{component_list_str}]", f"[{component_list_str}]"
                    ).replace(
                        "{element_list_json}", element_list_json  # Use provided element list
                    ).replace(
                        "{legend_context}", legend_context_str
                    ).replace(
                        "{viewshot_valve_examples}", viewshot_context.get('valve', '')
                    ).replace(
                        "{viewshot_flow_sensor_examples}", viewshot_context.get('flow_sensor', '')
                    ).replace(
                        "{viewshot_mixer_examples}", viewshot_context.get('mixer', '')
                    ).replace(
                        "{viewshot_source_examples}", viewshot_context.get('source', '')
                    ).replace(
                        "{viewshot_sample_point_examples}", viewshot_context.get('sample_point', '')
                    ).replace(
                        "{viewshot_pump_examples}", viewshot_context.get('pump', '')
                    ).replace(
                        "{viewshot_sink_examples}", viewshot_context.get('sink', '')
                    ).replace(
                        "{viewshot_section}", viewshot_context.get('section', '')
                    )
                else:
                    # No legend: Standard prompt (only connections, no additional elements)
                    monolith_prompt = monolith_prompt_template.replace(
                        "{ignore_zones_str}", "[]"
                    ).replace(
                        "[{component_list_str}]", f"[{component_list_str}]"
                    ).replace(
                        "{element_list_json}", element_list_json  # Use provided element list
                    ).replace(
                        "{legend_context}", legend_context_str
                    ).replace(
                        "{viewshot_valve_examples}", viewshot_context.get('valve', '')
                    ).replace(
                        "{viewshot_flow_sensor_examples}", viewshot_context.get('flow_sensor', '')
                    ).replace(
                        "{viewshot_mixer_examples}", viewshot_context.get('mixer', '')
                    ).replace(
                        "{viewshot_source_examples}", viewshot_context.get('source', '')
                    ).replace(
                        "{viewshot_sample_point_examples}", viewshot_context.get('sample_point', '')
                    ).replace(
                        "{viewshot_pump_examples}", viewshot_context.get('pump', '')
                    ).replace(
                        "{viewshot_sink_examples}", viewshot_context.get('sink', '')
                    ).replace(
                        "{viewshot_section}", viewshot_context.get('section', '')
                    )
            
            # Add error feedback if available
            if self.error_feedback:
                monolith_prompt += f"\n\n**ERROR FEEDBACK (FIX THESE ISSUES):**\n{self.error_feedback}\n"
            
            # ENHANCED LOGGING: Log monolith analysis call
            request_id = f"monolith_whole_{int(time.time())}"
            llm_logger.info(
                f"MONOLITH_WHOLE_IMAGE [image={image_path}] [prompt_length={len(monolith_prompt)}]",
                extra={'request_id': request_id}
            )
            
            # Call LLM with whole image
            try:
                logger.debug(f"[MONOLITH] Calling LLM with prompt length: {len(monolith_prompt)}")
                logger.debug(f"[MONOLITH] Prompt preview: {monolith_prompt[:500]}...")
                
                response = self.llm_client.call_llm(
                    detail_model_info,
                    system_prompt,
                    monolith_prompt,
                    image_path,
                    expected_json_keys=["elements", "connections"]
                )
                
                # ENHANCED LOGGING: Log raw response
                if response:
                    if hasattr(response, 'text'):
                        raw_response = response.text
                        llm_logger.debug(
                            f"MONOLITH_RESPONSE_RAW: {raw_response[:1000]}...",
                            extra={'request_id': request_id}
                        )
                    elif isinstance(response, dict):
                        raw_response = json.dumps(response, indent=2, ensure_ascii=False)
                        llm_logger.debug(
                            f"MONOLITH_RESPONSE_DICT: {raw_response[:1000]}...",
                            extra={'request_id': request_id}
                        )
                    else:
                        llm_logger.warning(
                            f"MONOLITH_RESPONSE_UNKNOWN_TYPE: {type(response)} - {str(response)[:500]}",
                            extra={'request_id': request_id}
                        )
                
            except Exception as e:
                llm_logger.error(
                    f"MONOLITH_ERROR: {type(e).__name__} - {str(e)}",
                    extra={'request_id': request_id}, exc_info=True
                )
                logger.error(f"Error in whole-image analysis: {e}", exc_info=True)
                return {"elements": [], "connections": []}
            
            # FIX: Handle both dict and string responses from LLM client
            if response:
                if isinstance(response, dict):
                    elements = response.get("elements", [])
                    connections = response.get("connections", [])
                    logger.info(f"[MONOLITH] Successfully parsed response: {len(elements)} elements, {len(connections)} connections")
                    llm_logger.info(
                        f"MONOLITH_SUCCESS [elements={len(elements)}] [connections={len(connections)}]",
                        extra={'request_id': request_id}
                    )
                    return {
                        "elements": elements,
                        "connections": connections
                    }
                elif isinstance(response, str):
                    # Try to parse JSON from string response
                    try:
                        parsed = json.loads(response)
                        if isinstance(parsed, dict):
                            elements = parsed.get("elements", [])
                            connections = parsed.get("connections", [])
                            logger.info(f"[MONOLITH] Successfully parsed JSON string: {len(elements)} elements, {len(connections)} connections")
                            llm_logger.info(
                                f"MONOLITH_SUCCESS [elements={len(elements)}] [connections={len(connections)}]",
                                extra={'request_id': request_id}
                            )
                            return {
                                "elements": elements,
                                "connections": connections
                            }
                    except json.JSONDecodeError as e:
                        logger.warning(f"[MONOLITH] Could not parse JSON from string response: {e}")
                        logger.warning(f"[MONOLITH] Response preview: {response[:200]}...")
                        llm_logger.warning(
                            f"MONOLITH_PARSE_ERROR: {str(e)}",
                            extra={'request_id': request_id}
                        )
                        return {"elements": [], "connections": []}
                elif hasattr(response, 'text'):
                    # Response is a Gemini response object
                    try:
                        text = response.text.strip()
                        # Try to extract JSON from markdown code blocks
                        import re
                        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                        if json_match:
                            parsed = json.loads(json_match.group(1))
                            if isinstance(parsed, dict):
                                elements = parsed.get("elements", [])
                                connections = parsed.get("connections", [])
                                logger.info(f"[MONOLITH] Successfully extracted JSON from markdown: {len(elements)} elements, {len(connections)} connections")
                                llm_logger.info(
                                    f"MONOLITH_SUCCESS [elements={len(elements)}] [connections={len(connections)}]",
                                    extra={'request_id': request_id}
                                )
                                return {
                                    "elements": elements,
                                    "connections": connections
                                }
                        # Try direct JSON parse
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            elements = parsed.get("elements", [])
                            connections = parsed.get("connections", [])
                            logger.info(f"[MONOLITH] Successfully parsed direct JSON: {len(elements)} elements, {len(connections)} connections")
                            llm_logger.info(
                                f"MONOLITH_SUCCESS [elements={len(elements)}] [connections={len(connections)}]",
                                extra={'request_id': request_id}
                            )
                            return {
                                "elements": elements,
                                "connections": connections
                            }
                    except json.JSONDecodeError as e:
                        logger.warning(f"[MONOLITH] Could not parse JSON from response.text: {e}")
                        logger.warning(f"[MONOLITH] Response.text preview: {response.text[:200] if hasattr(response, 'text') else 'N/A'}...")
                        llm_logger.warning(
                            f"MONOLITH_PARSE_ERROR: {str(e)}",
                            extra={'request_id': request_id}
                        )
            
            logger.warning("[MONOLITH] Invalid response from LLM for whole-image analysis")
            logger.warning(f"[MONOLITH] Response type: {type(response)}, Response value: {str(response)[:200] if response else 'None'}...")
            llm_logger.warning(
                f"MONOLITH_INVALID_RESPONSE [type={type(response)}]",
                extra={'request_id': request_id}
            )
            return {"elements": [], "connections": []}
                
        except Exception as e:
            logger.error(f"Error in whole-image analysis: {e}", exc_info=True)
            return {"elements": [], "connections": []}
    
    def _load_viewshot_examples(self) -> Dict[str, str]:
        """
        Load viewshot examples from training_data/viewshot_examples/ directory.
        
        Returns:
            Dictionary with viewshot example strings for each type
        """
        viewshot_context = {
            'valve': '',
            'flow_sensor': '',
            'mixer': '',
            'source': '',
            'sample_point': '',
            'pump': '',
            'sink': '',
            'section': ''
        }
        
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            viewshot_dir = project_root / "training_data" / "viewshot_examples"
            
            if not viewshot_dir.exists():
                logger.debug(f"Viewshot directory not found: {viewshot_dir}")
                return viewshot_context
            
            # Map type names to directory names
            type_to_dir = {
                'valve': 'valve',
                'flow_sensor': 'flow_sensor',
                'mixer': 'mixer',
                'source': 'source',
                'sample_point': 'sample_point',
                'pump': 'pump',
                'sink': 'sink'
            }
            
            # Load viewshots for each type
            for type_key, dir_name in type_to_dir.items():
                type_dir = viewshot_dir / dir_name
                if type_dir.exists():
                    viewshot_files = list(type_dir.glob("*.png"))
                    if viewshot_files:
                        # Take first 3 viewshots as examples
                        examples = viewshot_files[:3]
                        viewshot_str = f"\n\n**VIEWSHOT EXAMPLES FOR {type_key.upper().replace('_', ' ')} (from real Uni-Bilder):**\n"
                        viewshot_str += f"**CRITICAL:** These are actual visual examples from Uni-Bilder. Use these to recognize similar symbols.\n"
                        for idx, vs_path in enumerate(examples, 1):
                            viewshot_str += f"\n**Viewshot Example {idx}:** {vs_path.name}\n"
                            viewshot_str += f"- Visual: Real {type_key} symbol from Uni-Bild (see image)\n"
                            viewshot_str += f"- Common pattern: Use this visual pattern to identify similar {type_key} symbols\n"
                        viewshot_context[type_key] = viewshot_str
            
            # Build combined viewshot section
            viewshot_section = ""
            if any(viewshot_context[key] for key in ['valve', 'flow_sensor', 'mixer', 'source', 'sample_point', 'pump', 'sink']):
                viewshot_section = "\n\n**VIEWSHOT EXAMPLES (VISUAL REFERENCE FROM REAL UNI-BILDER):**\n"
                viewshot_section += "**CRITICAL:** The viewshots above are actual cropped symbols from Uni-Bilder. "
                viewshot_section += "Use these visual patterns to identify similar symbols in the current image.\n"
                viewshot_section += "**IMPORTANT:** When you see a symbol that matches a viewshot pattern, use the EXACT type from the viewshot.\n"
            
            viewshot_context['section'] = viewshot_section
            
        except Exception as e:
            logger.debug(f"Error loading viewshot examples: {e}")
        
        return viewshot_context
    
    def _build_legend_context_str(self, legend_context: Optional[Dict[str, Any]]) -> str:
        """
        Build legend context string for prompts using PRIORITIZED logic.
        
        Args:
            legend_context: Legend context dictionary
            
        Returns:
            Formatted legend context string
        """
        legend_context_str = ""
        if not legend_context:
            return ""

        symbol_map = legend_context.get('symbol_map', {})
        line_map = legend_context.get('line_map', {})
        
        if not symbol_map and not line_map:
            return ""  # Keine Legende gefunden

        import json
        logger.info(f"Building PRIORITIZED legend context: {len(symbol_map)} symbols, {len(line_map)} lines.")

        # --- TEIL 1: VERBINDLICHE REGELN (HÖCHSTE HOHEIT) ---
        legend_context_str += "\n\n**--- TEIL 1: VERBINDLICHE REGELN (HÖCHSTE HOHEIT) ---\n"
        legend_context_str += "Eine Legende wurde auf dem Diagramm gefunden. Diese Einträge sind verbindlich.\n"
        
        if symbol_map:
            symbol_map_json = json.dumps(symbol_map, indent=2)
            legend_context_str += f"**Symbol-Map (Legende):**\n```json\n{symbol_map_json}\n```\n"
        
        if line_map:
            line_map_json = json.dumps(line_map, indent=2)
            legend_context_str += f"**Line-Map (Legende):**\n```json\n{line_map_json}\n```\n"
            
        legend_context_str += (
            "**Anweisung Teil 1:** Wenn ein Symbol oder eine Linie im Diagramm *exakt* einem Eintrag in "
            "dieser Legende entspricht, **MUSST** du den Typ exakt so verwenden, wie in der Legende angegeben.\n"
        )

        # --- TEIL 2: FALLBACK-REGELN (ALLGEMEINWISSEN) ---
        legend_context_str += "\n**--- TEIL 2: FALLBACK-REGELN (ALLGEMEINWISSEN) ---\n"
        legend_context_str += (
            "Es ist möglich, dass die Legende (Teil 1) unvollständig ist oder das Diagramm Symbole enthält, "
            "die *nicht* in der Legende definiert sind.\n"
        )
        legend_context_str += (
            "Nur für Symbole, die **NICHT** in der Legende (Teil 1) definiert sind, darfst du auf dein "
            "Allgemeinwissen (den Stammdaten-Katalog) zurückgreifen, der im Prompt bereits vorhanden ist.\n"
        )
        legend_context_str += (
            "**Anweisung Teil 2:** Wenn ein Symbol *nicht* in der Legende (Teil 1) definiert ist, "
            "klassifiziere es basierend auf dem Stammdaten-Katalog (dein `[{component_list_str}]`).\n"
        )
        
        legend_context_str += "\n**ZUSAMMENFASSUNG REGELN:**\n"
        legend_context_str += "1. **Priorisiere Teil 1:** Die Legende ist die höchste Wahrheit.\n"
        legend_context_str += "2. **Nutze Teil 2 als Fallback:** Dein Allgemeinwissen ist für alles andere da, was die Legende nicht abdeckt.\n"

        return legend_context_str

