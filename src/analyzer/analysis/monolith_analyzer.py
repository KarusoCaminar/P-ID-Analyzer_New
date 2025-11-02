"""
Monolith Analyzer - Quadrant-based analysis for structure detection.

Analyzes P&ID diagrams by dividing them into 4 large overlapping quadrants
to capture structural relationships and connections.
"""

import os
import logging
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
        logic_parameters: Dict[str, Any]
    ):
        """
        Initialize monolith analyzer.
        
        Args:
            llm_client: LLM client for analysis
            knowledge_manager: Knowledge manager for type resolution
            config_service: Configuration service
            model_strategy: Model strategy configuration
            logic_parameters: Logic parameters for analysis
        """
        self.llm_client = llm_client
        self.knowledge_manager = knowledge_manager
        self.config_service = config_service
        self.model_strategy = model_strategy
        self.logic_parameters = logic_parameters
        
        config = config_service.get_config()
        self.prompts = config.prompts
    
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
        
        # Generate quadrant tiles (large, overlapping)
        tile_size = int(max(img_width, img_height) * 0.6)
        overlap = int(tile_size * 0.2)
        
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
        detail_model_info = self.model_strategy.get('detail_model')
        
        if not all([monolith_prompt_template, system_prompt, detail_model_info]):
            logger.error("Configuration for monolith analysis incomplete.")
            return []
        
        # Build component list
        known_types_list = self.knowledge_manager.get_known_types()
        component_list_str = ", ".join(f"'{name}'" for name in known_types_list)
        
        # Build legend context string for prompts
        legend_context_str = ""
        if legend_context:
            import json
            symbol_map = legend_context.get('symbol_map', {})
            line_map = legend_context.get('line_map', {})
            
            if symbol_map:
                legend_context_str += "\n\n**LEGEND INFORMATION (Extracted from this P&ID diagram):**\n"
                legend_context_str += "**Symbol Map (Use these symbol definitions from the legend):**\n"
                for symbol_key, symbol_type in symbol_map.items():
                    legend_context_str += f"- {symbol_key} → {symbol_type}\n"
            
            if line_map:
                legend_context_str += "\n**Line Map (Pipe/Line colors and styles from legend):**\n"
                for line_key, line_info in line_map.items():
                    if isinstance(line_info, dict):
                        color = line_info.get('color', 'unknown')
                        style = line_info.get('style', 'unknown')
                        legend_context_str += f"- {line_key}: color={color}, style={style}\n"
                    else:
                        legend_context_str += f"- {line_key}: {line_info}\n"
            
            if legend_context_str:
                legend_context_str += "\n**IMPORTANT:** Use the symbol and line definitions above to identify components correctly. "
                legend_context_str += "If you see a symbol that matches one in the legend, use the exact type from the legend.\n"
        
        monolith_prompt = monolith_prompt_template.replace(
            "{ignore_zones_str}", "[]"
        ).replace(
            "[{component_list_str}]", f"[{component_list_str}]"
        ).replace(
            "{legend_context}", legend_context_str
        )
        
        # Process quadrants
        max_workers = self.logic_parameters.get('llm_executor_workers', 4)
        timeout = self.logic_parameters.get('llm_default_timeout', 120)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {}
            
            for tile in tiles:
                future = executor.submit(
                    self.llm_client.call_llm,
                    detail_model_info,
                    system_prompt,
                    monolith_prompt,
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

