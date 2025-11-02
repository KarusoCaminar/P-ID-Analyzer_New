"""
Swarm Analyzer - Tiled analysis for component detection.

Analyzes P&ID diagrams by dividing them into overlapping tiles
and processing them in parallel for maximum component coverage.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from src.interfaces.analyzer import IAnalyzer
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService
from src.utils.image_utils import generate_raster_grid, is_tile_complex
from src.utils.graph_utils import GraphSynthesizer, SynthesizerConfig, calculate_iou
from src.utils.type_utils import is_valid_bbox

logger = logging.getLogger(__name__)


class SwarmAnalyzer(IAnalyzer):
    """
    Swarm analyzer for tiled component detection.
    
    Divides the image into overlapping tiles and analyzes them in parallel
    for comprehensive component detection across the entire diagram.
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
        Initialize swarm analyzer.
        
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
        Analyze image using swarm (tiled) approach.
        
        Args:
            image_path: Path to image
            output_dir: Optional output directory
            excluded_zones: Zones to exclude from analysis
            
        Returns:
            Dictionary with 'elements' and 'connections' keys
        """
        logger.info("Starting swarm analysis...")
        
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            logger.error(f"Image not found or could not be opened: {e}")
            return {"elements": [], "connections": []}
        
        output_dir = output_dir or Path(os.path.dirname(image_path))
        temp_dir_path = output_dir / "temp_swarm_tiles"
        temp_dir_path.mkdir(exist_ok=True, parents=True)
        
        excluded_zones = excluded_zones or []
        
        # Generate adaptive tile grid
        target_tile_count = self.logic_parameters.get('adaptive_target_tile_count', 40)
        tile_size = int(np.sqrt((img_width * img_height) / target_tile_count))
        tile_size = int(np.clip(tile_size, 256, 1024))
        overlap = int(tile_size * 0.15)
        
        all_tiles = generate_raster_grid(
            image_path,
            tile_size,
            overlap,
            excluded_zones,
            temp_dir_path
        )
        
        # Filter complex tiles
        swarm_tiles = [tile for tile in all_tiles if is_tile_complex(tile['path'])]
        logger.info(f"Adaptive tiling: {len(all_tiles)} tiles generated, {len(swarm_tiles)} relevant.")
        
        # Limit tile count
        max_tiles = self.logic_parameters.get('max_total_tiles', 80)
        if len(swarm_tiles) > max_tiles:
            swarm_tiles = swarm_tiles[:max_tiles]
            logger.warning(f"Tile count limited to {len(swarm_tiles)}.")
        
        # Process tiles in parallel
        # Get legend_context from instance attribute if set
        legend_context = getattr(self, 'legend_context', None)
        raw_results = self._process_tiles_parallel(swarm_tiles, image_path, legend_context)
        
        # Synthesize results
        synthesizer_config = SynthesizerConfig(
            iou_match_threshold=self.logic_parameters.get('iou_match_threshold', 0.5)
        )
        synthesizer = GraphSynthesizer(raw_results, img_width, img_height, config=synthesizer_config)
        swarm_graph = synthesizer.synthesize()
        
        # Clean up
        if temp_dir_path.exists():
            import shutil
            shutil.rmtree(temp_dir_path)
        
        logger.info(f"Swarm analysis complete: {len(swarm_graph['elements'])} elements, {len(swarm_graph['connections'])} connections")
        
        return swarm_graph
    
    def _process_tiles_parallel(
        self,
        tiles: List[Dict[str, Any]],
        image_path: str,
        legend_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process tiles in parallel."""
        raw_results = []
        
        # Prepare knowledge context (few-shot examples from learning database if available)
        few_shot_prompt_part = ""
        try:
            # Try to get few-shot examples from learning database
            learning_db = self.knowledge_manager.learning_database
            if learning_db and isinstance(learning_db, dict):
                # Get most recent successful analysis as example
                runs = learning_db.get('runs', [])
                if runs:
                    # Get last successful run
                    last_run = runs[-1] if isinstance(runs, list) else None
                    if last_run and isinstance(last_run, dict):
                        final_data = last_run.get('final_ai_data', {})
                        if final_data:
                            example_elements = final_data.get('elements', [])[:3]  # First 3 elements
                            example_connections = final_data.get('connections', [])[:2]  # First 2 connections
                            few_shot_example = {
                                'elements': example_elements,
                                'connections': example_connections
                            }
                            few_shot_prompt_part = f"\n\nFEW-SHOT EXAMPLE (Follow this structure exactly):\n{json.dumps(few_shot_example, indent=2)}"
        except Exception as e:
            logger.debug(f"Could not load few-shot examples: {e}")
            pass  # Continue without few-shot examples
        
        knowledge_context: Dict[str, Any] = {
            "known_types": self.knowledge_manager.get_known_types(),
            "type_aliases": self.knowledge_manager.get_all_aliases(),
        }
        known_types_json = json.dumps(knowledge_context, indent=2)
        
        # Build legend context string for prompts
        legend_context_str = ""
        if legend_context:
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
        
        # Get prompts and model
        # Handle PromptsConfig (Pydantic model) or dict
        if isinstance(self.prompts, dict):
            raster_prompt_template = self.prompts.get('raster_analysis_user_prompt_template')
            system_prompt = self.prompts.get('general_system_prompt')
        else:
            # PromptsConfig Pydantic model - use attribute access
            raster_prompt_template = getattr(self.prompts, 'raster_analysis_user_prompt_template', None)
            system_prompt = getattr(self.prompts, 'general_system_prompt', None) or 'You are an expert in analyzing technical diagrams.'
        detail_model_info = self.model_strategy.get('detail_model')
        
        if not all([raster_prompt_template, system_prompt, detail_model_info]):
            logger.error("Configuration for detail analysis incomplete. Aborting swarm analysis.")
            return []
        
        # Process tiles
        max_workers = self.logic_parameters.get('llm_executor_workers', 4)
        timeout = self.logic_parameters.get('llm_default_timeout', 120)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {}
            
            for tile in tiles:
                # Build prompt
                optimized_prompt = raster_prompt_template.replace(
                    "{known_types_json}", known_types_json
                ).replace(
                    "{few_shot_prompt_part}", few_shot_prompt_part
                ).replace(
                    "{error_feedback}", ""
                ).replace(
                    "{legend_context}", legend_context_str
                )
                
                future = executor.submit(
                    self.llm_client.call_llm,
                    detail_model_info,
                    system_prompt,
                    optimized_prompt,
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
                        # Add confidence scores to elements and connections
                        if isinstance(response_data, dict):
                            elements = response_data.get('elements', [])
                            connections = response_data.get('connections', [])
                            
                            # Add confidence scores based on tile analysis quality
                            tile_confidence = 0.8  # Default confidence for tile-based detection
                            
                            for el in elements:
                                if 'confidence' not in el:
                                    el['confidence'] = tile_confidence
                            
                            for conn in connections:
                                if 'confidence' not in conn:
                                    conn['confidence'] = tile_confidence
                        
                        raw_results.append({
                            'tile_coords': tile['coords'],
                            'tile_width': tile['tile_width'],
                            'tile_height': tile['tile_height'],
                            'data': response_data
                        })
                except Exception as exc:
                    logger.error(f"Error analyzing tile '{tile['path']}': {exc}", exc_info=True)
        
        return raw_results
    
    def _identify_uncertain_zones(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """
        Identify uncertain zones (hotspots) based on element and connection density.
        
        Hotspots are areas with:
        - Low element density (potential missed elements)
        - High connection density but low element density (complex areas)
        - Low confidence elements (uncertain detections)
        """
        if not elements:
            return []
        
        uncertain_zones = []
        
        try:
            # Calculate element density map
            element_positions = []
            low_confidence_positions = []
            
            for el in elements:
                bbox = el.get('bbox', {})
                if not bbox:
                    continue
                
                center_x = bbox.get('x', 0) + bbox.get('width', 0) / 2
                center_y = bbox.get('y', 0) + bbox.get('height', 0) / 2
                confidence = el.get('confidence', 0.5)
                
                element_positions.append((center_x, center_y, confidence))
                
                # Track low confidence elements
                if confidence < 0.6:
                    low_confidence_positions.append((center_x, center_y))
            
            # Identify sparse regions (potential missed elements)
            if len(element_positions) > 1:
                # Grid-based density analysis
                grid_size = 0.1  # 10x10 grid
                grid_counts = {}
                
                for x, y, conf in element_positions:
                    grid_x = int(x / grid_size)
                    grid_y = int(y / grid_size)
                    key = (grid_x, grid_y)
                    
                    if key not in grid_counts:
                        grid_counts[key] = {'count': 0, 'confidence_sum': 0}
                    
                    grid_counts[key]['count'] += 1
                    grid_counts[key]['confidence_sum'] += conf
                
                # Find sparse grids (low density areas)
                avg_density = len(element_positions) / (1.0 / (grid_size * grid_size))
                
                for (grid_x, grid_y), data in grid_counts.items():
                    density = data['count']
                    avg_conf = data['confidence_sum'] / density if density > 0 else 0
                    
                    # Mark as uncertain if:
                    # 1. Low density compared to average
                    # 2. Low average confidence
                    if density < avg_density * 0.5 or avg_conf < 0.6:
                        zone_x = grid_x * grid_size
                        zone_y = grid_y * grid_size
                        uncertain_zones.append({
                            'x': zone_x,
                            'y': zone_y,
                            'width': grid_size,
                            'height': grid_size,
                            'uncertainty': 1.0 - (density / max(avg_density, 1.0)) if avg_density > 0 else 1.0
                        })
            
            # Identify connection-heavy areas with low element density (complex hotspots)
            connection_positions = []
            for conn in connections:
                from_id = conn.get('from_id')
                to_id = conn.get('to_id')
                
                # Find element positions for this connection
                from_el = next((el for el in elements if el.get('id') == from_id), None)
                to_el = next((el for el in elements if el.get('id') == to_id), None)
                
                if from_el and to_el and from_el.get('bbox') and to_el.get('bbox'):
                    from_center_x = from_el['bbox'].get('x', 0) + from_el['bbox'].get('width', 0) / 2
                    from_center_y = from_el['bbox'].get('y', 0) + from_el['bbox'].get('height', 0) / 2
                    to_center_x = to_el['bbox'].get('x', 0) + to_el['bbox'].get('width', 0) / 2
                    to_center_y = to_el['bbox'].get('y', 0) + to_el['bbox'].get('height', 0) / 2
                    
                    # Midpoint of connection
                    mid_x = (from_center_x + to_center_x) / 2
                    mid_y = (from_center_y + to_center_y) / 2
                    connection_positions.append((mid_x, mid_y))
            
            # Identify areas with many connections but few elements (complex hotspots)
            if connection_positions and element_positions:
                for conn_x, conn_y in connection_positions:
                    # Count nearby elements
                    nearby_elements = sum(
                        1 for el_x, el_y, _ in element_positions
                        if abs(el_x - conn_x) < 0.05 and abs(el_y - conn_y) < 0.05
                    )
                    
                    # Count nearby connections
                    nearby_connections = sum(
                        1 for c_x, c_y in connection_positions
                        if abs(c_x - conn_x) < 0.05 and abs(c_y - conn_y) < 0.05
                    )
                    
                    # Hotspot: many connections but few elements
                    if nearby_connections > 3 and nearby_elements < 2:
                        uncertain_zones.append({
                            'x': max(0, conn_x - 0.05),
                            'y': max(0, conn_y - 0.05),
                            'width': 0.1,
                            'height': 0.1,
                            'uncertainty': min(1.0, nearby_connections / 5.0)
                        })
            
            # Add zones around low confidence elements
            for low_x, low_y in low_confidence_positions:
                uncertain_zones.append({
                    'x': max(0, low_x - 0.03),
                    'y': max(0, low_y - 0.03),
                    'width': 0.06,
                    'height': 0.06,
                    'uncertainty': 0.7
                })
            
            # Deduplicate overlapping zones
            if len(uncertain_zones) > 1:
                deduplicated = []
                for zone in uncertain_zones:
                    overlap = False
                    for existing in deduplicated:
                        # Check if zones overlap significantly
                        if (abs(zone['x'] - existing['x']) < zone['width'] and
                            abs(zone['y'] - existing['y']) < zone['height']):
                            overlap = True
                            # Merge: keep higher uncertainty
                            existing['uncertainty'] = max(existing['uncertainty'], zone['uncertainty'])
                            break
                    
                    if not overlap:
                        deduplicated.append(zone)
                
                uncertain_zones = deduplicated
            
            logger.info(f"Identified {len(uncertain_zones)} uncertain zones (hotspots)")
            
        except Exception as e:
            logger.error(f"Error identifying uncertain zones: {e}", exc_info=True)
        
        return uncertain_zones

