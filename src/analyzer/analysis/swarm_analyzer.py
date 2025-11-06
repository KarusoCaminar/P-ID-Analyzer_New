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
from src.analyzer.learning.symbol_library import SymbolLibrary
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
        logic_parameters: Dict[str, Any],
        symbol_library: Optional[SymbolLibrary] = None
    ):
        """
        Initialize swarm analyzer.
        
        Args:
            llm_client: LLM client for analysis
            knowledge_manager: Knowledge manager for type resolution
            config_service: Configuration service
            model_strategy: Model strategy configuration
            logic_parameters: Logic parameters for analysis
            symbol_library: Optional symbol library for symbol recognition
        """
        self.llm_client = llm_client
        self.knowledge_manager = knowledge_manager
        self.config_service = config_service
        self.model_strategy = model_strategy
        self.logic_parameters = logic_parameters
        self.symbol_library = symbol_library
        
        config = config_service.get_config()
        self.prompts = config.prompts
        
            # Additional attributes
        self.error_feedback = ""
        self.legend_context: Optional[Dict[str, Any]] = None
    
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
    
    def _calculate_optimal_tile_count(
        self,
        img_width: int,
        img_height: int,
        target_tile_count: int,
        complex_tile_count: int
    ) -> int:
        """
        Calculate optimal tile count based on image size and complexity.
        
        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            target_tile_count: Target tile count from config
            complex_tile_count: Number of complex tiles found
            
        Returns:
            Optimal maximum tile count
        """
        # Base calculation on image size
        image_area = img_width * img_height
        max_dimension = max(img_width, img_height)
        
        # PERFORMANCE: Calculate base max_tiles based on image size (optimized for speed)
        if max_dimension < 2000:
            # Small images: use fewer tiles (efficiency) - 20-30 tiles max for simple P&IDs
            base_max_tiles = min(30, target_tile_count)
        elif max_dimension < 4000:
            # Medium images: moderate tile count - 40-50 tiles max
            base_max_tiles = min(50, int(target_tile_count * 1.0))
        else:
            # Large images: allow more tiles for precision
            base_max_tiles = min(120, int(target_tile_count * 1.5))
        
        # Adjust based on complexity: if many complex tiles found, allow more
        complexity_factor = min(1.3, 1.0 + (complex_tile_count / (target_tile_count * 2)))
        optimal_max_tiles = int(base_max_tiles * complexity_factor)
        
        # Ensure we don't exceed the absolute maximum from config
        absolute_max = self.logic_parameters.get('max_total_tiles', 80)
        optimal_max_tiles = min(optimal_max_tiles, absolute_max)
        
        # Ensure we have at least some tiles if complex tiles exist
        if complex_tile_count > 0:
            optimal_max_tiles = max(optimal_max_tiles, min(20, complex_tile_count))
        
        logger.info(f"Dynamic tile strategy: image={img_width}x{img_height}, "
                   f"target={target_tile_count}, complex={complex_tile_count}, "
                   f"optimal_max={optimal_max_tiles}")
        
        return optimal_max_tiles
    
    def _calculate_tile_priority(
        self,
        tile: Dict[str, Any],
        image_width: int,
        image_height: int
    ) -> float:
        """
        Calculate priority for a tile based on various factors.
        
        Args:
            tile: Tile dictionary with 'x', 'y', 'width', 'height' keys (normalized 0-1)
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Priority score (higher = more important)
        """
        priority = 0.0
        
        # Convert normalized coordinates to pixel coordinates for calculations
        tile_x = tile.get('x', 0.0) * image_width
        tile_y = tile.get('y', 0.0) * image_height
        tile_w = tile.get('width', 0.0) * image_width
        tile_h = tile.get('height', 0.0) * image_height
        tile_area = tile_w * tile_h
        
        # Factor 1: Hotspot tiles (tiles with adaptive sizing get higher priority)
        # Small tiles (512px) indicate hotspots - higher priority
        if tile_area < 512 * 512:
            priority += 10.0
        
        # Factor 2: Tile size (smaller tiles = higher priority for detail)
        # Smaller tiles indicate areas requiring more attention
        normalized_area = tile_area / (image_width * image_height)
        if normalized_area < 0.05:  # Very small tiles
            priority += 3.0
        elif normalized_area < 0.1:  # Small tiles
            priority += 1.5
        
        # Factor 3: Adaptive sizing bonus
        # If tile is part of adaptive sizing strategy (marked in tile dict)
        if tile.get('adaptive', False):
            priority += 3.0
        
        # Factor 4: Center bias (center of image often contains important elements)
        center_x = tile_x + tile_w / 2
        center_y = tile_y + tile_h / 2
        image_center_x = image_width / 2
        image_center_y = image_height / 2
        
        distance_from_center = ((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2) ** 0.5
        max_distance = ((image_width / 2) ** 2 + (image_height / 2) ** 2) ** 0.5
        center_proximity = 1.0 - (distance_from_center / max_distance) if max_distance > 0 else 0.0
        priority += center_proximity * 1.0  # Up to 1.0 bonus for center proximity
        
        return priority
    
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
        
        # Two-Pass Pipeline: Coarse → Refine
        two_pass_enabled = self.logic_parameters.get('two_pass_enabled', False)
        max_dimension = max(img_width, img_height)
        
        # Use two-pass for large images (>4000px)
        if two_pass_enabled and max_dimension > 4000:
            logger.info("Using Two-Pass Pipeline (Coarse → Refine) for large image")
            return self._analyze_two_pass(
                image_path, output_dir, excluded_zones, img_width, img_height, temp_dir_path
            )
        
        # Single-pass for smaller images
        # Generate adaptive tile grid
        target_tile_count = self.logic_parameters.get('adaptive_target_tile_count', 40)
        tile_size = int(np.sqrt((img_width * img_height) / target_tile_count))
        tile_size = int(np.clip(tile_size, 256, 1024))
        overlap = int(tile_size * 0.15)
        
        # Calculate tile priorities before processing
        
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
        
        # Calculate priorities and sort tiles (highest priority first)
        prioritized_tiles = []
        for tile in swarm_tiles:
            priority = self._calculate_tile_priority(tile, img_width, img_height)
            prioritized_tiles.append((priority, tile))
        
        # Sort by priority (highest first)
        prioritized_tiles.sort(key=lambda x: x[0], reverse=True)
        swarm_tiles = [tile for _, tile in prioritized_tiles]
        logger.info(f"Sorted {len(swarm_tiles)} tiles by priority")
        
        # Calculate optimal tile count dynamically based on image size and complexity
        optimal_max_tiles = self._calculate_optimal_tile_count(
            img_width=img_width,
            img_height=img_height,
            target_tile_count=target_tile_count,
            complex_tile_count=len(swarm_tiles)
        )
        
        # Limit tile count using dynamic calculation
        if len(swarm_tiles) > optimal_max_tiles:
            swarm_tiles = swarm_tiles[:optimal_max_tiles]
            logger.info(f"Tile count limited to {len(swarm_tiles)} (optimal for {img_width}x{img_height} image).")
        
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
    
    def _analyze_two_pass(
        self,
        image_path: str,
        output_dir: Optional[Path],
        excluded_zones: Optional[List[Dict[str, float]]],
        img_width: int,
        img_height: int,
        temp_dir_path: Path
    ) -> Dict[str, Any]:
        """
        Two-Pass Pipeline: Coarse → Refine strategy.
        
        Pass 1 (Coarse): Large tiles (1024px) for overview
        Pass 2 (Refine): Small tiles (512px) for uncertain zones only
        
        Args:
            image_path: Path to image
            output_dir: Output directory
            excluded_zones: Zones to exclude
            img_width: Image width
            img_height: Image height
            temp_dir_path: Temporary directory for tiles
            
        Returns:
            Dictionary with 'elements' and 'connections' keys
        """
        logger.info("=== Two-Pass Pipeline: Starting Pass 1 (Coarse) ===")
        
        # Pass 1: Coarse (large tiles)
        coarse_tile_size = self.logic_parameters.get('coarse_tile_size', 1024)
        coarse_overlap = self.logic_parameters.get('coarse_overlap', 0.33)
        coarse_overlap_px = int(coarse_tile_size * coarse_overlap)
        
        coarse_tiles = generate_raster_grid(
            image_path,
            coarse_tile_size,
            coarse_overlap_px,
            excluded_zones,
            temp_dir_path / "coarse"
        )
        
        # Process coarse tiles
        legend_context = getattr(self, 'legend_context', None)
        coarse_results = self._process_tiles_parallel(coarse_tiles, image_path, legend_context)
        
        # Synthesize coarse results
        synthesizer_config = SynthesizerConfig(
            iou_match_threshold=self.logic_parameters.get('iou_match_threshold', 0.5)
        )
        coarse_synthesizer = GraphSynthesizer(coarse_results, img_width, img_height, config=synthesizer_config)
        coarse_graph = coarse_synthesizer.synthesize()
        
        logger.info(f"Pass 1 complete: {len(coarse_graph['elements'])} elements, {len(coarse_graph['connections'])} connections")
        
        # CRITICAL FIX 3: Remove non-deterministic uncertainty zone logic
        # Uncertainty zone identification is now handled by pipeline_coordinator (Phase 3)
        # For Two-Pass Pipeline, use simple refinement based on low confidence elements only
        uncertain_zones = []
        
        # Simple refinement: Only refine areas with low confidence elements
        low_confidence_threshold = self.logic_parameters.get('low_confidence_threshold', 0.7)
        elements = coarse_graph.get('elements', [])
        for el in elements:
            confidence = el.get('confidence', 0.5)
            bbox = el.get('bbox')
            if confidence < low_confidence_threshold and bbox:
                zone = {
                    'x': max(0.0, bbox.get('x', 0) - 0.05),
                    'y': max(0.0, bbox.get('y', 0) - 0.05),
                    'width': min(1.0 - bbox.get('x', 0), bbox.get('width', 0) + 0.1),
                    'height': min(1.0 - bbox.get('y', 0), bbox.get('height', 0) + 0.1)
                }
                uncertain_zones.append(zone)
        
        # Deduplicate overlapping zones
        uncertain_zones = self._deduplicate_zones(uncertain_zones)
        
        logger.info(f"Identified {len(uncertain_zones)} uncertain zones for refinement (based on low confidence elements only)")
        
        # Pass 2: Refine (small tiles for uncertain zones only)
        if uncertain_zones:
            logger.info("=== Two-Pass Pipeline: Starting Pass 2 (Refine) ===")
            
            refine_tile_size = self.logic_parameters.get('refine_tile_size', 512)
            refine_overlap = self.logic_parameters.get('refine_overlap', 0.5)
            refine_overlap_px = int(refine_tile_size * refine_overlap)
            
            # Generate refine tiles only for uncertain zones
            refine_tiles = self._generate_refine_tiles(
                image_path,
                uncertain_zones,
                refine_tile_size,
                refine_overlap_px,
                excluded_zones,
                temp_dir_path / "refine"
            )
            
            # Limit refine tiles based on budget
            max_refine_tiles = self.logic_parameters.get('max_refine_tiles', 80)
            if len(refine_tiles) > max_refine_tiles:
                # Prioritize refine tiles by uncertainty
                refine_tiles = self._prioritize_refine_tiles(refine_tiles, uncertain_zones)[:max_refine_tiles]
                logger.info(f"Refine tiles limited to {len(refine_tiles)} (budget: {max_refine_tiles})")
            
            # Process refine tiles
            refine_results = self._process_tiles_parallel(refine_tiles, image_path, legend_context)
            
            # Synthesize refine results
            refine_synthesizer = GraphSynthesizer(refine_results, img_width, img_height, config=synthesizer_config)
            refine_graph = refine_synthesizer.synthesize()
            
            logger.info(f"Pass 2 complete: {len(refine_graph['elements'])} elements, {len(refine_graph['connections'])} connections")
            
            # Merge coarse and refine results
            merged_graph = self._merge_coarse_refine(coarse_graph, refine_graph)
            
            logger.info(f"Two-Pass Pipeline complete: {len(merged_graph['elements'])} elements, {len(merged_graph['connections'])} connections")
            
            return merged_graph
        else:
            logger.info("No uncertain zones found. Using coarse results only.")
            return coarse_graph
    
    # CRITICAL FIX 3: Removed _identify_uncertain_zones method
    # Uncertainty zone identification is now handled deterministically in _analyze_two_pass
    # This method was non-deterministic and made testing impossible
    
    def _deduplicate_zones(self, zones: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Deduplicate overlapping zones."""
        if not zones:
            return []
        
        from src.utils.graph_utils import calculate_iou
        
        deduplicated = []
        for zone in zones:
            is_duplicate = False
            for existing_zone in deduplicated:
                iou = calculate_iou(zone, existing_zone)
                if iou > 0.5:  # High overlap
                    is_duplicate = True
                    # Merge zones (union)
                    existing_zone['x'] = min(existing_zone['x'], zone['x'])
                    existing_zone['y'] = min(existing_zone['y'], zone['y'])
                    existing_zone['width'] = max(
                        existing_zone['x'] + existing_zone['width'],
                        zone['x'] + zone['width']
                    ) - existing_zone['x']
                    existing_zone['height'] = max(
                        existing_zone['y'] + existing_zone['height'],
                        zone['y'] + zone['height']
                    ) - existing_zone['y']
                    break
            
            if not is_duplicate:
                deduplicated.append(zone)
        
        return deduplicated
    
    def _generate_refine_tiles(
        self,
        image_path: str,
        uncertain_zones: List[Dict[str, float]],
        tile_size: int,
        overlap: int,
        excluded_zones: List[Dict[str, float]],
        temp_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Generate refine tiles for uncertain zones.
        
        Args:
            image_path: Path to image
            uncertain_zones: List of uncertain zone bboxes (normalized)
            tile_size: Tile size in pixels
            overlap: Overlap in pixels
            excluded_zones: Zones to exclude
            temp_dir: Temporary directory for tiles
            
        Returns:
            List of refine tiles
        """
        refine_tiles = []
        
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            logger.error(f"Could not read image for refine tiles: {e}")
            return []
        
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        for zone in uncertain_zones:
            # Convert normalized bbox to pixel coordinates
            x_px = int(zone['x'] * img_width)
            y_px = int(zone['y'] * img_height)
            w_px = int(zone['width'] * img_width)
            h_px = int(zone['height'] * img_height)
            
            # Generate tiles for this zone
            stride = tile_size - overlap
            
            for tile_y in range(y_px, y_px + h_px, stride):
                for tile_x in range(x_px, x_px + w_px, stride):
                    # Ensure tile is within image bounds
                    tile_x = max(0, min(img_width - tile_size, tile_x))
                    tile_y = max(0, min(img_height - tile_size, tile_y))
                    
                    # Crop tile
                    tile_img = img.crop((tile_x, tile_y, tile_x + tile_size, tile_y + tile_size))
                    
                    # Save tile
                    tile_path = temp_dir / f"refine_tile_{tile_x}_{tile_y}.png"
                    tile_img.save(tile_path)
                    
                    refine_tiles.append({
                        'path': str(tile_path),
                        'x': tile_x,
                        'y': tile_y,
                        'width': tile_size,
                        'height': tile_size,
                        'normalized_x': tile_x / img_width,
                        'normalized_y': tile_y / img_height,
                        'normalized_width': tile_size / img_width,
                        'normalized_height': tile_size / img_height
                    })
        
        logger.info(f"Generated {len(refine_tiles)} refine tiles for {len(uncertain_zones)} uncertain zones")
        return refine_tiles
    
    def _prioritize_refine_tiles(
        self,
        refine_tiles: List[Dict[str, Any]],
        uncertain_zones: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Prioritize refine tiles by uncertainty level."""
        # Simple prioritization: tiles in zones with more issues have higher priority
        prioritized = []
        
        for tile in refine_tiles:
            # Calculate priority based on zone overlap
            priority = 1.0
            
            # Higher priority for tiles in uncertain zones
            tile_center_x = tile.get('normalized_x', 0) + tile.get('normalized_width', 0) / 2
            tile_center_y = tile.get('normalized_y', 0) + tile.get('normalized_height', 0) / 2
            
            for zone in uncertain_zones:
                zone_x = zone.get('x', 0)
                zone_y = zone.get('y', 0)
                zone_w = zone.get('width', 0)
                zone_h = zone.get('height', 0)
                
                # Check if tile center is in zone
                if (zone_x <= tile_center_x <= zone_x + zone_w and
                    zone_y <= tile_center_y <= zone_y + zone_h):
                    priority += 2.0
                    break
            
            prioritized.append((priority, tile))
        
        # Sort by priority (highest first)
        prioritized.sort(key=lambda x: x[0], reverse=True)
        return [tile for _, tile in prioritized]
    
    def _merge_coarse_refine(
        self,
        coarse_graph: Dict[str, Any],
        refine_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge coarse and refine results.
        
        Args:
            coarse_graph: Coarse analysis results
            refine_graph: Refine analysis results
            
        Returns:
            Merged graph with refined elements
        """
        from src.utils.graph_utils import GraphSynthesizer, SynthesizerConfig
        from src.utils.type_utils import calculate_iou
        
        # Combine elements and connections
        all_elements = coarse_graph.get('elements', []) + refine_graph.get('elements', [])
        all_connections = coarse_graph.get('connections', []) + refine_graph.get('connections', [])
        
        # Deduplicate using GraphSynthesizer
        synthesizer_config = SynthesizerConfig(
            iou_match_threshold=self.logic_parameters.get('iou_match_threshold', 0.5)
        )
        
        # Get image dimensions from first element bbox
        img_width, img_height = 1000, 1000  # Default
        if all_elements:
            first_bbox = all_elements[0].get('bbox', {})
            if first_bbox:
                # Estimate from normalized bbox
                img_width = int(1.0 / max(first_bbox.get('width', 0.01), 0.001))
                img_height = int(1.0 / max(first_bbox.get('height', 0.01), 0.001))
        
        synthesizer = GraphSynthesizer(
            [{'elements': all_elements, 'connections': all_connections}],
            img_width,
            img_height,
            config=synthesizer_config
        )
        
        merged_graph = synthesizer.synthesize()
        
        logger.info(f"Merged coarse and refine: {len(merged_graph['elements'])} elements, "
                   f"{len(merged_graph['connections'])} connections")
        
        return merged_graph
    
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
        # CRITICAL FIX 2: PRIORITIZED legend context (not exclusive)
        legend_context_str = ""
        if legend_context:
            symbol_map = legend_context.get('symbol_map', {})
            line_map = legend_context.get('line_map', {})
            
            if symbol_map or line_map:
                logger.info(f"Building PRIORITIZED legend context for Swarm: {len(symbol_map)} symbols, {len(line_map)} lines.")

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
                    "klassifiziere es basierend auf dem Stammdaten-Katalog (dein `known_types_json`).\n"
                )
                
                legend_context_str += "\n**ZUSAMMENFASSUNG REGELN:**\n"
                legend_context_str += "1. **Priorisiere Teil 1:** Die Legende ist die höchste Wahrheit.\n"
                legend_context_str += "2. **Nutze Teil 2 als Fallback:** Dein Allgemeinwissen ist für alles andere da, was die Legende nicht abdeckt.\n"
        
        # Get prompts and model
        # Handle PromptsConfig (Pydantic model) or dict
        # CRITICAL FIX: Use swarm_specialist_prompt_template if available and simple_pid_strategy is active
        use_specialist_prompt = False
        if isinstance(self.prompts, dict):
            raster_prompt_template = self.prompts.get('raster_analysis_user_prompt_template')
            swarm_specialist_prompt = self.prompts.get('swarm_specialist_prompt_template')
            system_prompt = self.prompts.get('general_system_prompt')
        else:
            # PromptsConfig Pydantic model - use attribute access
            raster_prompt_template = getattr(self.prompts, 'raster_analysis_user_prompt_template', None)
            swarm_specialist_prompt = getattr(self.prompts, 'swarm_specialist_prompt_template', None)
            system_prompt = getattr(self.prompts, 'general_system_prompt', None) or 'You are an expert in analyzing technical diagrams.'
        
        # Check if we should use specialist prompt (for simple_pid_strategy)
        if swarm_specialist_prompt:
            # Check if swarm_model is Flash-Lite (indicator of simple_pid_strategy)
            swarm_model = self.model_strategy.get('swarm_model') or self.model_strategy.get('detail_model')
            if swarm_model and ('Flash-Lite' in str(swarm_model) or 'simple_pid_strategy' in str(self.logic_parameters.get('strategy', '')).lower()):
                use_specialist_prompt = True
                raster_prompt_template = swarm_specialist_prompt
                logger.info("Using Swarm Specialist Prompt (focused on SamplePoint & ISA-Supply)")
        # IMPROVED: Prefer swarm_model, fallback to detail_model
        swarm_model = self.model_strategy.get('swarm_model') or self.model_strategy.get('detail_model')
        detail_model_info = swarm_model
        
        if not all([raster_prompt_template, system_prompt, detail_model_info]):
            logger.error("Configuration for detail analysis incomplete. Aborting swarm analysis.")
            return []
        
        # Process tiles with controlled concurrency to limit API overload
        max_workers = self.logic_parameters.get('llm_executor_workers', 4)
        timeout = self.logic_parameters.get('llm_default_timeout', 120)
        
        # Limit max_workers to prevent API overload (max 8 concurrent requests)
        max_workers = min(max_workers, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {}
            
            # Submit all tiles but ThreadPoolExecutor will limit concurrent execution
            for tile in tiles:
                # Check symbol library before LLM call (if available)
                symbol_hints = ""
                if self.symbol_library:
                    try:
                        with Image.open(tile['path']) as tile_img:
                            # PHASE 3.1: Lower threshold from 0.85 to 0.75 for aggressive symbol library usage
                            # Find similar symbols in library (aggressive threshold for speed)
                            similar_symbols = self.symbol_library.find_similar_symbols(
                                tile_img,
                                top_k=3,
                                threshold=0.75  # Lowered from 0.85 to 0.75 for more aggressive matching
                            )
                            if similar_symbols:
                                # AGGRESSIVE: If similarity >= 0.85, provide strong hints + Few-Shot Examples
                                best_match = similar_symbols[0] if similar_symbols else None
                                # PHASE 3.1: Lower threshold for pre-filtering (0.9 similarity)
                                if best_match and best_match[1] >= 0.9:
                                    # PHASE 3.1: High confidence match - use library results as strong hints
                                    # Still call LLM but with very strong hints from library
                                    symbol_type = best_match[2].get('element_type', 'Unknown')
                                    symbol_hints = f"\n\n**CRITICAL: HIGH CONFIDENCE SYMBOL LIBRARY MATCH (similarity: {best_match[1]:.2f} >= 0.9):**\n"
                                    symbol_hints += f"**FEW-SHOT EXAMPLE:** This symbol is '{symbol_type}' (from symbol library with 90%+ similarity).\n"
                                    symbol_hints += f"- Visual: This symbol matches a known '{symbol_type}' symbol EXACTLY.\n"
                                    symbol_hints += f"- Type: '{symbol_type}' (EXACT spelling, case-sensitive)\n"
                                    symbol_hints += f"- WRONG: '{symbol_type.lower()}', '{symbol_type.upper()}', variations\n"
                                    symbol_hints += f"- CORRECT: '{symbol_type}' (exact type from symbol library)\n"
                                    symbol_hints += f"**CRITICAL:** Use EXACT type '{symbol_type}' for this symbol. This is a library match with 90%+ similarity.\n"
                                else:
                                    # Lower confidence - provide hints + Few-Shot Examples
                                    symbol_hints = "\n\n**KNOWN SYMBOLS DETECTED (from symbol library):**\n"
                                    symbol_hints += "**FEW-SHOT EXAMPLES (use these exact types):**\n"
                                    for symbol_id, similarity, metadata in similar_symbols[:3]:  # Top 3
                                        symbol_type = metadata.get('element_type', 'Unknown')
                                        symbol_hints += f"\n**FEW-SHOT EXAMPLE {symbol_id}:**\n"
                                        symbol_hints += f"- Visual: Symbol similar to known '{symbol_type}' (similarity: {similarity:.2f})\n"
                                        symbol_hints += f"- Type: '{symbol_type}' (EXACT spelling, case-sensitive)\n"
                                        symbol_hints += f"- If you see a symbol similar to this, use EXACT type '{symbol_type}'.\n"
                                    symbol_hints += "\n**IMPORTANT:** If you see symbols matching the above, use the exact type from the symbol library.\n"
                    except Exception as e:
                        logger.debug(f"Error checking symbol library for tile {tile['path']}: {e}")
                
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
                                text_hints += f"- Found {len(text_regions)} text regions in this tile.\n"
                                text_hints += "**IMPORTANT:** These regions likely contain labels (e.g., 'P-101', 'V-42'). "
                                text_hints += "Extract labels from these regions and associate them with nearby symbols.\n"
                    except Exception as e:
                        logger.debug(f"Error in CV text detection for tile {tile['path']}: {e}")
                
                # Load and format viewshot examples
                viewshot_context = self._load_viewshot_examples()
                
                # Build prompt with symbol hints and viewshots
                optimized_prompt = raster_prompt_template.replace(
                    "{known_types_json}", known_types_json
                ).replace(
                    "{few_shot_prompt_part}", few_shot_prompt_part
                ).replace(
                    "{error_feedback}", self.error_feedback if hasattr(self, 'error_feedback') else ""
                ).replace(
                    "{legend_context}", legend_context_str + symbol_hints
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
                            
                            # CRITICAL FIX 1: PENALTY for missing confidence (not reward)
                            tile_confidence_penalty = 0.1  # PENALTY: Failed to provide confidence
                            
                            for el in elements:
                                if 'confidence' not in el:
                                    el['confidence'] = tile_confidence_penalty
                                    logger.warning(f"Swarm tile element '{el.get('id', 'unknown')}' missing confidence - applying penalty (0.1)")
                            
                            for conn in connections:
                                if 'confidence' not in conn:
                                    conn['confidence'] = tile_confidence_penalty
                                    logger.warning(f"Swarm tile connection '{conn.get('from_id', '?')} -> {conn.get('to_id', '?')}' missing confidence - applying penalty (0.1)")
                        
                        raw_results.append({
                            'tile_coords': tile['coords'],
                            'tile_width': tile['tile_width'],
                            'tile_height': tile['tile_height'],
                            'data': response_data
                        })
                except Exception as exc:
                    logger.error(f"Error analyzing tile '{tile['path']}': {exc}", exc_info=True)
        
        return raw_results
    
    # CRITICAL FIX 3: Removed second _identify_uncertain_zones method
    # This method was non-deterministic and made testing impossible
    # Uncertainty zone identification is now handled deterministically in _analyze_two_pass

