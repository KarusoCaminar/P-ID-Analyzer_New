"""
Pipeline Coordinator - Main orchestration class for P&ID analysis.

Refactored from core_processor.py with:
- Phase-based architecture
- Dependency injection
- Type-safe data structures
- Clean separation of concerns
"""

import os
import json
import time
import logging
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from src.interfaces.processor import IProcessor
from src.analyzer.models.pipeline import PipelineState, AnalysisResult
from src.analyzer.models.elements import Element, Connection
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.learning.active_learner import ActiveLearner
from src.services.config_service import ConfigService

logger = logging.getLogger(__name__)


class ProgressCallback:
    """Interface for progress updates."""
    def update_progress(self, value: int, message: str) -> None:
        """Update progress (0-100)."""
        pass
    
    def update_status_label(self, text: str) -> None:
        """Update status message."""
        pass
    
    def report_truth_mode(self, active: bool) -> None:
        """Report truth mode status."""
        pass
    
    def report_correction(self, correction_text: str) -> None:
        """Report correction information."""
        pass


class PipelineCoordinator(IProcessor):
    """
    Main pipeline coordinator for P&ID analysis.
    
    Orchestrates all phases of the analysis:
    1. Pre-analysis (metadata, legend)
    2. Parallel core analysis (swarm + monolith)
    3. Fusion
    4. Predictive completion
    5. Polyline refinement
    6. Self-correction loop
    7. Post-processing (KPIs, CGM, artifacts)
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_manager: KnowledgeManager,
        config_service: ConfigService,
        model_strategy: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        """
        Initialize the pipeline coordinator.
        
        Args:
            llm_client: LLM client for AI interactions
            knowledge_manager: Knowledge manager for learning and type resolution
            config_service: Configuration service
            model_strategy: Optional model strategy override
            progress_callback: Optional progress callback
        """
        self.llm_client = llm_client
        self.knowledge_manager = knowledge_manager
        self.config_service = config_service
        
        # Load model strategy from config if not provided
        if model_strategy:
            self.model_strategy = model_strategy
        else:
            # Get default strategy from config
            config = config_service.get_raw_config()
            strategies = config.get('strategies', {})
            default_strategy = strategies.get('default_flash', {})
            
            # Convert strategy model names to model configs
            self.model_strategy = {}
            models_config = config.get('models', {})
            for key, model_name in default_strategy.items():
                if model_name in models_config:
                    model_info = models_config[model_name]
                    if isinstance(model_info, dict):
                        self.model_strategy[key] = model_info
                    else:
                        self.model_strategy[key] = model_info.model_dump() if hasattr(model_info, 'model_dump') else {}
            
            # Fallback: Use first available model for all steps if strategy is empty
            if not self.model_strategy and models_config:
                first_model = list(models_config.values())[0]
                model_info = first_model.model_dump() if hasattr(first_model, 'model_dump') else first_model if isinstance(first_model, dict) else {}
                self.model_strategy = {
                    'meta_model': model_info,
                    'detail_model': model_info,
                    'polyline_model': model_info,
                    'hotspot_model': model_info,
                    'correction_model': model_info
                }
        
        self.progress_callback = progress_callback
        
        # Pipeline state
        self.state = PipelineState(
            image_path="",
            current_phase="initialization",
            progress=0.0,
            elements=[],
            connections=[],
            excluded_zones=[],
            metadata=None,
            legend_data=None,
            start_time=None
        )
        
        # Internal state
        self.start_time: float = 0.0
        self.current_image_path: Optional[str] = None
        self.active_logic_parameters: Dict[str, Any] = {}
        self._analysis_results: Dict[str, Any] = {}
        self._global_knowledge_repo: Dict[str, Any] = {}
        self._excluded_zones: List[Dict[str, float]] = []
        
        # Active learning system
        from src.analyzer.learning.symbol_library import SymbolLibrary
        symbol_library = SymbolLibrary(llm_client)
        config_dict = config_service.get_config().model_dump() if hasattr(config_service.get_config(), 'model_dump') else {}
        self.active_learner = ActiveLearner(
            knowledge_manager=knowledge_manager,
            symbol_library=symbol_library,
            llm_client=llm_client,
            config=config_dict
        )
    
    def process(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        params_override: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Process a P&ID image and return analysis results.
        
        Args:
            image_path: Path to the P&ID image
            output_dir: Optional output directory for artifacts
            params_override: Optional parameter overrides
            
        Returns:
            AnalysisResult with detected elements and connections
        """
        # Get output directory first (needed for initialization)
        final_output_dir = output_dir or self._get_output_directory(image_path)
        
        # TEST HARNESS: Save configuration snapshot and test metadata if test_name provided
        if params_override and 'test_name' in params_override:
            from src.utils.test_harness import save_config_snapshot, save_test_metadata
            save_config_snapshot(self.config_service, final_output_dir)
            test_name = params_override.get('test_name', 'unknown_test')
            test_description = params_override.get('test_description', 'No description provided')
            save_test_metadata(
                final_output_dir,
                test_name,
                test_description,
                self.model_strategy,
                self.active_logic_parameters
            )
        
        # Store output_dir for test harness
        self.current_output_dir = final_output_dir
        
        # Initialize run
        self._initialize_run(image_path, params_override)
        
        # Load truth data if available
        truth_data = self._load_truth_data(image_path)
        if self.progress_callback:
            self.progress_callback.report_truth_mode(active=bool(truth_data))
        
        # Detect P&ID type and adapt strategy
        metadata = self._detect_pid_type(image_path)
        if metadata:
            adaptation = self.active_learner.adapt_to_pid_type(metadata, {})
            if adaptation.get('strategy_adjustments'):
                # Apply adaptive strategy adjustments
                self.active_logic_parameters.update(adaptation.get('strategy_adjustments', {}))
        
        try:
            # CRITICAL FIX 1: Phase 0 MUST run BEFORE strategy selection
            # Phase 0: Complexity Analysis (CV-based fast analysis)
            self._update_progress(2, "Phase 0: Complexity Analysis...")
            self._run_phase_0_complexity_analysis(image_path)
            
            # Phase 1: Pre-analysis
            self._update_progress(5, "Phase 1: Pre-analysis...")
            self._run_phase_1_pre_analysis(image_path)
            
            # Phase 2: Core analysis - Different order for simple vs complex P&IDs
            # For simple P&IDs: Monolith first (recognizes elements + connections), then Swarm as hint list
            # For complex P&IDs: Swarm → Guard Rails → Monolith (current sequential approach)
            # CRITICAL: Now phase0_result is available (from Phase 0 above)
            phase0_result = self._analysis_results.get('phase0_result', {})
            strategy = phase0_result.get('strategy', 'optimal_swarm_monolith')

            use_swarm = self.active_logic_parameters.get('use_swarm_analysis', True)
            use_monolith = self.active_logic_parameters.get('use_monolith_analysis', True)

            if not use_swarm and not use_monolith:
                logger.error("Both swarm and monolith analysis are disabled. Cannot proceed.")
                return self._create_error_result("Both analyzers are disabled.")

            # CRITICAL: For simple P&IDs, use parallel analysis (Monolith + Swarm)
            if strategy == 'simple_pid_strategy':
                self._update_progress(15, "Phase 2: Simple P&ID analysis (Monolith + Swarm)...")
                logger.info("CRITICAL: Simple P&ID mode - Using parallel analysis (Monolith + Swarm)")
                swarm_result, monolith_result = self._run_phase_2_parallel_analysis(
                    image_path, final_output_dir
                )
            else:
                # Complex P&IDs: Parallel analysis (Swarm + Monolith)
                self._update_progress(15, "Phase 2: Parallel core analysis (Swarm + Monolith)...")
                swarm_result, monolith_result = self._run_phase_2_parallel_analysis(
                    image_path, final_output_dir
                )
            
            # Validate and correct coordinates (if validator exists)
            try:
                from src.utils.coordinate_validator import CoordinateValidator
                from PIL import Image
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                validator = CoordinateValidator(image_width=img_width, image_height=img_height)
                
                # Validate swarm elements
                if swarm_result and swarm_result.get("elements"):
                    validated_elements = []
                    for el in swarm_result.get("elements", []):
                        validated_el = validator.validate_element_coordinates(el)
                        if validated_el:
                            validated_elements.append(validated_el)
                    if validated_elements:
                        swarm_result["elements"] = validated_elements
                
                # Validate monolith elements
                if monolith_result and monolith_result.get("elements"):
                    validated_elements = []
                    for el in monolith_result.get("elements", []):
                        validated_el = validator.validate_element_coordinates(el)
                        if validated_el:
                            validated_elements.append(validated_el)
                    if validated_elements:
                        monolith_result["elements"] = validated_elements
            except ImportError:
                logger.warning("CoordinateValidator not available, skipping coordinate validation")
            except Exception as e:
                logger.warning(f"Coordinate validation failed: {e}")
            
            if not swarm_result or not swarm_result.get("elements"):
                logger.error("Initial analysis failed.")
                return self._create_error_result("Initial analysis failed.")
            
            # Phase 2c: Fusion
            self._update_progress(45, "Phase 2c: Fusion...")
            fused_result = self._run_phase_2c_fusion(swarm_result, monolith_result)
            self._analysis_results = fused_result
            
            # CRITICAL: Check for missing legend symbols and add them if found by Monolith
            # If legend is present, ensure all legend symbols are detected
            legend_data = self._analysis_results.get('legend_data', {})
            symbol_map = legend_data.get('symbol_map', {})
            if symbol_map:
                logger.info(f"Checking for missing legend symbols: {len(symbol_map)} symbols in legend")
                fused_result = self._add_missing_legend_symbols(fused_result, symbol_map, monolith_result)
                self._analysis_results = fused_result
            
            # Phase 2d: Predictive completion
            self._update_progress(50, "Phase 2d: Predictive completion...")
            self._run_phase_2d_predictive_completion()
            
            # Phase 2e: Polyline refinement
            self._update_progress(55, "Phase 2e: Polyline refinement...")
            self._run_phase_2e_polyline_refinement()
            
            # Phase 3: Self-correction loop
            self._update_progress(60, "Phase 3: Self-correction loop...")
            best_result = self._run_phase_3_self_correction(
                image_path, final_output_dir, truth_data
            )
            
            # Phase 4: Post-processing
            self._update_progress(90, "Phase 4: Post-processing...")
            final_result = self._run_phase_4_post_processing(
                best_result, image_path, final_output_dir, truth_data
            )
            
            self._update_progress(100, "Analysis complete!")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return self._create_error_result(str(e))
    
    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.state
    
    def pretrain_symbols(
        self,
        pretraining_path: Path,
        model_info: Dict[str, Any]
    ) -> List[Dict]:
        """
        Train symbol library from pretraining images using active learning.
        
        Args:
            pretraining_path: Path to pretraining symbols directory
            model_info: Model configuration for symbol extraction
            
        Returns:
            List of learning reports
        """
        logger.info(f"Pretraining symbols from: {pretraining_path} using active learning")
        
        try:
            # Use active learner to learn from pretraining symbols
            learning_report = self.active_learner.learn_from_pretraining_symbols(
                pretraining_path=pretraining_path,
                model_info=model_info
            )
            
            logger.info(
                f"Pretraining complete: {learning_report.get('symbols_learned', 0)} symbols learned, "
                f"{learning_report.get('symbols_processed', 0)} symbols processed"
            )
            
            return [learning_report]
        except Exception as e:
            logger.error(f"Error in pretraining: {e}", exc_info=True)
            return []
    
    # ==================== Internal Methods ====================
    
    def _initialize_run(
        self,
        image_path: str,
        params_override: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize state for a new analysis run."""
        self.current_image_path = image_path
        self.start_time = time.time()
        
        # Load logic parameters
        config = self.config_service.get_config()
        self.active_logic_parameters = config.logic_parameters.model_dump()
        if params_override:
            self.active_logic_parameters.update(params_override)
        
        # Reset state
        self._reset_state()
        
        # Update pipeline state
        self.state = PipelineState(
            image_path=image_path,
            current_phase="initialization",
            progress=0.0,
            elements=[],
            connections=[],
            excluded_zones=[],
            metadata=None,
            legend_data=None,
            start_time=datetime.now()
        )
        
        logger.info(f"Initialized pipeline for: {os.path.basename(image_path)}")
    
    def _reset_state(self) -> None:
        """Reset internal state."""
        self._analysis_results = {}
        self._global_knowledge_repo = {}
        self._excluded_zones = []
    
    def _update_progress(self, progress: int, message: str) -> None:
        """Update progress and status."""
        if progress < 0:
            progress = 0
        if progress > 100:
            progress = 100
        
        self.state.current_phase = message
        self.state.progress = float(progress)
        
        if self.progress_callback:
            elapsed = time.time() - self.start_time if self.start_time else 0
            time_str = f"Elapsed: {elapsed:.1f}s"
            full_message = f"{message} ({progress}%) {time_str}"
            self.progress_callback.update_progress(progress, full_message)
            self.progress_callback.update_status_label(full_message)
        
        logger.info(f"Progress: {progress}% - {message}")
    
    def _get_output_directory(self, image_path: str) -> str:
        """Get output directory for results."""
        base_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs") / f"{base_name}_output_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)
    
    def _load_truth_data(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Load truth data if available.
        
        Searches for truth files in multiple locations with multiple naming patterns:
        - Same directory as image: {base_name}_truth_cgm.json, {base_name}_truth.json
        - training_data/organized_tests: recursive search
        """
        if not image_path:
            return None
        
        base_name = Path(image_path).stem
        image_dir = Path(image_path).parent
        
        # Try multiple naming patterns in same directory
        truth_patterns = [
            f"{base_name}_truth_cgm.json",
            f"{base_name}_truth.json",
            f"{base_name}_truth_cgm.json".replace(" ", "_"),
            f"{base_name}_truth.json".replace(" ", "_"),
        ]
        
        for pattern in truth_patterns:
            truth_file = image_dir / pattern
            if truth_file.exists():
                try:
                    logger.info(f"Loading truth data from: {truth_file}")
                    with open(truth_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading truth data from {truth_file}: {e}")
        
        # Search in training_data/organized_tests if not found in same directory
        training_data_dir = Path("training_data") / "organized_tests"
        if training_data_dir.exists():
            # Try all patterns recursively
            for pattern in truth_patterns:
                truth_files = list(training_data_dir.rglob(pattern))
                if truth_files:
                    # Prefer files in same subdirectory structure
                    for truth_file in truth_files:
                        # Check if base name matches (allowing for path differences)
                        if base_name.replace(" ", "_").lower() in truth_file.stem.lower() or \
                           truth_file.stem.lower().replace("_truth_cgm", "").replace("_truth", "") in base_name.lower():
                            try:
                                logger.info(f"Loading truth data from training_data: {truth_file}")
                                with open(truth_file, 'r', encoding='utf-8') as f:
                                    return json.load(f)
                            except Exception as e:
                                logger.error(f"Error loading truth data from {truth_file}: {e}")
            
            # Last resort: search by base name (fuzzy match)
            all_truth_files = list(training_data_dir.rglob("*truth*.json"))
            for truth_file in all_truth_files:
                # Check if base name appears in truth file name
                truth_base = truth_file.stem.replace("_truth_cgm", "").replace("_truth", "")
                if base_name.lower().replace(" ", "_") == truth_base.lower() or \
                   base_name.lower() == truth_base.lower():
                    try:
                        logger.info(f"Loading truth data (fuzzy match): {truth_file}")
                        with open(truth_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading truth data from {truth_file}: {e}")
        
        logger.warning(f"No truth data found for {base_name} (searched: {image_dir}, {training_data_dir})")
        return None
    
    def _create_error_result(self, error_message: str) -> AnalysisResult:
        """Create an error result."""
        return AnalysisResult(
            image_name=os.path.basename(self.current_image_path or "unknown"),
            elements=[],
            connections=[],
            quality_score=0.0,
            error_details={"error": error_message}
        )
    
    # ==================== Phase Methods (Stubs - to be implemented) ====================
    
    def _run_phase_0_complexity_analysis(self, image_path: str) -> None:
        """
        Phase 0: Complexity Analysis (CV-based fast analysis).
        
        Determines the optimal strategy based on complexity:
        - 'simple' -> 'simple_pid_strategy' (Monolith first)
        - 'moderate'/'complex'/'very_complex' -> 'optimal_swarm_monolith' (Swarm → Monolith)
        
        CRITICAL: This MUST run BEFORE strategy selection in process().
        """
        logger.info("--- Running Phase 0: Complexity Analysis (CV-based) ---")
        
        try:
            from src.utils.complexity_analyzer import ComplexityAnalyzer
            
            # Use CV-based advanced analysis (fast, no LLM needed)
            complexity_analyzer = ComplexityAnalyzer(llm_client=None)  # CV-only for speed
            complexity_result = complexity_analyzer.analyze_complexity_cv_advanced(image_path)
            
            # Determine strategy based on complexity
            complexity = complexity_result.get('complexity', 'moderate')
            if complexity == 'simple':
                strategy = 'simple_pid_strategy'
            else:
                strategy = 'optimal_swarm_monolith'
            
            # Store result for use in process() method
            phase0_result = {
                'complexity': complexity,
                'strategy': strategy,
                'score': complexity_result.get('score', 0.5),
                'metrics': complexity_result.get('metrics', {}),
                'cv_used': True,
                'llm_used': False
            }
            
            self._analysis_results['phase0_result'] = phase0_result
            
            logger.info(f"Phase 0 complete: complexity={complexity}, strategy={strategy}")
            logger.info(f"Complexity metrics: {complexity_result.get('metrics', {})}")
            
        except Exception as e:
            logger.error(f"Error in Phase 0 complexity analysis: {e}", exc_info=True)
            # Fallback to default strategy
            self._analysis_results['phase0_result'] = {
                'complexity': 'moderate',
                'strategy': 'optimal_swarm_monolith',
                'score': 0.5,
                'metrics': {},
                'cv_used': False,
                'llm_used': False,
                'error': str(e)
            }
            logger.warning("Phase 0 failed - using default strategy: optimal_swarm_monolith")
    
    def _run_phase_1_pre_analysis(self, image_path: str) -> None:
        """
        Phase 1: Pre-analysis (metadata, legend extraction).
        
        Extracts:
        - Metadata (project, title, version, date)
        - Legend (symbol_map, line_map)
        - Validates symbol mappings against knowledge base
        - Excludes metadata and legend areas from analysis
        """
        logger.info("--- Running Phase 1: Pre-Analysis ---")
        
        config = self.config_service.get_config()
        prompts = config.prompts
        
        # Handle PromptsConfig (Pydantic model) or dict
        if isinstance(prompts, dict):
            system_prompt = prompts.get('general_system_prompt', 'You are an expert in analyzing technical diagrams.')
            metadata_prompt = prompts.get('metadata_extraction_user_prompt')
            legend_prompt = prompts.get('legend_extraction_user_prompt')
        else:
            # PromptsConfig Pydantic model - use attribute access
            system_prompt = getattr(prompts, 'general_system_prompt', None) or 'You are an expert in analyzing technical diagrams.'
            metadata_prompt = getattr(prompts, 'metadata_extraction_user_prompt', None)
            legend_prompt = getattr(prompts, 'legend_extraction_user_prompt', None)
        
        if not metadata_prompt:
            logger.error("Metadata extraction prompt missing in config.yaml. Aborting Phase 1.")
            return
        if not legend_prompt:
            logger.error("Legend extraction prompt missing in config.yaml. Aborting Phase 1.")
            return
        
        # Get model info from strategy
        model_info = self.model_strategy.get('meta_model')
        if not model_info:
            logger.error("Meta model not defined in strategy. Aborting Phase 1.")
            return
        
        # Extract metadata
        logger.info("Extracting metadata...")
        metadata_response = self.llm_client.call_llm(
            model_info,
            system_prompt,
            metadata_prompt,
            image_path,
            expected_json_keys=["project", "title", "version", "date", "metadata_bbox"]
        )
        
        metadata_dict: Optional[Dict[str, Any]] = None
        if metadata_response and isinstance(metadata_response, dict):
            metadata_dict = metadata_response
            self._analysis_results['metadata'] = metadata_dict
            logger.info(f"Successfully extracted metadata: {metadata_dict}")
        else:
            logger.warning(f"LLM response for metadata was not a dict or was None: {metadata_response}")
            self._analysis_results['metadata'] = {}
        
        # Process metadata bbox
        raw_metadata_bbox = metadata_dict.get("metadata_bbox") if metadata_dict else None
        parsed_metadata_bbox = None
        
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            logger.error(f"Could not read image size for metadata bbox parsing: {e}")
            img_width, img_height = 1000, 1000  # Fallback
        
        parsed_metadata_bbox = self._parse_bbox(raw_metadata_bbox)
        if parsed_metadata_bbox:
            self._excluded_zones.append(parsed_metadata_bbox)
            self.state.excluded_zones.append(parsed_metadata_bbox)
            logger.info(f"Identified metadata area to be excluded: {parsed_metadata_bbox}")
        else:
            logger.warning(f"Metadata bbox is malformed or unparsable: {raw_metadata_bbox}")
        
        # Extract legend
        logger.info("Extracting legend...")
        legend_response = self.llm_client.call_llm(
            model_info,
            system_prompt,
            legend_prompt,
            image_path,
            expected_json_keys=["symbol_map", "legend_bbox"]
        )
        
        legend_dict: Optional[Dict[str, Any]] = None
        if legend_response and isinstance(legend_response, dict):
            legend_dict = legend_response
        else:
            logger.warning(f"LLM response for legend was not a dict or was None: {legend_response}")
        
        # Process legend bbox
        legend_bbox_raw = legend_dict.get("legend_bbox") if legend_dict else None
        parsed_legend_bbox = self._parse_bbox(legend_bbox_raw)
        
        if parsed_legend_bbox:
            self._excluded_zones.append(parsed_legend_bbox)
            self.state.excluded_zones.append(parsed_legend_bbox)
            logger.info(f"Identified legend area to be excluded: {parsed_legend_bbox}")
        else:
            logger.warning(f"Legend bbox is malformed or unparsable: {legend_bbox_raw}")
        
        # Validate symbol map
        symbol_map = legend_dict.get("symbol_map", {}) if legend_dict else {}
        validated_symbol_map = self._validate_symbol_map(symbol_map)
        self._global_knowledge_repo['symbol_map'] = validated_symbol_map
        
        # Process line map
        line_map = legend_dict.get("line_map", {}) if legend_dict else {}
        if line_map:
            self._global_knowledge_repo['line_map'] = line_map
        
        # CRITICAL: Legend Critic - Plausibility Check (Step 1.2)
        legend_confidence = 0.0
        is_plausible = False
        if validated_symbol_map or line_map:
            logger.info("Running Legend Critic (Plausibility Check)...")
            critic_result = self._run_legend_critic(validated_symbol_map, line_map)
            is_plausible = critic_result.get('is_plausible', False)
            legend_confidence = critic_result.get('confidence', 0.0)
            reason = critic_result.get('reason', 'No reason provided')
            
            logger.info(f"Legend Critic Result: is_plausible={is_plausible}, confidence={legend_confidence:.2f}")
            logger.info(f"Reason: {reason}")
            
            # Decision (Step 1.3)
            if is_plausible and legend_confidence > 0.8:
                logger.info("Legend is TRUSTED TRUTH (High Authority) - will be used with high priority")
            elif is_plausible:
                logger.info("Legend is PARTIAL TRUTH (Medium Authority) - will be used with medium priority")
            else:
                logger.warning("Legend is LOW CONFIDENCE (Low Authority) - will be used with low priority")
        
        # Store complete legend data for use in analysis and output
        complete_legend_data = {
            'symbol_map': validated_symbol_map,
            'line_map': line_map,
            'legend_bbox': parsed_legend_bbox,
            'legend_confidence': legend_confidence,
            'is_plausible': is_plausible
        }
        self.state.legend_data = complete_legend_data
        self._analysis_results['legend_data'] = complete_legend_data
        
        # Also store metadata
        if metadata_dict:
            self._analysis_results['metadata'] = metadata_dict
            self.state.metadata = metadata_dict
        
        logger.info(f"Built knowledge repository with {len(validated_symbol_map)} validated symbol mappings and {len(line_map)} line rules.")
        if line_map:
            logger.info(f"Extracted {len(line_map)} line semantic rules from legend (colors, styles).")
        
        logger.info("Phase 1 completed. Legend knowledge ready for analysis.")
    
    def _run_legend_critic(
        self,
        symbol_map: Dict[str, str],
        line_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Legend Critic: Plausibility check for extracted legend.
        
        Checks:
        - Number of symbols ≈ number of symbol names?
        - Number of lines ≈ number of line names?
        - Is the legend complete and logically consistent?
        
        Args:
            symbol_map: Extracted symbol map
            line_map: Extracted line map
            
        Returns:
            Dictionary with 'is_plausible', 'confidence', and 'reason'
        """
        try:
            # Use Flash model for fast critic check
            critic_model = self.model_strategy.get('swarm_model') or self.model_strategy.get('detail_model')
            if not critic_model:
                logger.warning("No critic model available, using default confidence")
                return {
                    'is_plausible': True,
                    'confidence': 0.7,
                    'reason': 'No critic model available - using default'
                }
            
            # Normalize model info (handle dict, Pydantic model, or string)
            if isinstance(critic_model, dict):
                model_info = critic_model
            elif hasattr(critic_model, 'model_dump'):
                model_info = critic_model.model_dump()
            elif hasattr(critic_model, 'dict'):
                model_info = critic_model.dict()
            elif isinstance(critic_model, str):
                # If it's a string (model name), get from config
                config = self.config_service.get_config()
                models_config = config.models if hasattr(config, 'models') else config.get('models', {})
                model_info = models_config.get(critic_model, {})
            else:
                model_info = None
            
            if not model_info:
                logger.warning("Could not normalize critic model, using default confidence")
                return {
                    'is_plausible': True,
                    'confidence': 0.7,
                    'reason': 'Could not normalize critic model - using default'
                }
            
            # Build critic prompt
            import json
            symbol_map_json = json.dumps(symbol_map, indent=2)
            line_map_json = json.dumps(line_map, indent=2)
            
            critic_prompt = f"""**ROLE:** You are a P&ID Legend Critic. Your task is to evaluate the plausibility of an extracted legend.

**EXTRACTED LEGEND DATA:**
```json
{{
  "symbol_map": {symbol_map_json},
  "line_map": {line_map_json}
}}
```

**PLAUSIBILITY RULES:**
1. **Symbol Count Check:** Number of symbols in symbol_map should approximately match the number of unique symbol names/keys.
2. **Line Count Check:** Number of lines in line_map should approximately match the number of unique line names/keys.
3. **Completeness Check:** The legend should be logically consistent (no obvious missing entries).
4. **Consistency Check:** Symbol types should be standard P&ID types (e.g., "Valve", "Pump", "Volume Flow Sensor").

**YOUR TASK:**
Evaluate the plausibility of this legend extraction. Consider:
- Are the symbol counts reasonable?
- Are the line counts reasonable?
- Is the legend complete and consistent?
- Are the symbol types standard?

**OUTPUT FORMAT:**
Return ONLY a valid JSON object:
```json
{{
  "is_plausible": true/false,
  "confidence": 0.0-1.0,
  "reason": "Brief explanation of your evaluation"
}}
```

**CRITICAL:** 
- If is_plausible=true and confidence > 0.8: Legend is TRUSTED TRUTH (High Authority)
- If is_plausible=true and confidence 0.5-0.8: Legend is PARTIAL TRUTH (Medium Authority)
- If is_plausible=false: Legend is LOW CONFIDENCE (Low Authority)

Be honest and strict in your evaluation."""
            
            system_prompt = "You are an expert P&ID legend critic. Evaluate legend plausibility strictly and honestly."
            
            # Call LLM for critic evaluation
            critic_response = self.llm_client.call_llm(
                model_info,
                system_prompt,
                critic_prompt,
                None,  # No image needed for critic
                expected_json_keys=["is_plausible", "confidence", "reason"]
            )
            
            if critic_response and isinstance(critic_response, dict):
                is_plausible = critic_response.get('is_plausible', False)
                confidence = float(critic_response.get('confidence', 0.0))
                reason = critic_response.get('reason', 'No reason provided')
                
                return {
                    'is_plausible': is_plausible,
                    'confidence': confidence,
                    'reason': reason
                }
            else:
                logger.warning("Legend Critic returned invalid response, using default")
                return {
                    'is_plausible': True,
                    'confidence': 0.7,
                    'reason': 'Critic returned invalid response - using default'
                }
        
        except Exception as e:
            logger.error(f"Error in Legend Critic: {e}", exc_info=True)
            return {
                'is_plausible': True,
                'confidence': 0.7,
                'reason': f'Error in critic: {str(e)} - using default'
            }
    
    def _parse_bbox(self, bbox_raw: Any) -> Optional[Dict[str, float]]:
        """Parse a bounding box from various formats."""
        from src.utils.type_utils import normalize_bbox
        
        if isinstance(bbox_raw, dict) and all(k in bbox_raw for k in ['x', 'y', 'width', 'height']):
            return {
                'x': float(bbox_raw['x']),
                'y': float(bbox_raw['y']),
                'width': float(bbox_raw['width']),
                'height': float(bbox_raw['height'])
            }
        elif isinstance(bbox_raw, list) and len(bbox_raw) == 4:
            try:
                x, y, width, height = bbox_raw
                return {
                    'x': float(x),
                    'y': float(y),
                    'width': float(width),
                    'height': float(height)
                }
            except (ValueError, IndexError):
                logger.error(f"Could not convert bbox list {bbox_raw} to dict format.")
                return None
        
        return None
    
    def _validate_symbol_map(self, symbol_map: Dict[str, Any]) -> Dict[str, str]:
        """Validate symbol map against knowledge base."""
        validated_map = {}
        
        if not isinstance(symbol_map, dict):
            logger.warning(f"Malformed 'symbol_map' in LLM response. Expected dict, got {type(symbol_map)}")
            return validated_map
        
        for key, value in symbol_map.items():
            symbol_type_name = None
            
            if isinstance(value, str):
                symbol_type_name = value
            elif isinstance(value, dict) and 'type' in value:
                symbol_type_name = value['type']
            elif isinstance(value, list) and len(value) >= 4:
                logger.warning(f"Malformed symbol_map value for key '{key}'. Looks like a bbox, not a type name.")
                continue
            else:
                logger.warning(f"Unexpected type for symbol_map value '{value}' (Key: '{key}').")
                continue
            
            if symbol_type_name:
                # CRITICAL FIX 3: STRICT VALIDATION - only accept resolved types
                # Try to resolve type name (finds also aliases/synonyms)
                element_type_data = self.knowledge_manager.find_element_type_by_name(symbol_type_name)
                
                if element_type_data:
                    # SUCCESS: Type was resolved - use official name
                    validated_map[key] = element_type_data['name']
                else:
                    # ERROR: Type is unknown and could not be resolved
                    logger.warning(
                        f"Type '{symbol_type_name}' from legend (Key: '{key}') "
                        f"could NOT be resolved against the knowledge base ('element_type_list.json'). "
                        f"This legend entry will be IGNORED."
                    )
                    # Do NOT add key to validated_map - strict validation
        
        return validated_map
    
    def _run_phase_2_parallel_analysis(
        self,
        image_path: str,
        output_dir: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Phase 2: Parallel analysis (swarm + monolith).
        
        Runs swarm and monolith analysis in parallel for efficiency.
        """
        logger.info("--- Running Phase 2: Parallel Core Analysis ---")
        
        from src.analyzer.analysis import SwarmAnalyzer, MonolithAnalyzer
        from concurrent.futures import ThreadPoolExecutor
        
        # Initialize analyzers
        swarm_analyzer = SwarmAnalyzer(
            self.llm_client,
            self.knowledge_manager,
            self.config_service,
            self.model_strategy,
            self.active_logic_parameters
        )
        
        monolith_analyzer = MonolithAnalyzer(
            self.llm_client,
            self.knowledge_manager,
            self.config_service,
            self.model_strategy,
            self.active_logic_parameters
        )
        
        # Prepare legend context for analyzers
        legend_context = {
            'symbol_map': self._global_knowledge_repo.get('symbol_map', {}),
            'line_map': self._global_knowledge_repo.get('line_map', {})
        }
        
        # Run in parallel
        swarm_graph: Dict[str, Any] = {}
        monolith_graph: Optional[Dict[str, Any]] = None
        
        output_path = Path(output_dir)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            logger.info("Starting swarm (components) and monolith (structure) analysis in parallel...")
            logger.info(f"Using legend context: {len(legend_context['symbol_map'])} symbols, {len(legend_context['line_map'])} line rules")
            
            # Store legend_context in analyzers for use in prompts
            swarm_analyzer.legend_context = legend_context
            monolith_analyzer.legend_context = legend_context
            
            swarm_future = executor.submit(
                swarm_analyzer.analyze,
                image_path,
                output_path,
                self._excluded_zones
            )
            
            monolith_future = executor.submit(
                monolith_analyzer.analyze,
                image_path,
                output_path,
                self._excluded_zones
            )
            
            try:
                swarm_result = swarm_future.result()
                if swarm_result:
                    swarm_graph = swarm_result
                
                monolith_result = monolith_future.result()
                if monolith_result:
                    monolith_graph = monolith_result
            except Exception as e:
                logger.error(f"Error during parallel analysis execution: {e}", exc_info=True)
        
        return swarm_graph, monolith_graph
    
    def _run_phase_2c_fusion(
        self,
        swarm_result: Dict[str, Any],
        monolith_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Phase 2c: Fusion of swarm and monolith results.
        
        Intelligently combines swarm (component-focused) and monolith (structure-focused)
        results to produce the best possible analysis.
        
        CRITICAL: Uses confidence-based fusion logic with legend authority.
        """
        from src.analyzer.analysis import FusionEngine
        
        # Get legend data for confidence-based fusion
        legend_data = self._analysis_results.get('legend_data', {})
        symbol_map = legend_data.get('symbol_map', {})
        line_map = legend_data.get('line_map', {})
        legend_confidence = legend_data.get('legend_confidence', 0.0)
        is_plausible = legend_data.get('is_plausible', False)
        
        iou_threshold = self.active_logic_parameters.get('iou_match_threshold', 0.1)
        fusion_engine = FusionEngine(iou_match_threshold=iou_threshold)
        
        # CRITICAL: Pass legend data for confidence-based fusion
        fused_result = fusion_engine.fuse_with_legend_authority(
            swarm_result, 
            monolith_result,
            symbol_map=symbol_map,
            legend_confidence=legend_confidence,
            is_plausible=is_plausible,
            line_map=line_map
        )
        
        # TEST HARNESS: Save intermediate result after fusion
        if hasattr(self, 'current_output_dir'):
            from src.utils.test_harness import save_intermediate_result
            save_intermediate_result("phase_2c_fusion", fused_result, self.current_output_dir)
        
        # Run metacritic review (cross-validation between monolith and swarm)
        if monolith_result and self.active_logic_parameters.get('use_metacritic', True):
            try:
                from src.analyzer.analysis.metacritic import Metacritic
                
                config = self.config_service.get_config()
                config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.__dict__
                
                metacritic = Metacritic(
                    self.llm_client,
                    config_dict,
                    self.model_strategy
                )
                
                discrepancies = metacritic.review(monolith_result, swarm_result)
                
                # Store discrepancies for later use in corrections
                if discrepancies:
                    fused_result['metacritic_discrepancies'] = discrepancies
                    logger.info(f"Metacritic found {len(discrepancies)} discrepancies - will be used in corrections")
            
            except Exception as e:
                logger.warning(f"Metacritic review failed: {e} - continuing without metacritic")
        
        return fused_result
    
    def _add_missing_legend_symbols(
        self,
        fused_result: Dict[str, Any],
        symbol_map: Dict[str, str],
        monolith_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add missing legend symbols that were found by Monolith but not by Swarm.
        
        This ensures that if Swarm missed a symbol from the legend, Monolith can still find it
        and it will be added to the final result.
        
        Args:
            fused_result: Current fused result
            symbol_map: Legend symbol map (symbol_key -> type)
            monolith_result: Monolith analysis result (may contain additional elements)
            
        Returns:
            Updated fused result with missing legend symbols added
        """
        if not symbol_map:
            return fused_result
        
        if not monolith_result:
            return fused_result
        
        # Get current elements
        current_elements = fused_result.get('elements', [])
        current_element_ids = {el.get('id') for el in current_elements if el.get('id')}
        current_element_types = {el.get('type') for el in current_elements if el.get('type')}
        
        # Get legend types
        legend_types = set(symbol_map.values())
        
        # Check if all legend types are detected
        missing_types = legend_types - current_element_types
        
        if not missing_types:
            logger.info("All legend symbol types are already detected.")
            return fused_result
        
        logger.info(f"Missing legend symbol types: {missing_types}")
        
        # Check if Monolith found any of these missing types
        monolith_elements = monolith_result.get('elements', [])
        added_count = 0
        
        for monolith_el in monolith_elements:
            el_type = monolith_el.get('type', '')
            el_id = monolith_el.get('id', '')
            
            # Check if this element type is in legend but missing from fused result
            if el_type in missing_types:
                # Check if element ID is not already in fused result
                if el_id not in current_element_ids:
                    # Add element from Monolith
                    current_elements.append(monolith_el)
                    current_element_ids.add(el_id)
                    added_count += 1
                    logger.info(f"Added missing legend symbol: {el_id} (type: {el_type}) from Monolith")
        
        if added_count > 0:
            logger.info(f"Added {added_count} missing legend symbols from Monolith to fused result")
            fused_result['elements'] = current_elements
        else:
            logger.warning(f"Monolith did not find any of the missing legend symbol types: {missing_types}")
        
        return fused_result
    
    def _run_phase_2d_predictive_completion(self) -> None:
        """
        Phase 2d: Predictive graph completion.
        
        Uses geometric heuristics to add probable, missing connections between
        nearby, unconnected elements.
        """
        if not self.active_logic_parameters.get('use_predictive_completion', True):
            return
        
        self._update_progress(58, "Phase 2d: Closing heuristic gaps...")
        
        original_connections = self._analysis_results.get("connections", [])
        
        from src.utils.graph_utils import predict_and_complete_graph
        
        distance_threshold = self.active_logic_parameters.get(
            'graph_completion_distance_threshold',
            50.0
        )
        
        all_connections_after_prediction = predict_and_complete_graph(
            elements=self._analysis_results.get("elements", []),
            connections=original_connections,
            logger_instance=logger,
            distance_threshold=distance_threshold
        )
        
        self._analysis_results["connections"] = all_connections_after_prediction
        
        added_count = len(all_connections_after_prediction) - len(original_connections)
        if added_count > 0:
            logger.info(f"{added_count} connection gaps predictively closed.")
        
        # TEST HARNESS: Save intermediate result after predictive completion
        if hasattr(self, 'current_output_dir'):
            from src.utils.test_harness import save_intermediate_result
            predictive_result = {
                "elements": self._analysis_results.get("elements", []),
                "connections": all_connections_after_prediction
            }
            save_intermediate_result("phase_2d_predictive", predictive_result, self.current_output_dir)
    
    def _run_phase_2e_polyline_refinement(self) -> None:
        """
        Phase 2e: Polyline refinement.
        
        For each found connection, creates a small, preprocessed image snippet
        and uses a specialized AI agent to find the exact line path (polyline).
        """
        if not self.current_image_path:
            return
        
        if not self.active_logic_parameters.get('use_polyline_refinement', True):
            return
        
        logger.info("--- Phase 2e: Starting high-precision polyline extraction ---")
        self._update_progress(59, "Phase 2e: Refining line paths...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.utils.image_utils import crop_image_for_correction, preprocess_image_for_line_detection
        from src.utils.graph_utils import match_polylines_to_connections
        from src.utils.type_utils import bbox_from_connection
        from pathlib import Path
        import os
        
        config = self.config_service.get_config()
        prompts = config.prompts
        
        # Handle PromptsConfig (Pydantic model) or dict
        if isinstance(prompts, dict):
            prompt = prompts.get('polyline_extraction_user_prompt')
        else:
            # PromptsConfig Pydantic model - use attribute access
            prompt = getattr(prompts, 'polyline_extraction_user_prompt', None)
        
        model_info = self.model_strategy.get('polyline_model') or self.model_strategy.get('detail_model')
        
        if not prompt or not model_info:
            logger.warning("Polyline prompt or model not found. Skipping polyline refinement.")
            return
        
        connections = self._analysis_results.get('connections', [])
        elements = self._analysis_results.get('elements', [])
        
        if not connections:
            logger.info("No connections to refine.")
            return
        
        elements_map = {el.get('id'): el for el in elements if el.get('id')}
        output_dir = Path(self._get_output_directory(self.current_image_path))
        temp_dir = output_dir / "temp_polylines"
        temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing {len(connections)} connections for polyline extraction...")
        
        def process_connection(conn):
            """Process a single connection to extract its polyline."""
            try:
                from_el = elements_map.get(conn.get('from_id'))
                to_el = elements_map.get(conn.get('to_id'))
                
                if not from_el or not to_el:
                    return None
                
                # Get connection bbox
                conn_bbox = bbox_from_connection(conn, elements_map)
                if not conn_bbox:
                    return None
                
                # Crop image snippet
                snippet_path = crop_image_for_correction(
                    str(self.current_image_path),
                    conn_bbox,
                    context_margin=0.1
                )
                
                if not snippet_path:
                    return None
                
                # Preprocess for line detection
                processed_path = temp_dir / f"processed_{Path(snippet_path).name}"
                preprocess_image_for_line_detection(snippet_path, str(processed_path))
                
                # Call LLM for polyline extraction
                response = self.llm_client.call_llm(
                    model_info,
                    system_prompt="",
                    user_prompt=prompt,
                    image_path=str(processed_path)
                )
                
                # Cleanup
                try:
                    os.remove(snippet_path)
                    if processed_path.exists():
                        os.remove(processed_path)
                except Exception:
                    pass
                
                if response and isinstance(response, dict):
                    polylines = response.get('polylines', [])
                    if polylines:
                        # Convert to normalized coordinates
                        img = Image.open(self.current_image_path)
                        img_width, img_height = img.size
                        
                        best_polyline = max(polylines, key=lambda p: len(p.get('polyline', [])) if isinstance(p, dict) else len(p))
                        polyline_data = best_polyline.get('polyline', []) if isinstance(best_polyline, dict) else best_polyline
                        
                        # Convert local snippet coordinates to global normalized coordinates
                        global_polyline = []
                        for point in polyline_data:
                            if len(point) >= 2:
                                # Convert from snippet local (0-1) to global image (0-1)
                                local_x, local_y = point[0], point[1]
                                global_x = (conn_bbox['x'] + local_x * conn_bbox['width']) / img_width
                                global_y = (conn_bbox['y'] + local_y * conn_bbox['height']) / img_height
                                global_polyline.append([global_x, global_y])
                        
                        return {
                            'connection_id': conn.get('id'),
                            'polyline': global_polyline
                        }
                
                return None
            except Exception as e:
                logger.error(f"Error processing connection {conn.get('id', 'unknown')}: {e}", exc_info=True)
                return None
        
        # Process connections in parallel with optimized worker count
        config_workers = self.active_logic_parameters.get('llm_executor_workers', 4)
        max_workers = min(config_workers, len(connections), 8)  # Cap at 8 for performance
        polyline_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_conn = {
                executor.submit(process_connection, conn): conn
                for conn in connections
            }
            
            for future in as_completed(future_to_conn):
                result = future.result()
                if result:
                    polyline_results.append(result)
        
        # Match polylines to connections
        all_polylines = [r['polyline'] for r in polyline_results]
        updated_connections = match_polylines_to_connections(elements, connections, all_polylines)
        
        self._analysis_results["connections"] = updated_connections
        
        logger.info(f"Polyline extraction complete: {len(polyline_results)} polylines extracted.")
        
        # TEST HARNESS: Save intermediate result after polyline refinement
        if hasattr(self, 'current_output_dir'):
            from src.utils.test_harness import save_intermediate_result
            polyline_result = {
                "elements": self._analysis_results.get("elements", []),
                "connections": updated_connections
            }
            save_intermediate_result("phase_2e_polyline", polyline_result, self.current_output_dir)
    
    def _run_phase_3_self_correction(
        self,
        image_path: str,
        output_dir: str,
        truth_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Phase 3: Self-correction loop.
        
        Iteratively validates and corrects analysis results based on feedback.
        """
        logger.info("--- Phase 3: Starting self-correction loop ---")
        
        if not self.active_logic_parameters.get('use_self_correction', True):
            logger.info("Self-correction disabled. Skipping Phase 3.")
            return self._analysis_results
        
        max_iterations = self.active_logic_parameters.get('max_self_correction_iterations', 3)
        target_score = self.active_logic_parameters.get('target_quality_score', 98.0)
        
        best_result: Dict[str, Any] = {
            "quality_score": -1.0,
            "final_ai_data": copy.deepcopy(self._analysis_results)
        }
        
        for i in range(max_iterations):
            iteration_name = f"Correction Iteration {i+1}/{max_iterations}"
            progress = 60 + int((i / max_iterations) * 25)
            self._update_progress(progress, iteration_name)
            
            # Validate and get quality score
            current_score, current_errors = self._run_phase_3_validation_and_critic(truth_data)
            
            if "score_history" not in self._analysis_results:
                self._analysis_results["score_history"] = []
            self._analysis_results["score_history"].append(current_score)
            
            # Update best result
            if current_score > best_result["quality_score"]:
                best_result = {
                    "quality_score": current_score,
                    "final_ai_data": copy.deepcopy(self._analysis_results)
                }
            
            # TEST HARNESS: Save intermediate result after each iteration
            if hasattr(self, 'current_output_dir'):
                from src.utils.test_harness import save_intermediate_result
                iteration_result = {
                    "iteration": i + 1,
                    "quality_score": current_score,
                    "elements": self._analysis_results.get("elements", []),
                    "connections": self._analysis_results.get("connections", []),
                    "errors": current_errors
                }
                save_intermediate_result(f"phase_3_selfcorrect_ITER_{i+1}", iteration_result, self.current_output_dir)
            
            # Early termination conditions
            if current_score >= target_score:
                logger.info(f"Target quality score reached ({current_score:.2f} >= {target_score:.2f}). Stopping corrections.")
                break
            
            if not current_errors or not any(current_errors.values()):
                logger.info("No errors found. Stopping corrections.")
                break
            
            # Re-analyze problematic segments
            corrected_results = self._re_analyze_with_feedback(
                image_path,
                output_dir,
                current_errors,
                i
            )
            
            if not corrected_results:
                logger.info("No corrected results. Stopping corrections.")
                break
            
            # Update analysis results
            self._analysis_results = corrected_results
        
        logger.info(f"Self-correction complete. Best score: {best_result['quality_score']:.2f}")
        return best_result
    
    def _run_phase_3_validation_and_critic(
        self,
        truth_data: Optional[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Phase 3: Comprehensive validation and critic.
        
        Validates analysis results and generates error feedback with confidence scores.
        
        Returns:
            Tuple of (quality_score, errors_dict)
        """
        from src.analyzer.evaluation.kpi_calculator import KPICalculator
        
        elements = self._analysis_results.get("elements", [])
        connections = self._analysis_results.get("connections", [])
        
        # Calculate comprehensive KPIs
        kpi_calculator = KPICalculator()
        kpis = kpi_calculator.calculate_comprehensive_kpis(self._analysis_results, truth_data)
        
        # Calculate quality score from KPIs
        if truth_data:
            # Use quality score from KPIs if available
            quality_score = kpis.get('quality_score', 0.0)
            
            # Fallback: If quality_score is 0.0 but we have elements, calculate internal score
            # This handles cases where truth data exists but IoU matching failed
            if quality_score == 0.0 and elements:
                logger.warning("Quality score from KPIs is 0.0 despite elements found. Using internal fallback calculation.")
                # Calculate internal quality score as fallback
                quality_score = 50.0  # Base score
                avg_confidence = sum(el.get('confidence', 0.5) for el in elements) / len(elements) if elements else 0.0
                quality_score += min(len(elements) * 1.5, 25.0)  # Max 25 points for elements
                quality_score += avg_confidence * 15.0  # Up to 15 points for confidence
                
                if connections:
                    avg_conn_confidence = sum(conn.get('confidence', 0.5) for conn in connections) / len(connections) if connections else 0.0
                    quality_score += min(len(connections) * 1.0, 15.0)  # Max 15 points for connections
                    quality_score += avg_conn_confidence * 10.0  # Up to 10 points for confidence
                
                graph_density = kpis.get('graph_density', 0.0)
                quality_score += min(graph_density * 50.0, 10.0)  # Up to 10 points for graph density
        else:
            # Calculate internal quality score
            quality_score = 50.0  # Base score
            
            if elements:
                # Score based on element count and confidence
                avg_confidence = sum(el.get('confidence', 0.5) for el in elements) / len(elements) if elements else 0.0
                quality_score += min(len(elements) * 1.5, 25.0)  # Max 25 points for elements
                quality_score += avg_confidence * 15.0  # Up to 15 points for confidence
            
            if connections:
                # Score based on connection count and confidence
                avg_conn_confidence = sum(conn.get('confidence', 0.5) for conn in connections) / len(connections) if connections else 0.0
                quality_score += min(len(connections) * 1.0, 15.0)  # Max 15 points for connections
                quality_score += avg_conn_confidence * 10.0  # Up to 10 points for confidence
            
            # Graph structure quality
            graph_density = kpis.get('graph_density', 0.0)
            quality_score += min(graph_density * 50.0, 10.0)  # Up to 10 points for graph density
        
        # Generate errors dict (extract actual element lists for re-analysis)
        # Get unmatched/hallucinated elements from analysis result if available
        analysis_elements = self._analysis_results.get('elements', elements)
        truth_elements = truth_data.get('elements', []) if truth_data else []
        
        # Find actually missed and hallucinated elements (not just counts)
        matched_element_ids = set()
        if truth_data:
            # Build set of matched element IDs from KPIs (if available)
            # For now, use simple approach: unmatched analysis elements are hallucinated
            truth_ids = {el.get('id') for el in truth_elements if el.get('id')}
            analysis_ids = {el.get('id') for el in analysis_elements if el.get('id')}
            matched_ids = analysis_ids & truth_ids  # Simplified: assume IDs match if same
        
        # Extract hallucinated elements (elements in analysis but not in truth)
        hallucinated_els = []
        if truth_data and truth_elements:
            truth_ids_set = {el.get('id') for el in truth_elements if el.get('id')}
            for el in analysis_elements:
                el_id = el.get('id')
                if el_id and el_id not in truth_ids_set:
                    hallucinated_els.append(el)
        else:
            # If no truth data, use low confidence elements as "hallucinated"
            hallucinated_els = [el for el in analysis_elements if el.get('confidence', 1.0) < 0.5]
        
        # Missed elements: in truth but not in analysis (simplified)
        missed_els = []
        if truth_data and truth_elements:
            analysis_ids_set = {el.get('id') for el in analysis_elements if el.get('id')}
            for el in truth_elements:
                el_id = el.get('id')
                if el_id and el_id not in analysis_ids_set:
                    missed_els.append(el)
        
        # Add metacritic discrepancies to errors if available
        metacritic_discrepancies = self._analysis_results.get('metacritic_discrepancies', [])
        
        errors: Dict[str, Any] = {
            'missed_elements': missed_els,  # List of actual elements
            'hallucinated_elements': hallucinated_els,  # List of actual elements
            'missed_connections': kpis.get('missed_connections', 0),
            'hallucinated_connections': kpis.get('hallucinated_connections', 0),
            'low_confidence_elements': [
                el.get('id') for el in elements
                if el.get('confidence', 1.0) < 0.5
            ],
            'isolated_elements': kpis.get('isolated_elements', 0),
            'metacritic_discrepancies': metacritic_discrepancies  # Add metacritic issues
        }
        
        # Store KPIs for later use
        self._analysis_results['final_kpis'] = kpis
        
        return min(max(quality_score, 0.0), 100.0), errors
    
    def _re_analyze_with_feedback(
        self,
        image_path: str,
        output_dir: str,
        errors: Dict[str, Any],
        iteration: int
    ) -> Optional[Dict[str, Any]]:
        """
        Re-analyze problematic segments with error feedback.
        
        Args:
            image_path: Path to image
            output_dir: Output directory
            errors: Error dictionary containing missed/hallucinated elements
            iteration: Current iteration number
            
        Returns:
            Updated analysis results or None
        """
        logger.info(f"Re-analyzing with feedback (iteration {iteration + 1})...")
        
        try:
            from src.analyzer.analysis import SwarmAnalyzer
            from concurrent.futures import ThreadPoolExecutor
            
            # Extract error feedback
            missed_elements = errors.get('missed_elements', [])
            hallucinated_elements = errors.get('hallucinated_elements', [])
            
            if not missed_elements and not hallucinated_elements:
                logger.info("No errors to fix, skipping re-analysis")
                return self._analysis_results
            
            logger.info(f"Re-analysis: {len(missed_elements)} missed, {len(hallucinated_elements)} hallucinated elements")
            
            # Build error feedback message for LLM
            error_feedback = []
            if missed_elements:
                missed_types = [el.get('type', 'Unknown') for el in missed_elements[:5]]  # Limit to 5
                error_feedback.append(f"MISSED ELEMENTS: {', '.join(missed_types)}")
            if hallucinated_elements:
                hall_types = [el.get('type', 'Unknown') for el in hallucinated_elements[:5]]  # Limit to 5
                error_feedback.append(f"HALLUCINATED ELEMENTS (remove): {', '.join(hall_types)}")
            
            # Add metacritic discrepancies if available
            metacritic_discrepancies = self._analysis_results.get('metacritic_discrepancies', [])
            if metacritic_discrepancies:
                metacritic_feedback = []
                for disc in metacritic_discrepancies[:5]:  # Limit to 5
                    disc_id = disc.get('id', 'UNKNOWN')
                    disc_desc = disc.get('description', '')[:100]  # Limit length
                    metacritic_feedback.append(f"METACRITIC_ISSUE_{disc_id}: {disc_desc}")
                if metacritic_feedback:
                    error_feedback.append(f"METACRITIC FEEDBACK:\n" + "\n".join(metacritic_feedback))
            
            error_feedback_str = "\n".join(error_feedback) if error_feedback else ""
            
            # Create analyzer with error feedback
            swarm_analyzer = SwarmAnalyzer(
                self.llm_client,
                self.knowledge_manager,
                self.config_service,
                self.model_strategy,
                self.active_logic_parameters
            )
            
            # Set error feedback in analyzer
            swarm_analyzer.error_feedback = error_feedback_str
            swarm_analyzer.legend_context = {
                'symbol_map': self._global_knowledge_repo.get('symbol_map', {}),
                'line_map': self._global_knowledge_repo.get('line_map', {})
            }
            
            # Re-analyze using swarm (tile-based for precision)
            logger.info("Re-analyzing problematic areas with error feedback...")
            output_path = Path(output_dir)
            swarm_result = swarm_analyzer.analyze(
                image_path,
                output_path,
                self._excluded_zones
            )
            
            if swarm_result:
                # Merge with existing results (prioritize new results)
                current_elements = self._analysis_results.get('elements', [])
                current_connections = self._analysis_results.get('connections', [])
                
                new_elements = swarm_result.get('elements', [])
                new_connections = swarm_result.get('connections', [])
                
                # Merge: Add new elements, remove hallucinated ones
                element_ids_to_remove = {el.get('id') for el in hallucinated_elements}
                filtered_current = [el for el in current_elements if el.get('id') not in element_ids_to_remove]
                
                # Deduplicate by ID
                existing_ids = {el.get('id') for el in filtered_current}
                new_unique = [el for el in new_elements if el.get('id') not in existing_ids]
                
                merged_elements = filtered_current + new_unique
                merged_connections = current_connections + new_connections
                
                self._analysis_results['elements'] = merged_elements
                self._analysis_results['connections'] = merged_connections
                
                logger.info(f"Re-analysis complete: {len(merged_elements)} elements, {len(merged_connections)} connections")
                
                # Learn from corrections immediately
                try:
                    self.active_learner.learn_from_analysis_result(
                        analysis_result=self._analysis_results,
                        truth_data=None,  # Can be enhanced with truth data
                        quality_score=self._calculate_quality_score(self._analysis_results)
                    )
                    logger.info("Live learning: Learned from re-analysis corrections")
                except Exception as e:
                    logger.warning(f"Error in live learning: {e}")
            
            return self._analysis_results
            
        except Exception as e:
            logger.error(f"Error in re-analysis with feedback: {e}", exc_info=True)
            return self._analysis_results
    
    def _run_phase_4_post_processing(
        self,
        best_result: Dict[str, Any],
        image_path: str,
        output_dir: str,
        truth_data: Optional[Dict[str, Any]]
    ) -> AnalysisResult:
        """
        Phase 4: Post-processing (KPIs, CGM, artifacts).
        
        Generates KPIs, CGM abstraction, and saves artifacts.
        """
        logger.info("--- Phase 4: Starting post-processing ---")
        
        final_ai_data = best_result.get("final_ai_data", {})
        
        if not isinstance(final_ai_data, dict) or not final_ai_data.get("elements"):
            logger.error("Post-processing aborted: No valid final data found.")
            return self._create_error_result("Post-processing failed: No valid data")
        
        # Calculate KPIs
        kpis = self._calculate_kpis(final_ai_data, truth_data)
        
        # Generate CGM data (simplified for now)
        cgm_data = self._generate_cgm_data(final_ai_data)
        
        # Save artifacts
        self._save_artifacts(output_dir, image_path, final_ai_data, kpis, cgm_data)
        
        # Generate visualizations
        score_history = best_result.get("final_ai_data", {}).get("score_history", [])
        if output_dir:
            self._generate_visualizations(output_dir, image_path, final_ai_data, kpis, score_history)
        
        # Convert elements and connections to Pydantic models if needed
        elements_data = final_ai_data.get("elements", [])
        connections_data = final_ai_data.get("connections", [])
        
        # Post-Processing: Filter low-confidence elements and hallucinations
        confidence_threshold = self.active_logic_parameters.get('confidence_threshold', 0.7)
        logger.info(f"Post-processing: Filtering elements with confidence < {confidence_threshold}")
        
        filtered_elements = []
        filtered_elements_ids = set()
        removed_count = 0
        
        for el in elements_data:
            el_dict = el if isinstance(el, dict) else el.model_dump() if hasattr(el, 'model_dump') else el.__dict__ if hasattr(el, '__dict__') else {}
            confidence = el_dict.get('confidence', 0.5)
            
            if confidence >= confidence_threshold:
                filtered_elements.append(el)
                el_id = el_dict.get('id')
                if el_id:
                    filtered_elements_ids.add(el_id)
            else:
                removed_count += 1
                logger.debug(f"Removed low-confidence element: {el_dict.get('id', 'unknown')} (confidence: {confidence:.2f} < {confidence_threshold})")
        
        if removed_count > 0:
            logger.info(f"Post-processing: Removed {removed_count} low-confidence elements (confidence < {confidence_threshold})")
        
        # Filter connections: Only keep connections between filtered elements
        filtered_connections = []
        removed_conn_count = 0
        
        for conn in connections_data:
            conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ if hasattr(conn, '__dict__') else {}
            from_id = conn_dict.get('from_id')
            to_id = conn_dict.get('to_id')
            conn_confidence = conn_dict.get('confidence', 0.5)
            
            # Keep connection if both elements exist AND connection has sufficient confidence
            if from_id in filtered_elements_ids and to_id in filtered_elements_ids and conn_confidence >= confidence_threshold:
                filtered_connections.append(conn)
            else:
                removed_conn_count += 1
                logger.debug(f"Removed connection: {from_id} -> {to_id} (missing elements or low confidence: {conn_confidence:.2f})")
        
        if removed_conn_count > 0:
            logger.info(f"Post-processing: Removed {removed_conn_count} connections (missing elements or confidence < {confidence_threshold})")
        
        # Update data with filtered results
        elements_data = filtered_elements
        connections_data = filtered_connections
        logger.info(f"Post-processing complete: {len(elements_data)} elements, {len(connections_data)} connections (after filtering)")
        
        # Ensure elements are Element objects (AnalysisResult expects List[Element])
        elements_models = []
        for el in elements_data:
            if isinstance(el, Element):
                # Validate existing Element
                if el.bbox.width > 0 and el.bbox.height > 0:
                    elements_models.append(el)
                else:
                    logger.warning(f"Skipping element {el.id}: Invalid bbox (width={el.bbox.width}, height={el.bbox.height})")
                    continue
            elif isinstance(el, dict):
                try:
                    # CRITICAL FIX 2: VALIDATE instead of REPAIR - reject invalid bboxes
                    if 'bbox' not in el or not isinstance(el['bbox'], dict):
                        logger.warning(f"Skipping element {el.get('id', 'unknown')}: Missing or invalid bbox structure")
                        continue
                    
                    bbox = el['bbox']
                    element_id = el.get('id', 'unknown')
                    
                    # STRICT VALIDATION: BBox must have valid, positive dimensions
                    from src.utils.type_utils import is_valid_bbox
                    if not is_valid_bbox(bbox) or bbox.get('width', 0) <= 0 or bbox.get('height', 0) <= 0:
                        logger.warning(
                            f"REJECTING element {element_id}: Invalid bbox dimensions "
                            f"(width={bbox.get('width')}, height={bbox.get('height')}). "
                            f"Element is likely a hallucination."
                        )
                        continue  # Element is rejected, not repaired
                    
                    # BBox must be within bounds (0.0 - 1.0)
                    bbox['x'] = max(0.0, min(1.0, bbox.get('x', 0.0)))
                    bbox['y'] = max(0.0, min(1.0, bbox.get('y', 0.0)))
                    bbox['width'] = max(0.001, min(1.0 - bbox['x'], bbox.get('width', 0.001)))
                    bbox['height'] = max(0.001, min(1.0 - bbox['y'], bbox.get('height', 0.001)))
                    
                    # Create BBox Pydantic model
                    from src.analyzer.models.elements import BBox
                    el['bbox'] = BBox(**bbox)
                    
                    # Ensure label is not None (required by Pydantic)
                    if 'label' not in el or el.get('label') is None:
                        el['label'] = el.get('type', 'Unknown') or 'Unknown'
                        logger.debug(f"Element {el.get('id', 'unknown')}: Set label to '{el['label']}' (was None)")
                    
                    # Final confidence check before adding (defensive programming)
                    confidence = el.get('confidence', 0.5)
                    if confidence >= confidence_threshold:
                        elements_models.append(Element(**el))
                    else:
                        logger.debug(f"Skipping element {el.get('id', 'unknown')} in final conversion: confidence {confidence:.2f} < {confidence_threshold}")
                except Exception as e:
                    logger.warning(f"Could not convert element dict to Element model: {e}")
                    continue
            else:
                logger.warning(f"Unexpected element type: {type(el)}")
        
        # Ensure connections are Connection objects (AnalysisResult expects List[Connection])
        connections_models = []
        for conn in connections_data:
            if isinstance(conn, Connection):
                connections_models.append(conn)
            elif isinstance(conn, dict):
                try:
                    connections_models.append(Connection(**conn))
                except Exception as e:
                    logger.warning(f"Could not convert connection dict to Connection model: {e}")
                    continue
            else:
                logger.warning(f"Unexpected connection type: {type(conn)}")
        
        # Prepare complete legend data for output
        complete_legend_data = self._analysis_results.get('legend_data', {})
        if not complete_legend_data:
            # Fallback: construct from global knowledge repo
            complete_legend_data = {
                'symbol_map': self._global_knowledge_repo.get('symbol_map', {}),
                'line_map': self._global_knowledge_repo.get('line_map', {}),
                'legend_bbox': None
            }
        
        # Convert to AnalysisResult
        result = AnalysisResult(
            image_name=os.path.basename(image_path),
            elements=elements_models,
            connections=connections_models,
            quality_score=best_result.get("quality_score", 0.0),
            score_history=best_result.get("final_ai_data", {}).get("score_history", []),
            metadata=final_ai_data.get("metadata"),
            legend_data=complete_legend_data,
            cgm_data=cgm_data,
            kpis=kpis
        )
        
        logger.info(f"Post-processing complete. Quality score: {best_result.get('quality_score', 0.0):.2f}")
        
        # Active learning: Learn from analysis result
        try:
            quality_score = best_result.get('quality_score', 0.0)
            learning_report = self.active_learner.learn_from_analysis_result(
                analysis_result=final_ai_data,
                truth_data=truth_data,
                quality_score=quality_score
            )
            logger.info(f"Active learning complete: {learning_report.get('patterns_learned', 0)} patterns, "
                       f"{learning_report.get('corrections_learned', 0)} corrections learned")
        except Exception as e:
            logger.error(f"Error in active learning: {e}", exc_info=True)
        
        return result
    
    def _detect_pid_type(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Detect P&ID type and complexity to adapt analysis strategy.
        
        Args:
            image_path: Path to P&ID image
            
        Returns:
            Metadata dictionary with type and complexity
        """
        try:
            from PIL import Image
            
            with Image.open(image_path) as img:
                width, height = img.size
                total_pixels = width * height
            
            # Simple heuristics for P&ID type detection
            complexity = 'medium'
            pid_type = 'generic'
            
            # Detect complexity based on image size
            if total_pixels > 10_000_000:  # >10MP
                complexity = 'complex'
            elif total_pixels < 2_000_000:  # <2MP
                complexity = 'simple'
            
            # Detect type based on metadata if available
            metadata = self.state.metadata
            if metadata:
                if 'type' in metadata:
                    pid_type = metadata.get('type', 'generic')
            
            return {
                'type': pid_type,
                'complexity': complexity,
                'width': width,
                'height': height,
                'total_pixels': total_pixels
            }
        except Exception as e:
            logger.error(f"Error detecting P&ID type: {e}", exc_info=True)
            return None
    
    def _calculate_kpis(
        self,
        final_data: Dict[str, Any],
        truth_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive Key Performance Indicators.
        
        Args:
            final_data: Final analysis data
            truth_data: Optional truth data for comparison
            
        Returns:
            Comprehensive KPI dictionary
        """
        from src.analyzer.evaluation.kpi_calculator import KPICalculator
        
        calculator = KPICalculator()
        kpis = calculator.calculate_comprehensive_kpis(final_data, truth_data)
        
        return kpis
    
    def _generate_cgm_data(self, final_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive CGM (Component Grouping Model) abstraction.
        
        Uses advanced graph theory to represent:
        - Component positions (with full BBox coordinates)
        - Split and merge points (with calculated positions)
        - Pipeline flows (complete flow paths with positions)
        - Network graph structure (using NetworkX)
        
        Args:
            final_data: Final analysis data
            
        Returns:
            Comprehensive CGM data dictionary with:
            - Python dataclass format code
            - JSON format with full coordinates
            - Split/merge points with positions
            - System flows with positions
        """
        logger.info("Generating comprehensive CGM data with graph theory...")
        
        try:
            from src.analyzer.output.cgm_generator import CGMGenerator
        except ImportError:
            logger.warning("CGMGenerator not available, returning basic CGM data")
            return {
                "python_code": "",
                "components": [],
                "connectors": [],
                "split_merge_points": [],
                "system_flows": [],
                "component_groups": {},
                "metadata": {}
            }
        
        elements_raw = final_data.get("elements", [])
        connections_raw = final_data.get("connections", [])
        
        # Helper function to recursively convert Pydantic models to dicts
        def to_dict_recursive(obj):
            """Recursively convert Pydantic models to dicts, handling nested models."""
            if isinstance(obj, dict):
                return {k: to_dict_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_dict_recursive(item) for item in obj]
            elif hasattr(obj, 'model_dump'):
                # Pydantic v2
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                # Pydantic v1
                return obj.dict()
            elif hasattr(obj, '__dict__'):
                # Generic object with __dict__
                return {k: to_dict_recursive(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            else:
                return obj
        
        # Convert Pydantic models to dictionaries if needed
        elements = []
        for el in elements_raw:
            try:
                if hasattr(el, 'model_dump'):
                    # Pydantic v2 model - recursively convert
                    el_dict = to_dict_recursive(el)
                    elements.append(el_dict)
                elif hasattr(el, 'dict'):
                    # Pydantic v1 model - recursively convert
                    el_dict = to_dict_recursive(el)
                    elements.append(el_dict)
                elif isinstance(el, dict):
                    # Already a dict - recursively convert nested objects
                    elements.append(to_dict_recursive(el))
                else:
                    # Try to convert to dict
                    el_dict = to_dict_recursive(el)
                    elements.append(el_dict)
            except Exception as e:
                logger.warning(f"Could not convert element to dict: {el} - {e}")
                continue
        
        connections = []
        for conn in connections_raw:
            try:
                if hasattr(conn, 'model_dump'):
                    # Pydantic v2 model - recursively convert
                    conn_dict = to_dict_recursive(conn)
                    connections.append(conn_dict)
                elif hasattr(conn, 'dict'):
                    # Pydantic v1 model - recursively convert
                    conn_dict = to_dict_recursive(conn)
                    connections.append(conn_dict)
                elif isinstance(conn, dict):
                    # Already a dict - recursively convert nested objects
                    connections.append(to_dict_recursive(conn))
                else:
                    # Try to convert to dict
                    conn_dict = to_dict_recursive(conn)
                    connections.append(conn_dict)
            except Exception as e:
                logger.warning(f"Could not convert connection to dict: {conn} - {e}")
                continue
        
        # Use CGM Generator with graph theory
        try:
            cgm_generator = CGMGenerator(elements, connections)
            
            # Generate both formats
            cgm_python_code = cgm_generator.generate_cgm_python_code()
            cgm_json_data = cgm_generator.generate_cgm_json()
        except Exception as e:
            logger.error(f"Error generating CGM data: {e}", exc_info=True)
            cgm_python_code = ""
            cgm_json_data = {
                "components": [],
                "connectors": [],
                "split_merge_points": [],
                "system_flows": [],
                "metadata": {}
            }
        
        # Get main components from config
        main_components = self.active_logic_parameters.get(
            'cgm_main_components',
            ["Boiler", "Pump", "Heat Exchanger", "Buffer Storage", "Thermal Consumer"]
        )
        
        cgm_data = {
            "python_code": cgm_python_code,  # Python dataclass format
            "components": cgm_json_data.get("components", []),
            "connectors": cgm_json_data.get("connectors", []),
            "split_merge_points": cgm_json_data.get("split_merge_points", []),
            "system_flows": cgm_json_data.get("system_flows", []),
            "component_groups": {},
            "metadata": cgm_json_data.get("metadata", {})
        }
        
        # Group elements by type (elements are now guaranteed to be dicts)
        elements_by_type = {}
        for el in elements:
            el_type = el.get("type", "unknown") if isinstance(el, dict) else getattr(el, 'type', 'unknown')
            el_id = el.get("id") if isinstance(el, dict) else getattr(el, 'id', None)
            if el_id:
                if el_type not in elements_by_type:
                    elements_by_type[el_type] = []
                elements_by_type[el_type].append(el)
        
        # Create component groups (for main components)
        for el_type, elements_list in elements_by_type.items():
            if el_type in main_components or len(elements_list) > 1:
                # Main component group
                element_ids = []
                bboxes = []
                confidences = []
                
                for el in elements_list:
                    el_id = el.get("id") if isinstance(el, dict) else getattr(el, 'id', None)
                    bbox = el.get("bbox") if isinstance(el, dict) else getattr(el, 'bbox', None)
                    confidence = el.get("confidence", 0.5) if isinstance(el, dict) else getattr(el, 'confidence', 0.5)
                    
                    if el_id:
                        element_ids.append(el_id)
                    if bbox:
                        bboxes.append(bbox.model_dump() if hasattr(bbox, 'model_dump') else bbox)
                    confidences.append(confidence)
                
                group = {
                    "type": el_type,
                    "count": len(elements_list),
                    "element_ids": element_ids,
                    "bboxes": bboxes,
                    "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0
                }
                cgm_data["components"].append(group)
                cgm_data["component_groups"][el_type] = group
        
        # Create connectors (connections between main components)
        main_component_elements = {}
        for el in elements:
            el_type = el.get("type", "unknown") if isinstance(el, dict) else getattr(el, 'type', 'unknown')
            el_id = el.get("id") if isinstance(el, dict) else getattr(el, 'id', None)
            if el_id and el_type in main_components:
                main_component_elements[el_id] = el
        
        connectors = []
        for conn in connections:
            # Handle both dict and Pydantic model
            if isinstance(conn, dict):
                from_id = conn.get("from_id")
                to_id = conn.get("to_id")
                conn_id = conn.get("id")
                confidence = conn.get("confidence", 0.5)
                kind = conn.get("kind", "process")
            else:
                from_id = getattr(conn, 'from_id', None)
                to_id = getattr(conn, 'to_id', None)
                conn_id = getattr(conn, 'id', None)
                confidence = getattr(conn, 'confidence', 0.5)
                kind = getattr(conn, 'kind', 'process')
            
            if from_id and to_id and from_id in main_component_elements and to_id in main_component_elements:
                from_el = main_component_elements[from_id]
                to_el = main_component_elements[to_id]
                
                from_type = from_el.get("type") if isinstance(from_el, dict) else getattr(from_el, 'type', 'unknown')
                to_type = to_el.get("type") if isinstance(to_el, dict) else getattr(to_el, 'type', 'unknown')
                
                connector = {
                    "id": conn_id,
                    "from_component": from_type,
                    "to_component": to_type,
                    "from_id": from_id,
                    "to_id": to_id,
                    "confidence": confidence,
                    "kind": kind if isinstance(kind, str) else getattr(kind, 'value', 'process')
                }
                connectors.append(connector)
        
        cgm_data["connectors"] = connectors
        
        # Identify system flows (sequences of connected main components)
        from src.utils.graph_utils import dedupe_connections
        main_connections = [
            conn for conn in connections
            if conn.get("from_id") in main_component_elements and
               conn.get("to_id") in main_component_elements
        ]
        
        # Build flow sequences
        flows = []
        processed_connections = set()
        
        for conn in main_connections:
            conn_key = (conn.get("from_id"), conn.get("to_id"))
            if conn_key in processed_connections:
                continue
            
            # Build flow path
            flow_path = [conn.get("from_id"), conn.get("to_id")]
            current_id = conn.get("to_id")
            processed_connections.add(conn_key)
            
            # Extend flow path
            while True:
                next_conn = next((
                    c for c in main_connections
                    if c.get("from_id") == current_id and
                       (c.get("from_id"), c.get("to_id")) not in processed_connections
                ), None)
                
                if next_conn:
                    flow_path.append(next_conn.get("to_id"))
                    processed_connections.add((next_conn.get("from_id"), next_conn.get("to_id")))
                    current_id = next_conn.get("to_id")
                else:
                    break
            
            if len(flow_path) > 1:
                flow_components = [
                    main_component_elements[eid].get("type") for eid in flow_path
                    if eid in main_component_elements
                ]
                flows.append({
                    "path": flow_path,
                    "components": flow_components,
                    "length": len(flow_path)
                })
        
        cgm_data["system_flows"] = flows
        
        logger.info(f"CGM data generated: {len(cgm_data['components'])} components, {len(cgm_data['connectors'])} connectors, {len(flows)} flows")
        return cgm_data
    
    def _save_artifacts(
        self,
        output_dir: str,
        image_path: str,
        final_data: Dict[str, Any],
        kpis: Dict[str, Any],
        cgm_data: Dict[str, Any]
    ) -> None:
        """
        Save analysis artifacts to output directory.
        
        Args:
            output_dir: Output directory path
            image_path: Path to input image
            final_data: Final analysis data
            kpis: KPIs dictionary
            cgm_data: CGM data dictionary
        """
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        # Get legend info for structured output
        legend_info = {
            'symbol_map': final_data.get('legend_data', {}).get('symbol_map', {}),
            'line_map': final_data.get('legend_data', {}).get('line_map', {}),
            'metadata': final_data.get('metadata', {})
        }
        
        # Save JSON results with legend info
        results_data = final_data.copy()
        results_data['legend_info'] = legend_info
        results_path = output_path / f"{base_name}_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to: {results_path}")
        
        # Save separate legend info file for easy access
        legend_info_path = output_path / f"{base_name}_legend_info.json"
        with open(legend_info_path, 'w', encoding='utf-8') as f:
            json.dump(legend_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved legend info to: {legend_info_path}")
        
        # Save KPIs
        kpis_path = output_path / f"{base_name}_kpis.json"
        with open(kpis_path, 'w', encoding='utf-8') as f:
            json.dump(kpis, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved KPIs to: {kpis_path}")
        
        # Save CGM data (JSON format)
        cgm_path = output_path / f"{base_name}_cgm_data.json"
        cgm_json_save = {
            'components': cgm_data.get('components', []),
            'connectors': cgm_data.get('connectors', []),
            'split_merge_points': cgm_data.get('split_merge_points', []),
            'system_flows': cgm_data.get('system_flows', []),
            'metadata': cgm_data.get('metadata', {})
        }
        with open(cgm_path, 'w', encoding='utf-8') as f:
            json.dump(cgm_json_save, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved CGM JSON data to: {cgm_path}")
        
        # Save CGM Python code (dataclass format)
        if cgm_data.get('python_code'):
            cgm_py_path = output_path / f"{base_name}_cgm_network_generated.py"
            with open(cgm_py_path, 'w', encoding='utf-8') as f:
                f.write(cgm_data['python_code'])
            logger.info(f"Saved CGM Python code to: {cgm_py_path}")
        
        # Generate HTML report for professional presentation
        self._generate_html_report(output_dir, image_path, final_data, legend_info, kpis, cgm_data)
    
    def _generate_visualizations(
        self,
        output_dir: str,
        image_path: str,
        final_data: Dict[str, Any],
        kpis: Dict[str, Any],
        score_history: List[float]
    ) -> None:
        """
        Generate visualizations (heatmaps, debug maps, score curves).
        
        Args:
            output_dir: Output directory path
            image_path: Path to input image
            final_data: Final analysis data
            kpis: KPIs dictionary
            score_history: Score history list
        """
        try:
            from PIL import Image as PILImage
            from src.analyzer.visualization.visualizer import Visualizer
            
            # Get image dimensions
            with PILImage.open(image_path) as img:
                img_width, img_height = img.size
            
            visualizer = Visualizer(img_width, img_height)
            output_path = Path(output_dir)
            base_name = Path(image_path).stem
            
            # 1. Uncertainty heatmap
            uncertain_zones = final_data.get("uncertain_zones", [])
            if uncertain_zones:
                heatmap_path = output_path / f"{base_name}_uncertainty_heatmap.png"
                visualizer.draw_uncertainty_heatmap(
                    image_path=image_path,
                    uncertain_zones=uncertain_zones,
                    output_path=str(heatmap_path)
                )
            
            # 2. Debug map
            elements = final_data.get("elements", [])
            connections = final_data.get("connections", [])
            if elements or connections:
                debug_map_path = output_path / f"{base_name}_debug_map.png"
                visualizer.draw_debug_map(
                    image_path=image_path,
                    elements=elements,
                    connections=connections,
                    output_path=str(debug_map_path)
                )
            
            # 3. Confidence map
            if elements:
                confidence_map_path = output_path / f"{base_name}_confidence_map.png"
                visualizer.draw_confidence_map(
                    image_path=image_path,
                    elements=elements,
                    output_path=str(confidence_map_path)
                )
            
            # 4. Score curve
            if score_history:
                score_curve_path = output_path / f"{base_name}_score_curve.png"
                visualizer.plot_score_curve(
                    score_history=score_history,
                    output_path=str(score_curve_path)
                )
            
            # 5. KPI dashboard
            if kpis:
                kpi_dashboard_path = output_path / f"{base_name}_kpi_dashboard.png"
                visualizer.plot_kpi_dashboard(
                    kpis=kpis,
                    output_path=str(kpi_dashboard_path)
                )
            
            logger.info("Visualizations generated successfully")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
    
    def _generate_html_report(
        self,
        output_dir: str,
        image_path: str,
        final_data: Dict[str, Any],
        legend_info: Dict[str, Any],
        kpis: Dict[str, Any],
        cgm_data: Dict[str, Any]
    ) -> None:
        """
        Generate professional HTML report for companies.
        
        Args:
            output_dir: Output directory path
            image_path: Path to input image
            final_data: Final analysis data
            legend_info: Legend information (symbol_map, line_map, metadata)
            kpis: KPIs dictionary
            cgm_data: CGM data dictionary
        """
        try:
            from pathlib import Path
            from datetime import datetime
            import base64
            from PIL import Image as PILImage
            
            output_path = Path(output_dir)
            base_name = Path(image_path).stem
            
            # Load original image for embedding
            try:
                with PILImage.open(image_path) as img:
                    img_format = img.format or 'PNG'
                    # Convert to base64 for embedding
                    import io
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format=img_format)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    img_data_url = f"data:image/{img_format.lower()};base64,{img_base64}"
            except Exception as e:
                logger.warning(f"Could not load image for HTML report: {e}")
                img_data_url = ""
            
            # Get visualization paths
            viz_base = output_path / f"{base_name}"
            debug_map_path = viz_base.parent / f"{base_name}_debug_map.png"
            confidence_map_path = viz_base.parent / f"{base_name}_confidence_map.png"
            kpi_dashboard_path = viz_base.parent / f"{base_name}_kpi_dashboard.png"
            
            # Helper to embed images
            def embed_image(image_path: Path) -> str:
                if image_path.exists():
                    try:
                        with open(image_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                            return f"data:image/png;base64,{img_data}"
                    except:
                        return ""
                return ""
            
            debug_map_data = embed_image(debug_map_path)
            confidence_map_data = embed_image(confidence_map_path)
            kpi_dashboard_data = embed_image(kpi_dashboard_path)
            
            # Extract data
            elements = final_data.get('elements', [])
            connections = final_data.get('connections', [])
            symbol_map = legend_info.get('symbol_map', {})
            line_map = legend_info.get('line_map', {})
            metadata = legend_info.get('metadata', {})
            
            # Generate HTML
            html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P&ID Analyse Report - {base_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; margin-bottom: 15px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        h3 {{ color: #555; margin-top: 20px; margin-bottom: 10px; }}
        .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .metadata-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
        .metadata-item {{ padding: 8px; background: white; border-radius: 3px; }}
        .metadata-label {{ font-weight: bold; color: #7f8c8d; }}
        .legend-section {{ margin: 20px 0; }}
        .legend-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .legend-table th, .legend-table td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        .legend-table th {{ background: #3498db; color: white; }}
        .legend-table tr:nth-child(even) {{ background: #f9f9f9; }}
        .elements-section, .connections-section {{ margin: 20px 0; }}
        .data-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }}
        .data-table th, .data-table td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
        .data-table th {{ background: #2c3e50; color: white; position: sticky; top: 0; }}
        .data-table tr:hover {{ background: #f5f5f5; }}
        .visualizations {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .viz-item {{ text-align: center; }}
        .viz-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .kpi-summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
        .kpi-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .kpi-value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
        .kpi-label {{ font-size: 14px; opacity: 0.9; }}
        .timestamp {{ color: #7f8c8d; font-size: 12px; margin-bottom: 20px; }}
        .color-swatch {{ display: inline-block; width: 20px; height: 20px; border-radius: 3px; margin-right: 5px; vertical-align: middle; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>P&ID Analyse Report</h1>
        <div class="timestamp">Generiert am: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</div>
        
        <h2>1. Metadaten</h2>
        <div class="metadata">
            <div class="metadata-grid">
                <div class="metadata-item"><span class="metadata-label">Projekt:</span> {metadata.get('project', 'N/A')}</div>
                <div class="metadata-item"><span class="metadata-label">Titel:</span> {metadata.get('title', 'N/A')}</div>
                <div class="metadata-item"><span class="metadata-label">Version:</span> {metadata.get('version', 'N/A')}</div>
                <div class="metadata-item"><span class="metadata-label">Datum:</span> {metadata.get('date', 'N/A')}</div>
            </div>
        </div>
        
        <h2>2. Legende</h2>
        <div class="legend-section">
            <h3>Symbol-Legende</h3>
            <table class="legend-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Typ</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            # Add symbol map rows
            if symbol_map:
                for symbol_key, symbol_type in symbol_map.items():
                    html_content += f"""
                    <tr>
                        <td><strong>{symbol_key}</strong></td>
                        <td>{symbol_type}</td>
                    </tr>
"""
            else:
                html_content += '<tr><td colspan="2">Keine Symbole in Legende gefunden</td></tr>'
            
            html_content += """
                </tbody>
            </table>
            
            <h3>Rohrleitungs-Legende</h3>
            <table class="legend-table">
                <thead>
                    <tr>
                        <th>Rohrleitung</th>
                        <th>Farbe</th>
                        <th>Stil</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            # Add line map rows
            if line_map:
                for line_key, line_info in line_map.items():
                    if isinstance(line_info, dict):
                        color = line_info.get('color', 'unbekannt')
                        style = line_info.get('style', 'unbekannt')
                        color_style = f'background-color: {color};' if color != 'unbekannt' else ''
                        html_content += f"""
                    <tr>
                        <td><strong>{line_key}</strong></td>
                        <td><span class="color-swatch" style="{color_style}"></span>{color}</td>
                        <td>{style}</td>
                    </tr>
"""
                    else:
                        html_content += f"""
                    <tr>
                        <td><strong>{line_key}</strong></td>
                        <td colspan="2">{line_info}</td>
                    </tr>
"""
            else:
                html_content += '<tr><td colspan="3">Keine Rohrleitungen in Legende gefunden</td></tr>'
            
            html_content += f"""
                </tbody>
            </table>
        </div>
        
        <h2>3. KPI Zusammenfassung</h2>
        <div class="kpi-summary">
            <div class="kpi-card">
                <div class="kpi-label">Quality Score</div>
                <div class="kpi-value">{kpis.get('quality_score', 0.0):.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Elemente</div>
                <div class="kpi-value">{len(elements)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Verbindungen</div>
                <div class="kpi-value">{len(connections)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Element-Typen</div>
                <div class="kpi-value">{kpis.get('unique_element_types', 0)}</div>
            </div>
        </div>
        
        <h2>4. Elemente</h2>
        <div class="elements-section">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Typ</th>
                        <th>Label</th>
                        <th>Confidence</th>
                        <th>Position (x, y)</th>
                        <th>Größe (w, h)</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            # Add element rows
            for el in elements[:100]:  # Limit to 100 for performance
                el_dict = el if isinstance(el, dict) else el.model_dump() if hasattr(el, 'model_dump') else {}
                el_id = el_dict.get('id', 'N/A')
                el_type = el_dict.get('type', 'N/A')
                el_label = el_dict.get('label', 'N/A')
                el_confidence = el_dict.get('confidence', 0.0)
                bbox = el_dict.get('bbox', {})
                if isinstance(bbox, dict):
                    x = bbox.get('x', 0)
                    y = bbox.get('y', 0)
                    w = bbox.get('width', 0)
                    h = bbox.get('height', 0)
                else:
                    x, y, w, h = 0, 0, 0, 0
                
                html_content += f"""
                    <tr>
                        <td>{el_id}</td>
                        <td>{el_type}</td>
                        <td>{el_label}</td>
                        <td>{el_confidence:.2f}</td>
                        <td>({x:.3f}, {y:.3f})</td>
                        <td>({w:.3f}, {h:.3f})</td>
                    </tr>
"""
            
            if len(elements) > 100:
                html_content += f'<tr><td colspan="6"><em>... und {len(elements) - 100} weitere Elemente</em></td></tr>'
            
            html_content += """
                </tbody>
            </table>
        </div>
        
        <h2>5. Verbindungen</h2>
        <div class="connections-section">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Von</th>
                        <th>Zu</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            # Add connection rows
            for conn in connections[:100]:  # Limit to 100 for performance
                conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else {}
                from_id = conn_dict.get('from_id', 'N/A')
                to_id = conn_dict.get('to_id', 'N/A')
                conn_confidence = conn_dict.get('confidence', 0.0)
                
                html_content += f"""
                    <tr>
                        <td>{from_id}</td>
                        <td>{to_id}</td>
                        <td>{conn_confidence:.2f}</td>
                    </tr>
"""
            
            if len(connections) > 100:
                html_content += f'<tr><td colspan="3"><em>... und {len(connections) - 100} weitere Verbindungen</em></td></tr>'
            
            html_content += """
                </tbody>
            </table>
        </div>
        
        <h2>6. Visualisierungen</h2>
        <div class="visualizations">
"""
            
            if debug_map_data:
                html_content += f"""
            <div class="viz-item">
                <h3>Debug Map</h3>
                <img src="{debug_map_data}" alt="Debug Map">
            </div>
"""
            
            if confidence_map_data:
                html_content += f"""
            <div class="viz-item">
                <h3>Confidence Map</h3>
                <img src="{confidence_map_data}" alt="Confidence Map">
            </div>
"""
            
            if kpi_dashboard_data:
                html_content += f"""
            <div class="viz-item">
                <h3>KPI Dashboard</h3>
                <img src="{kpi_dashboard_data}" alt="KPI Dashboard">
            </div>
"""
            
            if img_data_url:
                html_content += f"""
            <div class="viz-item">
                <h3>Original Diagramm</h3>
                <img src="{img_data_url}" alt="Original P&ID Diagramm">
            </div>
"""
            
            html_content += """
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #ecf0f1; text-align: center; color: #7f8c8d; font-size: 12px;">
            <p>P&ID Analyzer v2.0 - Automatisch generierter Report</p>
            <p>Für Fragen oder Support kontaktieren Sie bitte das Entwicklungsteam.</p>
        </div>
    </div>
</body>
</html>
"""
            
            # Save HTML report
            html_path = output_path / f"{base_name}_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTML report saved to: {html_path}")
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}", exc_info=True)

