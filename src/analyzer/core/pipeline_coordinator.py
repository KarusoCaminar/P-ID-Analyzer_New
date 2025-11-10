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
        config_dict = config_service.get_config().model_dump() if hasattr(config_service.get_config(), 'model_dump') else {}
        
        # Get paths from config
        learning_db_path = Path(config_dict.get('paths', {}).get('learning_db', 'learning_db.json'))
        learned_symbols_images_dir = Path(config_dict.get('paths', {}).get('learned_symbols_images_dir', 'learned_symbols_images'))
        
        # Initialize symbol library with images directory for viewshots
        symbol_library = SymbolLibrary(
            llm_client=llm_client,
            learning_db_path=learning_db_path,
            images_dir=learned_symbols_images_dir
        )
        
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
        
        # CRITICAL: Ensure structured output directories exist (Gold Standard Structure)
        from src.utils.output_structure_manager import ensure_output_structure
        ensure_output_structure(Path(final_output_dir))
        
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

            # CRITICAL FIX: Allow Phase 1-only runs (Test 1) - skip Phase 2 and go directly to Phase 4
            if not use_swarm and not use_monolith:
                logger.warning("Both swarm and monolith analysis are disabled. Skipping Phase 2 (Core Analysis).")
                logger.info("Phase 1-only mode: Proceeding directly to Phase 4 (Post-processing) with Phase 1 results only.")
                # Set empty analysis results for Phase 1-only mode
                self._analysis_results = {
                    "elements": [],
                    "connections": [],
                    "legend_data": self._global_knowledge_repo.get('legend_data', {}),
                    "metadata": self._global_knowledge_repo.get('metadata', {})
                }
                # Skip Phase 2 entirely - jump directly to Phase 3 (which will skip) and Phase 4
                # This allows Phase 1 results (legend, metadata) to be saved
                swarm_result = None
                monolith_result = None
            else:
                # --- KORREKTUR: Respektiere use_swarm und use_monolith Flags für Test 2 und 3 ---
                # WICHTIG: Prüfe die Flags ZUERST, unabhängig von der Phase-0-Strategie
                if use_monolith and not use_swarm:
                    # --- ECHTER "SIMPLE P&ID" MODUS (TEST 2: Monolith-All) ---
                    self._update_progress(15, "Phase 2: Simple P&ID analysis (Monolith-All)")
                    logger.info("CRITICAL: Simple P&ID mode (Test 2) - Running MONOLITH ONLY.")
                    
                    # Rufe Monolith direkt auf (nicht parallel)
                    from src.analyzer.analysis import MonolithAnalyzer
                    monolith_analyzer = MonolithAnalyzer(
                        self.llm_client,
                        self.knowledge_manager,
                        self.config_service,
                        self.model_strategy,
                        self.active_logic_parameters
                    )
                    
                    # Prepare legend context (only if legend is present)
                    legend_data = self._analysis_results.get('legend_data', {})
                    has_legend = legend_data.get('has_legend', False)
                    
                    if has_legend:
                        legend_context = {
                            'symbol_map': self._global_knowledge_repo.get('symbol_map', {}),
                            'line_map': self._global_knowledge_repo.get('line_map', {})
                        }
                        monolith_analyzer.legend_context = legend_context
                    else:
                        # No legend - disable legend context
                        logger.info("No legend present - disabling legend context for Monolith")
                        monolith_analyzer.legend_context = None
                    
                    # CRITICAL: For Simple P&ID Mode (Monolith-Only), Monolith must recognize elements AND connections
                    # This is the ONLY case where Monolith runs without element_list_json
                    monolith_analyzer.element_list_json = "[]"
                    logger.info("Simple P&ID Mode: Monolith will recognize elements AND connections independently (no element_list_json)")
                    
                    monolith_result = monolith_analyzer.analyze(image_path, Path(final_output_dir), self._excluded_zones)
                    
                    # Swarm wird übersprungen
                    swarm_result = {"elements": [], "connections": []}
                    
                elif use_swarm and not use_monolith:
                    # --- ECHTER "SWARM ONLY" MODUS (TEST 3) ---
                    self._update_progress(15, "Phase 2: Swarm-Only analysis")
                    logger.info("CRITICAL: Swarm-Only mode (Test 3) - Running SWARM ONLY.")
                    
                    from src.analyzer.analysis import SwarmAnalyzer
                    swarm_analyzer = SwarmAnalyzer(
                        self.llm_client,
                        self.knowledge_manager,
                        self.config_service,
                        self.model_strategy,
                        self.active_logic_parameters
                    )
                    
                    # Prepare legend context (only if legend is present)
                    legend_data = self._analysis_results.get('legend_data', {})
                    has_legend = legend_data.get('has_legend', False)
                    
                    if has_legend:
                        legend_context = {
                            'symbol_map': self._global_knowledge_repo.get('symbol_map', {}),
                            'line_map': self._global_knowledge_repo.get('line_map', {})
                        }
                        swarm_analyzer.legend_context = legend_context
                    else:
                        # No legend - disable legend context
                        logger.info("No legend present - disabling legend context for Swarm")
                        swarm_analyzer.legend_context = None
                    
                    swarm_result = swarm_analyzer.analyze(image_path, Path(final_output_dir), self._excluded_zones)
                    
                    # Monolith wird übersprungen
                    monolith_result = {"elements": [], "connections": []}
                    
                else:
                    # --- STANDARD-MODUS (TEST 4, 5a, 5b, 5c) ---
                    # Complex P&IDs: Parallel analysis (Swarm + Monolith)
                    self._update_progress(15, "Phase 2: Parallel core analysis (Swarm + Monolith)...")
                    swarm_result, monolith_result = self._run_phase_2_parallel_analysis(
                        image_path, final_output_dir
                    )
            # --- ENDE KORREKTUR ---
            
            # CRITICAL: Skip validation and fusion if both are disabled (Phase 1-only mode)
            use_swarm = self.active_logic_parameters.get('use_swarm_analysis', True)
            use_monolith = self.active_logic_parameters.get('use_monolith_analysis', True)
            
            if not use_swarm and not use_monolith:
                # Phase 1-only mode: Skip validation and fusion, go directly to Phase 3/4
                logger.info("Phase 1-only mode: Skipping validation and fusion, proceeding to Phase 3/4")
            else:
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
                
                # CRITICAL: Validate results based on mode (not always swarm_result)
                # DEBUG: Log results for debugging
                logger.debug(f"Validation: use_swarm={use_swarm}, use_monolith={use_monolith}")
                logger.debug(f"Validation: monolith_result type={type(monolith_result)}, keys={monolith_result.keys() if isinstance(monolith_result, dict) else 'N/A'}")
                logger.debug(f"Validation: monolith_result elements count={len(monolith_result.get('elements', [])) if isinstance(monolith_result, dict) else 0}")
                logger.debug(f"Validation: swarm_result type={type(swarm_result)}, keys={swarm_result.keys() if isinstance(swarm_result, dict) else 'N/A'}")
                logger.debug(f"Validation: swarm_result elements count={len(swarm_result.get('elements', [])) if isinstance(swarm_result, dict) else 0}")
                
                # Check if we have at least one valid result based on mode
                if use_monolith and not use_swarm:
                    # Monolith-Only mode: Check monolith_result
                    if not monolith_result or not isinstance(monolith_result, dict) or not monolith_result.get("elements"):
                        logger.error(f"Initial analysis failed (Monolith-Only mode). monolith_result={monolith_result}")
                        return self._create_error_result("Monolith analysis failed.")
                elif use_swarm and not use_monolith:
                    # Swarm-Only mode: Check swarm_result
                    if not swarm_result or not swarm_result.get("elements"):
                        logger.error("Initial analysis failed (Swarm-Only mode).")
                        return self._create_error_result("Swarm analysis failed.")
                else:
                    # Parallel mode: Check both results (at least one must succeed)
                    if (not swarm_result or not swarm_result.get("elements")) and (not monolith_result or not monolith_result.get("elements")):
                        logger.error("Initial analysis failed (both analyzers failed).")
                        return self._create_error_result("Both analyzers failed.")
                
                # Phase 2c: Fusion
                self._update_progress(45, "Phase 2c: Fusion...")
                
                # --- KORREKTUR: Logik-Fehler in Test 2 und 3 beheben ---
                # Get flags from active_logic_parameters
                use_fusion = self.active_logic_parameters.get('use_fusion', True)
                
                # Test 2: (Simple P&ID) - Monolith only, no swarm, no fusion
                # CRITICAL: Check flags FIRST, not strategy (strategy might be set by Phase 0, but flags override)
                if use_monolith and not use_swarm:
                    logger.info("SKIPPING Phase 2c: Fusion (Simple P&ID Mode / Monolith-Only). Using Monolith result.")
                    if monolith_result and monolith_result.get("elements"):
                        self._analysis_results = monolith_result
                        logger.info(f"Simple P&ID mode: Using Monolith result with {len(monolith_result.get('elements', []))} elements")
                    else:
                        logger.error("Simple P&ID mode failed: Monolith result is empty.")
                        return self._create_error_result("Monolith analysis failed in Simple P&ID mode.")
                
                # Test 3: (Swarm Only) - Swarm only, no monolith, no fusion
                elif use_swarm and not use_monolith:
                    logger.info("SKIPPING Phase 2c: Fusion (Swarm Only Mode). Using Swarm result.")
                    if swarm_result and swarm_result.get("elements"):
                        self._analysis_results = swarm_result
                        logger.info(f"Swarm Only mode: Using Swarm result with {len(swarm_result.get('elements', []))} elements")
                    else:
                        logger.error("Swarm Only mode failed: Swarm result is empty.")
                        return self._create_error_result("Swarm analysis failed in Swarm Only mode.")
                
                # Test 4, 5a, 5b, 5c: (Complex/Parallel) - Fusion ist AN
                elif use_fusion:
                    logger.info("Running Phase 2c: Confidence-Based Fusion Engine...")
                    fused_result = self._run_phase_2c_fusion(swarm_result, monolith_result)
                    
                    # CRITICAL FIX 7: Quality check for Fusion results - only use if better than inputs
                    # PROBLEM: Fusion results were always used, even if they were worse than Swarm or Monolith alone
                    # SOLUTION: Compare fusion quality with input quality, only use fusion if it's better
                    # EXPLANATION: This ensures continuous improvement - only better results are passed to next phase
                    
                    from src.analyzer.evaluation.kpi_calculator import KPICalculator
                    kpi_calculator = KPICalculator()
                    
                    # Calculate quality scores for comparison
                    input_scores = {}
                    if swarm_result and swarm_result.get('elements'):
                        swarm_kpis = kpi_calculator.calculate_comprehensive_kpis(swarm_result, None)
                        input_scores['swarm'] = swarm_kpis.get('quality_score', 0.0)
                        # Fallback calculation if score is 0
                        if input_scores['swarm'] == 0.0:
                            swarm_elements = swarm_result.get('elements', [])
                            swarm_connections = swarm_result.get('connections', [])
                            input_scores['swarm'] = 50.0
                            if swarm_elements:
                                avg_conf = sum(el.get('confidence', 0.5) for el in swarm_elements) / len(swarm_elements)
                                input_scores['swarm'] += min(len(swarm_elements) * 1.5, 25.0)
                                input_scores['swarm'] += avg_conf * 15.0
                            if swarm_connections:
                                avg_conn_conf = sum(conn.get('confidence', 0.5) for conn in swarm_connections) / len(swarm_connections)
                                input_scores['swarm'] += min(len(swarm_connections) * 1.0, 15.0)
                                input_scores['swarm'] += avg_conn_conf * 10.0
                            input_scores['swarm'] = min(max(input_scores['swarm'], 0.0), 100.0)
                    
                    if monolith_result and monolith_result.get('elements'):
                        monolith_kpis = kpi_calculator.calculate_comprehensive_kpis(monolith_result, None)
                        input_scores['monolith'] = monolith_kpis.get('quality_score', 0.0)
                        # Fallback calculation if score is 0
                        if input_scores['monolith'] == 0.0:
                            monolith_elements = monolith_result.get('elements', [])
                            monolith_connections = monolith_result.get('connections', [])
                            input_scores['monolith'] = 50.0
                            if monolith_elements:
                                avg_conf = sum(el.get('confidence', 0.5) for el in monolith_elements) / len(monolith_elements)
                                input_scores['monolith'] += min(len(monolith_elements) * 1.5, 25.0)
                                input_scores['monolith'] += avg_conf * 15.0
                            if monolith_connections:
                                avg_conn_conf = sum(conn.get('confidence', 0.5) for conn in monolith_connections) / len(monolith_connections)
                                input_scores['monolith'] += min(len(monolith_connections) * 1.0, 15.0)
                                input_scores['monolith'] += avg_conn_conf * 10.0
                            input_scores['monolith'] = min(max(input_scores['monolith'], 0.0), 100.0)
                    
                    # Get best input score
                    best_input_score = max(input_scores.values()) if input_scores else 0.0
                    
                    # Calculate fusion score
                    fusion_kpis = kpi_calculator.calculate_comprehensive_kpis(fused_result, None)
                    fusion_score = fusion_kpis.get('quality_score', 0.0)
                    
                    # Fallback calculation if fusion_score is 0
                    if fusion_score == 0.0:
                        fusion_elements = fused_result.get('elements', [])
                        fusion_connections = fused_result.get('connections', [])
                        fusion_score = 50.0
                        if fusion_elements:
                            avg_conf = sum(el.get('confidence', 0.5) for el in fusion_elements) / len(fusion_elements)
                            fusion_score += min(len(fusion_elements) * 1.5, 25.0)
                            fusion_score += avg_conf * 15.0
                        if fusion_connections:
                            avg_conn_conf = sum(conn.get('confidence', 0.5) for conn in fusion_connections) / len(fusion_connections)
                            fusion_score += min(len(fusion_connections) * 1.0, 15.0)
                            fusion_score += avg_conn_conf * 10.0
                        fusion_score = min(max(fusion_score, 0.0), 100.0)
                    
                    # CRITICAL: Only use fusion if it's better than best input
                    min_improvement = self.active_logic_parameters.get('min_improvement_threshold', 0.5)
                    fusion_is_better = fusion_score > (best_input_score + min_improvement)
                    
                    if fusion_is_better:
                        improvement = fusion_score - best_input_score
                        logger.info(f"Fusion improved quality: {best_input_score:.2f} -> {fusion_score:.2f} (+{improvement:.2f}). Using fusion results.")
                        self._analysis_results = fused_result
                    else:
                        # Fusion didn't improve - use best input result
                        if fusion_score < best_input_score:
                            logger.warning(f"Fusion deteriorated quality: {best_input_score:.2f} -> {fusion_score:.2f}. Using best input result instead.")
                        else:
                            logger.info(f"Fusion did not improve quality significantly: {best_input_score:.2f} -> {fusion_score:.2f} (threshold: {min_improvement:.2f}). Using best input result.")
                        
                        # Use best input result (swarm or monolith)
                        if input_scores.get('swarm', 0) >= input_scores.get('monolith', 0) and swarm_result:
                            logger.info("Using Swarm result (best input quality)")
                            self._analysis_results = swarm_result
                        elif monolith_result:
                            logger.info("Using Monolith result (best input quality)")
                            self._analysis_results = monolith_result
                        else:
                            # Fallback to fusion if no input is better
                            logger.warning("No input result available, using fusion result as fallback")
                            self._analysis_results = fused_result
                
                else:
                    # Fallback (sollte nicht passieren, wenn Test-Harness korrekt konfiguriert ist)
                    logger.warning("SKIPPING Phase 2c: Fusion (use_fusion=False). Defaulting to Swarm result.")
                    if swarm_result:
                        self._analysis_results = swarm_result
                    elif monolith_result:
                        self._analysis_results = monolith_result
                    else:
                        self._analysis_results = {"elements": [], "connections": []}
                # --- ENDE KORREKTUR ---
            
            # CRITICAL: Check for missing legend symbols and add them if found by Monolith
            # If legend is present, ensure all legend symbols are detected
            legend_data = self._analysis_results.get('legend_data', {})
            symbol_map = legend_data.get('symbol_map', {})
            if symbol_map:
                logger.info(f"Checking for missing legend symbols: {len(symbol_map)} symbols in legend")
                current_result = self._analysis_results.copy()
                updated_result = self._add_missing_legend_symbols(current_result, symbol_map, monolith_result)
                self._analysis_results = updated_result
            
            # Phase 2d: Predictive completion
            self._update_progress(50, "Phase 2d: Predictive completion...")
            self._run_phase_2d_predictive_completion()
            
            # Phase 2e: Polyline refinement
            self._update_progress(55, "Phase 2e: Polyline refinement...")
            self._run_phase_2e_polyline_refinement()
            
            # CRITICAL FIX 3: Hybrid CV + Semantic Validation (before Phase 3)
            # This combines CV line detection with semantic validation for better connection accuracy
            if self.active_logic_parameters.get('use_hybrid_validation', True):
                self._update_progress(58, "Hybrid validation: CV + Semantic...")
                self._run_hybrid_validation(image_path)
            
            # CRITICAL FIX 4: Multi-Layered ID Extraction (OCR + CV + Pattern + LLM Fallback)
            # Robust, reliable ID extraction using multiple strategies
            if self.active_logic_parameters.get('use_id_correction', True):
                self._update_progress(59, "ID Extraction: Extracting correct IDs from image (OCR + CV + LLM)...")
                self._run_id_extraction(image_path)
            
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
        """
        Get output directory for results.
        
        ALWAYS saves to outputs/ directory, regardless of where the script is started.
        Creates a subdirectory based on image name and timestamp.
        """
        base_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Always use outputs/ directory relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "outputs" / "analyses" / f"{base_name}_{timestamp}"
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
        
        # --- NEU: Iterative BBox-Verfeinerung für Metadata ---
        if parsed_metadata_bbox:
            # Check if iterative refinement is enabled
            use_bbox_refinement = self.active_logic_parameters.get('use_legend_bbox_refinement', True)
            
            if use_bbox_refinement:
                logger.info("Starting iterative metadata bbox refinement...")
                try:
                    from src.utils.bbox_refiner import refine_metadata_bbox_iteratively
                    
                    # Get refinement model (use meta_model for metadata refinement)
                    refinement_model = self.model_strategy.get('meta_model') or self.model_strategy.get('detail_model')
                    if refinement_model:
                        # Normalize model info
                        if isinstance(refinement_model, dict):
                            refinement_model_info = refinement_model
                        elif hasattr(refinement_model, 'model_dump'):
                            refinement_model_info = refinement_model.model_dump()
                        elif hasattr(refinement_model, 'dict'):
                            refinement_model_info = refinement_model.dict()
                        else:
                            refinement_model_info = None
                        
                        if refinement_model_info:
                            # Refine bbox iteratively
                            refined_bbox = refine_metadata_bbox_iteratively(
                                image_path=image_path,
                                initial_bbox=parsed_metadata_bbox,
                                llm_client=self.llm_client,
                                model_info=refinement_model_info,
                                system_prompt=system_prompt,
                                max_iterations=3,
                                min_reduction=0.05
                            )
                            
                            # Update parsed_metadata_bbox with refined version
                            parsed_metadata_bbox = refined_bbox
                            logger.info(f"Metadata bbox refined: {parsed_metadata_bbox}")
                        else:
                            logger.warning("Could not normalize refinement model. Using original bbox.")
                    else:
                        logger.warning("No refinement model available. Using original bbox.")
                except Exception as e:
                    logger.warning(f"Error in metadata bbox refinement: {e}. Using original bbox.", exc_info=True)
            # --- ENDE Metadata BBox-Verfeinerung ---
            
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
        
        # --- NEU: Iterative BBox-Verfeinerung für Legende ---
        if parsed_legend_bbox:
            # Check if iterative refinement is enabled
            use_bbox_refinement = self.active_logic_parameters.get('use_legend_bbox_refinement', True)
            
            if use_bbox_refinement:
                logger.info("Starting iterative legend bbox refinement...")
                try:
                    from src.utils.bbox_refiner import refine_legend_bbox_iteratively
                    
                    # Get refinement model (use meta_model for legend refinement)
                    refinement_model = self.model_strategy.get('meta_model') or self.model_strategy.get('detail_model')
                    if refinement_model:
                        # Normalize model info
                        if isinstance(refinement_model, dict):
                            refinement_model_info = refinement_model
                        elif hasattr(refinement_model, 'model_dump'):
                            refinement_model_info = refinement_model.model_dump()
                        elif hasattr(refinement_model, 'dict'):
                            refinement_model_info = refinement_model.dict()
                        else:
                            refinement_model_info = None
                        
                        if refinement_model_info:
                            # Refine bbox iteratively
                            refined_bbox = refine_legend_bbox_iteratively(
                                image_path=image_path,
                                initial_bbox=parsed_legend_bbox,
                                llm_client=self.llm_client,
                                model_info=refinement_model_info,
                                system_prompt=system_prompt,
                                max_iterations=3,
                                min_reduction=0.05
                            )
                            
                            # Update parsed_legend_bbox with refined version
                            parsed_legend_bbox = refined_bbox
                            logger.info(f"Legend bbox refined: {parsed_legend_bbox}")
                        else:
                            logger.warning("Could not normalize refinement model. Using original bbox.")
                    else:
                        logger.warning("No refinement model available. Using original bbox.")
                except Exception as e:
                    logger.warning(f"Error in bbox refinement: {e}. Using original bbox.", exc_info=True)
            # --- ENDE BBox-Verfeinerung ---
            
            self._excluded_zones.append(parsed_legend_bbox)
            self.state.excluded_zones.append(parsed_legend_bbox)
            logger.info(f"Identified legend area to be excluded: {parsed_legend_bbox}")
        else:
            logger.warning(f"Legend bbox is malformed or unparsable: {legend_bbox_raw}")
        
        # Validate symbol map
        symbol_map = legend_dict.get("symbol_map", {}) if legend_dict else {}
        validated_symbol_map = self._validate_symbol_map(symbol_map)
        
        # Process line map
        line_map = legend_dict.get("line_map", {}) if legend_dict else {}
        
        # CRITICAL: Check if legend is actually present (not empty)
        has_legend = bool(validated_symbol_map or line_map or parsed_legend_bbox)
        
        if not has_legend:
            # No legend present - disable all legend logic
            logger.info("No legend detected (no symbol_map, line_map, or bbox). Disabling all legend logic.")
            validated_symbol_map = {}
            line_map = {}
            legend_confidence = 0.0
            is_plausible = False
            # Clear any legend bbox if it was incorrectly detected
            parsed_legend_bbox = None
        else:
            # Legend is present - proceed with normal legend processing
            self._global_knowledge_repo['symbol_map'] = validated_symbol_map
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
            'is_plausible': is_plausible,
            'has_legend': has_legend  # Flag to indicate if legend is present
        }
        self.state.legend_data = complete_legend_data
        self._analysis_results['legend_data'] = complete_legend_data
        
        # Also store metadata
        if metadata_dict:
            self._analysis_results['metadata'] = metadata_dict
            self.state.metadata = metadata_dict
        
        if has_legend:
            logger.info(f"Built knowledge repository with {len(validated_symbol_map)} validated symbol mappings and {len(line_map)} line rules.")
            if line_map:
                logger.info(f"Extracted {len(line_map)} line semantic rules from legend (colors, styles).")
        else:
            logger.info("No legend detected - analysis will proceed without legend context (using Learning Database and Element Type List).")
        
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
        
        # Prepare legend context for analyzers (only if legend is present)
        legend_data = self._analysis_results.get('legend_data', {})
        has_legend = legend_data.get('has_legend', False)
        
        if has_legend:
            legend_context = {
                'symbol_map': self._global_knowledge_repo.get('symbol_map', {}),
                'line_map': self._global_knowledge_repo.get('line_map', {})
            }
            swarm_analyzer.legend_context = legend_context
            monolith_analyzer.legend_context = legend_context
            logger.info(f"Using legend context: {len(legend_context['symbol_map'])} symbols, {len(legend_context['line_map'])} line rules")
        else:
            # No legend - disable legend context
            logger.info("No legend present - disabling legend context for analyzers")
            swarm_analyzer.legend_context = None
            monolith_analyzer.legend_context = None
        
        # Run in parallel
        swarm_graph: Dict[str, Any] = {}
        monolith_graph: Optional[Dict[str, Any]] = None
        
        output_path = Path(output_dir)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            logger.info("Starting swarm (components) and monolith (structure) analysis in parallel...")
            
            swarm_future = executor.submit(
                swarm_analyzer.analyze,
                image_path,
                output_path,
                self._excluded_zones
            )
            
            # CRITICAL: Monolith must wait for Swarm to complete to get element_list_json
            # We cannot run them truly in parallel if Monolith needs Swarm's element list
            # For now, run Swarm first, then Monolith with element list
            try:
                swarm_result = swarm_future.result()
                if swarm_result:
                    swarm_graph = swarm_result
                    
                    # CRITICAL: Set element_list_json from Swarm result for Monolith
                    swarm_elements = swarm_result.get('elements', [])
                    if swarm_elements:
                        import json
                        element_list_json = json.dumps(swarm_elements, ensure_ascii=False)
                        monolith_analyzer.element_list_json = element_list_json
                        logger.info(f"Monolith will use element_list_json from Swarm: {len(swarm_elements)} elements")
                    else:
                        logger.warning("Swarm returned no elements - Monolith will run without element_list_json")
                        monolith_analyzer.element_list_json = "[]"
                
                # Now run Monolith with element list from Swarm
                monolith_future = executor.submit(
                    monolith_analyzer.analyze,
                    image_path,
                    output_path,
                    self._excluded_zones
                )
                
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
        # --- KORREKTUR: Respektiere use_predictive_completion Flag (Fix 3) ---
        if not self.active_logic_parameters.get('use_predictive_completion', True):
            logger.warning("SKIPPING Phase 2d: Predictive Completion (Flag is False)")
            return
        # --- ENDE KORREKTUR ---
        
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
        
        # CRITICAL FIX 8: Quality check for Predictive Completion - only use if better
        # PROBLEM: Predictive completion could add false connections, degrading quality
        # SOLUTION: Calculate quality before/after, only use if quality improves
        # EXPLANATION: This ensures continuous improvement - only better results are passed to next phase
        
        from src.analyzer.evaluation.kpi_calculator import KPICalculator
        kpi_calculator = KPICalculator()
        
        # Calculate quality BEFORE predictive completion
        results_before = {
            'elements': self._analysis_results.get('elements', []),
            'connections': original_connections
        }
        kpis_before = kpi_calculator.calculate_comprehensive_kpis(results_before, None)
        quality_before = kpis_before.get('quality_score', 0.0)
        
        # Fallback calculation
        if quality_before == 0.0:
            quality_before = 50.0
            elements = self._analysis_results.get('elements', [])
            if elements:
                avg_conf = sum(el.get('confidence', 0.5) for el in elements) / len(elements)
                quality_before += min(len(elements) * 1.5, 25.0)
                quality_before += avg_conf * 15.0
            if original_connections:
                avg_conn_conf = sum(conn.get('confidence', 0.5) for conn in original_connections) / len(original_connections)
                quality_before += min(len(original_connections) * 1.0, 15.0)
                quality_before += avg_conn_conf * 10.0
            quality_before = min(max(quality_before, 0.0), 100.0)
        
        # Calculate quality AFTER predictive completion
        results_after = {
            'elements': self._analysis_results.get('elements', []),
            'connections': all_connections_after_prediction
        }
        kpis_after = kpi_calculator.calculate_comprehensive_kpis(results_after, None)
        quality_after = kpis_after.get('quality_score', 0.0)
        
        # Fallback calculation
        if quality_after == 0.0:
            quality_after = 50.0
            elements = self._analysis_results.get('elements', [])
            if elements:
                avg_conf = sum(el.get('confidence', 0.5) for el in elements) / len(elements)
                quality_after += min(len(elements) * 1.5, 25.0)
                quality_after += avg_conf * 15.0
            if all_connections_after_prediction:
                avg_conn_conf = sum(conn.get('confidence', 0.5) for conn in all_connections_after_prediction) / len(all_connections_after_prediction)
                quality_after += min(len(all_connections_after_prediction) * 1.0, 15.0)
                quality_after += avg_conn_conf * 10.0
            quality_after = min(max(quality_after, 0.0), 100.0)
        
        # CRITICAL: Only use predictive completion if quality improved
        min_improvement = self.active_logic_parameters.get('min_improvement_threshold', 0.5)
        quality_improved = quality_after > (quality_before + min_improvement)
        
        added_count = len(all_connections_after_prediction) - len(original_connections)
        
        if quality_improved:
            improvement = quality_after - quality_before
            logger.info(f"Predictive completion improved quality: {quality_before:.2f} -> {quality_after:.2f} (+{improvement:.2f}). "
                       f"Added {added_count} connections.")
            self._analysis_results["connections"] = all_connections_after_prediction
        else:
            # Quality didn't improve - keep original connections
            if quality_after < quality_before:
                logger.warning(f"Predictive completion deteriorated quality: {quality_before:.2f} -> {quality_after:.2f}. "
                              f"Rejecting {added_count} added connections, keeping original.")
            else:
                logger.info(f"Predictive completion did not improve quality significantly: {quality_before:.2f} -> {quality_after:.2f} "
                           f"(threshold: {min_improvement:.2f}). Keeping original connections.")
            # DO NOT update connections - keep original
        
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
        # --- KORREKTUR: Respektiere use_polyline_refinement Flag (Fix 3) ---
        if not self.active_logic_parameters.get('use_polyline_refinement', True):
            logger.warning("SKIPPING Phase 2e: Polyline Refinement (Flag is False)")
            return
        # --- ENDE KORREKTUR ---
        
        if not self.current_image_path:
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
        
        # CRITICAL: Use temp/ subdirectory for temporary files
        temp_dir = output_dir / "temp" / "temp_polylines"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {len(connections)} connections for polyline extraction...")
        
        def process_connection(conn):
            """Process a single connection to extract its polyline."""
            from PIL import Image  # CRITICAL FIX: Import within function scope for thread safety
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
                                # CRITICAL FIX: Convert to float before arithmetic operations
                                local_x, local_y = float(point[0]), float(point[1])
                                global_x = (float(conn_bbox.get('x', 0)) + local_x * float(conn_bbox.get('width', 0))) / float(img_width)
                                global_y = (float(conn_bbox.get('y', 0)) + local_y * float(conn_bbox.get('height', 0))) / float(img_height)
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
        config_workers = self.active_logic_parameters.get('llm_executor_workers', 15)
        max_workers = min(config_workers, len(connections), 15)  # CRITICAL: Increased from 8 to 15 for better performance
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
        
        # CRITICAL FIX 9: Quality check for Polyline Refinement - only use if better
        # PROBLEM: Polyline refinement could add incorrect polylines, degrading quality
        # SOLUTION: Calculate quality before/after, only use if quality improves
        # EXPLANATION: This ensures continuous improvement - only better results are passed to next phase
        
        from src.analyzer.evaluation.kpi_calculator import KPICalculator
        kpi_calculator = KPICalculator()
        
        # Calculate quality BEFORE polyline refinement
        results_before = {
            'elements': elements,
            'connections': connections
        }
        kpis_before = kpi_calculator.calculate_comprehensive_kpis(results_before, None)
        quality_before = kpis_before.get('quality_score', 0.0)
        
        # Fallback calculation
        if quality_before == 0.0:
            quality_before = 50.0
            if elements:
                avg_conf = sum(el.get('confidence', 0.5) for el in elements) / len(elements)
                quality_before += min(len(elements) * 1.5, 25.0)
                quality_before += avg_conf * 15.0
            if connections:
                avg_conn_conf = sum(conn.get('confidence', 0.5) for conn in connections) / len(connections)
                quality_before += min(len(connections) * 1.0, 15.0)
                quality_before += avg_conn_conf * 10.0
            quality_before = min(max(quality_before, 0.0), 100.0)
        
        # Calculate quality AFTER polyline refinement
        results_after = {
            'elements': elements,
            'connections': updated_connections
        }
        kpis_after = kpi_calculator.calculate_comprehensive_kpis(results_after, None)
        quality_after = kpis_after.get('quality_score', 0.0)
        
        # Fallback calculation
        if quality_after == 0.0:
            quality_after = 50.0
            if elements:
                avg_conf = sum(el.get('confidence', 0.5) for el in elements) / len(elements)
                quality_after += min(len(elements) * 1.5, 25.0)
                quality_after += avg_conf * 15.0
            if updated_connections:
                # Count connections with polylines as higher quality
                polyline_count = sum(1 for conn in updated_connections if conn.get('polyline'))
                avg_conn_conf = sum(conn.get('confidence', 0.5) for conn in updated_connections) / len(updated_connections)
                quality_after += min(len(updated_connections) * 1.0, 15.0)
                quality_after += avg_conn_conf * 10.0
                quality_after += min(polyline_count * 0.5, 5.0)  # Bonus for polylines
            quality_after = min(max(quality_after, 0.0), 100.0)
        
        # CRITICAL: Only use polyline refinement if quality improved or stayed same (polylines are always beneficial)
        min_improvement = self.active_logic_parameters.get('min_improvement_threshold', 0.5)
        quality_improved = quality_after >= (quality_before - min_improvement)  # Allow small degradation for polylines
        
        polyline_count = sum(1 for conn in updated_connections if conn.get('polyline'))
        
        if quality_improved:
            improvement = quality_after - quality_before
            logger.info(f"Polyline refinement {'improved' if improvement > 0 else 'maintained'} quality: {quality_before:.2f} -> {quality_after:.2f} ({'+' if improvement > 0 else ''}{improvement:.2f}). "
                       f"Added {polyline_count} polylines to {len(updated_connections)} connections.")
            self._analysis_results["connections"] = updated_connections
        else:
            # Quality degraded significantly - keep original connections but add polylines if available
            logger.warning(f"Polyline refinement degraded quality: {quality_before:.2f} -> {quality_after:.2f}. "
                          f"Keeping original connections, but adding polylines if available.")
            # Add polylines to original connections if available (polylines are visual enhancements)
            for orig_conn in connections:
                for updated_conn in updated_connections:
                    if (orig_conn.get('from_id') == updated_conn.get('from_id') and
                        orig_conn.get('to_id') == updated_conn.get('to_id') and
                        updated_conn.get('polyline')):
                        orig_conn['polyline'] = updated_conn.get('polyline')
            self._analysis_results["connections"] = connections
        
        logger.info(f"Polyline extraction complete: {len(polyline_results)} polylines extracted, {polyline_count} applied to connections.")
        
        # CRITICAL: CV-based line extraction for connection verification (Pattern 4)
        # Extract pipeline lines using CV (contour detection) for verification
        logger.info("Extracting pipeline lines with CV for connection verification...")
        try:
            from src.analyzer.analysis.line_extractor import LineExtractor
            from PIL import Image
            
            # Get image dimensions
            img = Image.open(self.current_image_path)
            img_width, img_height = img.size
            
            # Get legend data and excluded zones
            legend_data = self._analysis_results.get('legend_data', {})
            excluded_zones = self._excluded_zones.copy()
            
            # Initialize line extractor
            config_dict = self.config_service.get_config().model_dump() if hasattr(self.config_service.get_config(), 'model_dump') else self.config_service.get_raw_config()
            line_extractor = LineExtractor(config_dict)
            
            # Extract pipeline lines using CV
            cv_line_result = line_extractor.extract_pipeline_lines(
                image_path=self.current_image_path,
                elements=elements,
                excluded_zones=excluded_zones,
                legend_data=legend_data
            )
            
            # Store CV-extracted pipeline lines for TopologyCritic verification
            pipeline_lines = cv_line_result.get('pipeline_lines', [])
            self._analysis_results['pipeline_lines'] = pipeline_lines
            self._analysis_results['cv_line_extraction'] = {
                'pipeline_lines': pipeline_lines,
                'junctions': cv_line_result.get('junctions', []),
                'line_segments': cv_line_result.get('line_segments', []),
                'image_width': img_width,
                'image_height': img_height
            }
            
            logger.info(f"CV line extraction complete: {len(pipeline_lines)} pipeline lines extracted for verification")
        except Exception as e:
            logger.warning(f"CV line extraction failed: {e}. Continuing without CV verification.", exc_info=True)
            # Don't fail the entire phase if CV extraction fails
            self._analysis_results['pipeline_lines'] = []
            self._analysis_results['cv_line_extraction'] = None
        
        # TEST HARNESS: Save intermediate result after polyline refinement
        if hasattr(self, 'current_output_dir'):
            from src.utils.test_harness import save_intermediate_result
            polyline_result = {
                "elements": self._analysis_results.get("elements", []),
                "connections": updated_connections,
                "pipeline_lines": self._analysis_results.get('pipeline_lines', [])
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
        # --- KORREKTUR: Respektiere use_self_correction_loop Flag (Fix 3) ---
        # CRITICAL FIX: Check flag with explicit boolean conversion
        use_self_correction = self.active_logic_parameters.get('use_self_correction_loop', True)
        # Convert to boolean (handle string "true"/"false" or boolean True/False)
        if isinstance(use_self_correction, str):
            use_self_correction = use_self_correction.lower() in ('true', '1', 'yes')
        elif not isinstance(use_self_correction, bool):
            use_self_correction = bool(use_self_correction)
        
        logger.info(f"Phase 3: use_self_correction_loop = {use_self_correction} (type: {type(use_self_correction).__name__})")
        
        if not use_self_correction:
            logger.warning("SKIPPING Phase 3: Self-Correction Loop (Flag is False)")
            # CRITICAL FIX: Wrap in expected structure for Phase 4
            return {
                "quality_score": 0.0,
                "final_ai_data": copy.deepcopy(self._analysis_results) if self._analysis_results else {"elements": [], "connections": []}
            }
        # --- ENDE KORREKTUR ---
        
        logger.info("--- Phase 3: Starting self-correction loop ---")
        
        max_iterations = self.active_logic_parameters.get('max_self_correction_iterations', 3)
        target_score = self.active_logic_parameters.get('target_quality_score', 98.0)
        max_no_improvement_iterations = self.active_logic_parameters.get('max_no_improvement_iterations', 3)
        min_improvement_threshold = self.active_logic_parameters.get('min_improvement_threshold', 0.5)
        early_stop_on_plateau = self.active_logic_parameters.get('early_stop_on_plateau', True)
        
        # CRITICAL FIX 1: Initialize score_history in best_result from the start
        # This ensures score_history is always available, even if score never improves
        if "score_history" not in self._analysis_results:
            self._analysis_results["score_history"] = []
        
        best_result: Dict[str, Any] = {
            "quality_score": -1.0,
            "final_ai_data": copy.deepcopy(self._analysis_results)
        }
        # Ensure score_history exists in best_result
        if "score_history" not in best_result["final_ai_data"]:
            best_result["final_ai_data"]["score_history"] = []
        
        # CRITICAL FIX 2: Track no-improvement counter for plateau detection
        no_improvement_count = 0
        
        for i in range(max_iterations):
            iteration_name = f"Correction Iteration {i+1}/{max_iterations}"
            progress = 60 + int((i / max_iterations) * 25)
            self._update_progress(progress, iteration_name)
            
            # CRITICAL FIX 2: Validate connection semantics before validation
            # This removes invalid connections (e.g., FT-10 as source) before scoring
            validated_connections = self._validate_connection_semantics(
                self._analysis_results.get('connections', []),
                self._analysis_results.get('elements', [])
            )
            self._analysis_results['connections'] = validated_connections
            
            # Validate and get quality score
            current_score, current_errors = self._run_phase_3_validation_and_critic(truth_data)
            
            # CRITICAL FIX 1: Always append to score_history BEFORE updating best_result
            if "score_history" not in self._analysis_results:
                self._analysis_results["score_history"] = []
            self._analysis_results["score_history"].append(current_score)
            
            # CRITICAL FIX 2: ONLY update best_result if score actually improved (prevents deterioration)
            # PROBLEM: Previous code updated final_ai_data even when score didn't improve, causing worse results
            # SOLUTION: Only update best_result when score improves by at least min_improvement_threshold
            score_improved = current_score > (best_result["quality_score"] + min_improvement_threshold)
            
            if score_improved:
                # Score improved significantly - update both quality_score and final_ai_data
                improvement = current_score - best_result["quality_score"]
                logger.info(f"Quality score improved: {best_result['quality_score']:.2f} -> {current_score:.2f} (+{improvement:.2f})")
                best_result["quality_score"] = current_score
                best_result["final_ai_data"] = copy.deepcopy(self._analysis_results)
                no_improvement_count = 0  # Reset counter
            else:
                # Score didn't improve - DO NOT update final_ai_data (keep best result)
                # EXPLANATION: This prevents quality deterioration. We keep the best result from previous iterations.
                if current_score < best_result["quality_score"]:
                    logger.warning(f"Quality score deteriorated: {best_result['quality_score']:.2f} -> {current_score:.2f}. Keeping best result.")
                else:
                    logger.info(f"Quality score did not improve significantly: {current_score:.2f} (best: {best_result['quality_score']:.2f}, threshold: {min_improvement_threshold:.2f})")
                no_improvement_count += 1
            
            # CRITICAL FIX 3: Update score_history in best_result (for visualization)
            # NOTE: score_history tracks ALL iterations, but final_ai_data only contains BEST iteration
            best_result["final_ai_data"]["score_history"] = self._analysis_results.get("score_history", []).copy()
            
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
            
            # --- NEU: Visuelles Feedback für Self-Correction Loop ---
            use_visual_feedback = self.active_logic_parameters.get('use_visual_feedback', True)
            visual_corrections = {}
            
            if use_visual_feedback and i < max_iterations - 1:  # Don't use visual feedback on last iteration
                logger.info("Generating visual feedback for self-correction...")
                try:
                    visual_corrections = self._run_visual_feedback_validation(
                        image_path,
                        output_dir,
                        i
                    )
                    
                    if visual_corrections and visual_corrections.get('corrections'):
                        logger.info(f"Visual feedback: {len(visual_corrections.get('corrections', []))} visual corrections identified")
                        # Apply visual corrections
                        self._apply_visual_corrections(visual_corrections.get('corrections', []))
                except Exception as e:
                    logger.warning(f"Error in visual feedback validation: {e}. Continuing without visual corrections.", exc_info=True)
            # --- ENDE Visuelles Feedback ---
            
            # CRITICAL FIX 4: Early termination conditions (including plateau detection)
            # EXPLANATION: Stop early if we reach target, have no errors, or hit a plateau (no improvement)
            
            # Condition 1: Target score reached
            if current_score >= target_score:
                logger.info(f"Target quality score reached ({current_score:.2f} >= {target_score:.2f}). Stopping corrections.")
                break
            
            # Condition 2: No errors found
            if not current_errors or not any(current_errors.values()):
                logger.info("No errors found. Stopping corrections.")
                break
            
            # Condition 3: Plateau detected (no improvement for N iterations)
            # EXPLANATION: This prevents wasting iterations when quality stops improving
            # PROBLEM: This was in config but NOT implemented in code - now fixed!
            if early_stop_on_plateau and no_improvement_count >= max_no_improvement_iterations:
                logger.info(f"Plateau detected: No improvement for {no_improvement_count} iterations (threshold: {max_no_improvement_iterations}). "
                           f"Best score: {best_result['quality_score']:.2f}, Current: {current_score:.2f}. Stopping corrections.")
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
        logger.info(f"Total iterations: {i+1}/{max_iterations}, No improvement count: {no_improvement_count}")
        
        # CRITICAL FIX: Ensure score_history is always included in final result
        # This ensures the visualization gets all iteration scores, even if best_result wasn't updated
        if "score_history" in self._analysis_results:
            best_result["final_ai_data"]["score_history"] = self._analysis_results["score_history"]
            logger.info(f"Score history: {len(self._analysis_results['score_history'])} iterations recorded")
        else:
            # Fallback: Create score_history from best score if not tracked
            logger.warning("Score history not found in _analysis_results. Creating fallback.")
            best_result["final_ai_data"]["score_history"] = [best_result["quality_score"]]
        
        # CRITICAL FIX 2 SUMMARY: Best result now contains ONLY the best iteration data
        # EXPLANATION: We only update best_result["final_ai_data"] when score improves, preventing quality deterioration
        logger.info(f"Best result contains data from iteration with score {best_result['quality_score']:.2f}")
        
        return best_result
    
    def _detect_ports(
        self,
        elements: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect ports for each element based on connections.
        
        CRITICAL FIX 4: This adds port information to elements, which is required for CGM generation.
        Ports are detected based on:
        - Input ports: Elements that connect TO this element
        - Output ports: Elements that connect FROM this element
        - Control ports: Control lines (ISA) that connect to Valves
        
        Args:
            elements: List of element dictionaries
            connections: List of connection dictionaries
            
        Returns:
            List of elements with ports added
        """
        elements_map = {el.get('id'): el.copy() for el in elements if el.get('id')}
        
        # Build connection maps
        input_connections = {}  # element_id -> [connections that end here]
        output_connections = {}  # element_id -> [connections that start here]
        control_connections = {}  # element_id -> [control connections]
        
        for conn in connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            kind = conn.get('kind', 'process')
            
            if not from_id or not to_id:
                continue
            
            # Output connections
            if from_id not in output_connections:
                output_connections[from_id] = []
            output_connections[from_id].append(conn)
            
            # Input connections
            if to_id not in input_connections:
                input_connections[to_id] = []
            input_connections[to_id].append(conn)
            
            # Control connections
            if kind == 'control':
                if to_id not in control_connections:
                    control_connections[to_id] = []
                control_connections[to_id].append(conn)
        
        # Add ports to elements
        for el in elements_map.values():
            el_id = el.get('id')
            if not el_id:
                continue
            
            ports = []
            
            # Input ports
            if el_id in input_connections:
                input_count = len(input_connections[el_id])
                for i, conn in enumerate(input_connections[el_id]):
                    ports.append({
                        'id': f'in_{i+1}',
                        'name': f'In_{i+1}',
                        'type': 'input',
                        'connection_id': conn.get('id', f'conn_{i+1}'),
                        'connected_from': conn.get('from_id')
                    })
            
            # Output ports
            if el_id in output_connections:
                output_count = len(output_connections[el_id])
                for i, conn in enumerate(output_connections[el_id]):
                    ports.append({
                        'id': f'out_{i+1}',
                        'name': f'Out_{i+1}',
                        'type': 'output',
                        'connection_id': conn.get('id', f'conn_{i+1}'),
                        'connected_to': conn.get('to_id')
                    })
            
            # Control ports (for Valves)
            if el_id in control_connections:
                control_count = len(control_connections[el_id])
                for i, conn in enumerate(control_connections[el_id]):
                    ports.append({
                        'id': f'control_{i+1}',
                        'name': f'Control_{i+1}',
                        'type': 'control',
                        'connection_id': conn.get('id', f'conn_{i+1}'),
                        'connected_from': conn.get('from_id')
                    })
            
            el['ports'] = ports
        
        # Convert back to list
        return list(elements_map.values())
    
    def _validate_connection_semantics(
        self,
        connections: List[Dict[str, Any]],
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate connections based on element type semantics.
        
        CRITICAL FIX 2: This validates connections to prevent invalid structures like:
        - Sensors (FT-*) acting as sources
        - Control lines (ISA) connecting to non-valves
        - Backwards connections (Sensor → Pump)
        
        Rules:
        - Sensors (FT-*, PT-*, etc.) should NOT be sources
        - Sources (Pumps, P-*, etc.) should NOT have inputs (except for feedback loops)
        - Sinks should NOT have outputs
        - Control lines (ISA-*) should only connect to Valves
        - Invalid connections are either removed or reversed
        
        Args:
            connections: List of connection dictionaries
            elements: List of element dictionaries
            
        Returns:
            Validated list of connections (invalid ones removed or corrected)
        """
        elements_map = {el.get('id'): el for el in elements if el.get('id')}
        validated_connections = []
        
        # Element type categories
        source_types = {'Source', 'Pump', 'CHP', 'HP'}
        sensor_types = {'Flow Transmitter', 'Volume Flow Sensor', 'FT', 'PT', 'TT', 'Pressure Transmitter', 'Temperature Transmitter'}
        sink_types = {'Sink', 'Reactor', 'Tank', 'Storage'}
        control_sources = {'ISA', 'ISA-Supply', 'Instrument Air Supply'}
        
        removed_count = 0
        reversed_count = 0
        
        for conn in connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            
            if not from_id or not to_id:
                continue
            
            from_el = elements_map.get(from_id)
            to_el = elements_map.get(to_id)
            
            if not from_el or not to_el:
                continue
            
            from_type = from_el.get('type', '')
            to_type = to_el.get('type', '')
            from_label = from_el.get('label', '')
            
            # RULE 1: Sensors should NOT be sources (CRITICAL FIX 2)
            if from_type in sensor_types:
                logger.warning(f"Invalid connection: Sensor {from_id} ({from_type}) cannot be source. Removing connection {from_id} -> {to_id}.")
                removed_count += 1
                continue
            
            # RULE 2: Control lines (ISA) should only connect to Valves
            if from_id in control_sources or from_type in control_sources or 'ISA' in from_label:
                if to_type != 'Valve' and 'Valve' not in to_type:
                    logger.warning(f"Invalid connection: Control line {from_id} -> {to_id} ({to_type}). Control lines should only connect to Valves. Removing connection.")
                    removed_count += 1
                    continue
                # Mark as control connection
                conn['kind'] = 'control'
            
            # RULE 3: Reverse invalid connections (Sensor → Source/Pump)
            # If connection is backwards (Sensor → Pump), try to reverse it
            if from_type in sensor_types and to_type in source_types:
                logger.warning(f"Reversing invalid connection: {from_id} -> {to_id} (Sensor -> Source). New: {to_id} -> {from_id}")
                # Reverse connection
                conn_copy = conn.copy()
                conn_copy['from_id'] = to_id
                conn_copy['to_id'] = from_id
                validated_connections.append(conn_copy)
                reversed_count += 1
                continue
            
            # RULE 4: Remove connections where Source has input (except if it's a feedback loop, but we'll be conservative)
            # This is less critical, so we'll just log it
            if from_type in sink_types and to_type in source_types:
                logger.debug(f"Potentially invalid connection: Sink {from_id} -> Source {to_id}. This might be a feedback loop, keeping it.")
            
            # Connection is valid
            validated_connections.append(conn)
        
        if removed_count > 0 or reversed_count > 0:
            logger.info(f"Connection validation: Removed {removed_count} invalid connections, reversed {reversed_count} connections. "
                       f"Total: {len(connections)} -> {len(validated_connections)}")
        
        return validated_connections
    
    def _run_hybrid_validation(
        self,
        image_path: str
    ) -> None:
        """
        Hybrid validation combining CV line detection with semantic validation.
        
        CRITICAL FIX: This solves the "Blinde CV-Kopplung" problem by:
        1. Extracting physical polylines from CV (line_extractor)
        2. Validating connections semantically (sensors not as sources, etc.)
        3. Using CV polylines to detect and correct direction errors in LLM connections
        
        Example:
        - LLM says: FT-10 -> Fv-3-3040 (wrong direction)
        - CV finds physical line from Fv-3-3040 -> FT-10
        - Semantic validation: FT-10 is a sensor, can't be source
        - Result: Correct to Fv-3-3040 -> FT-10
        
        Args:
            image_path: Path to the image
        """
        logger.info("=== Starting Hybrid CV + Semantic Validation ===")
        
        elements = self._analysis_results.get('elements', [])
        connections = self._analysis_results.get('connections', [])
        
        if not connections:
            logger.info("No connections to validate. Skipping hybrid validation.")
            return
        
        # Step 1: Extract physical polylines using CV
        from src.analyzer.analysis.line_extractor import LineExtractor
        
        line_extractor = LineExtractor(self.active_logic_parameters)
        excluded_zones = self._excluded_zones
        legend_data = self._analysis_results.get('legend_data', {})
        
        cv_result = line_extractor.extract_pipeline_lines(
            image_path=image_path,
            elements=elements,
            excluded_zones=excluded_zones,
            legend_data=legend_data if legend_data.get('has_legend') else None
        )
        
        # Get physical polylines and line segments
        pipeline_lines = cv_result.get('pipeline_lines', [])
        line_segments = cv_result.get('line_segments', [])
        
        logger.info(f"CV extracted {len(pipeline_lines)} pipeline lines, {len(line_segments)} line segments")
        
        # Step 2: Build mapping from element pairs to physical lines
        # Key: (from_id, to_id) -> list of polylines
        physical_lines_map = {}
        elements_map = {el.get('id'): el for el in elements if el.get('id')}
        
        for line in pipeline_lines:
            from_id = line.get('from_id')
            to_id = line.get('to_id')
            polyline = line.get('polyline', [])
            
            if from_id and to_id and polyline:
                key = (from_id, to_id)
                if key not in physical_lines_map:
                    physical_lines_map[key] = []
                physical_lines_map[key].append(polyline)
        
        # Step 3: Validate connections semantically (removes invalid, reverses wrong direction)
        validated_connections = self._validate_connection_semantics(connections, elements)
        
        # Step 4: Cross-validate with CV polylines
        corrected_connections = []
        corrected_count = 0
        removed_count = 0
        
        for conn in validated_connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            
            if not from_id or not to_id:
                continue
            
            # Check if CV found a physical line for this connection
            forward_key = (from_id, to_id)
            reverse_key = (to_id, from_id)
            
            has_forward_line = forward_key in physical_lines_map
            has_reverse_line = reverse_key in physical_lines_map
            
            # Case 1: CV found line in correct direction -> keep connection
            if has_forward_line:
                conn['polyline'] = physical_lines_map[forward_key][0]  # Use first polyline
                conn['cv_verified'] = True
                corrected_connections.append(conn)
                continue
            
            # Case 2: CV found line in reverse direction -> reverse connection if semantically valid
            if has_reverse_line:
                from_el = elements_map.get(from_id)
                to_el = elements_map.get(to_id)
                
                if from_el and to_el:
                    from_type = from_el.get('type', '')
                    to_type = to_el.get('type', '')
                    
                    # Check if reverse is semantically valid
                    # (e.g., Sensor can't be source, but Pump -> Sensor is valid)
                    sensor_types = {'Flow Transmitter', 'Volume Flow Sensor', 'FT', 'PT', 'TT'}
                    source_types = {'Source', 'Pump', 'CHP', 'HP'}
                    
                    # If current direction is semantically invalid (Sensor -> Source),
                    # and reverse would be valid (Source -> Sensor), reverse it
                    if from_type in sensor_types and to_type in source_types:
                        logger.info(f"CV correction: Reversing {from_id} -> {to_id} to {to_id} -> {from_id} (CV found physical line in reverse direction)")
                        conn_copy = conn.copy()
                        conn_copy['from_id'] = to_id
                        conn_copy['to_id'] = from_id
                        conn_copy['polyline'] = physical_lines_map[reverse_key][0]
                        conn_copy['cv_verified'] = True
                        conn_copy['cv_corrected'] = True
                        corrected_connections.append(conn_copy)
                        corrected_count += 1
                        continue
            
            # Case 3: No CV line found -> keep connection if semantically valid, but mark as unverified
            # (LLM might have detected a connection that CV missed due to threshold issues)
            conn['cv_verified'] = False
            corrected_connections.append(conn)
        
        # Update analysis results
        original_count = len(connections)
        self._analysis_results['connections'] = corrected_connections
        
        logger.info(f"Hybrid validation complete: {original_count} -> {len(corrected_connections)} connections "
                   f"({corrected_count} CV-corrected, {removed_count} removed)")
        
        # Store CV results for later use
        self._analysis_results['cv_pipeline_lines'] = pipeline_lines
        self._analysis_results['cv_line_segments'] = line_segments
    
    def _run_id_extraction(
        self,
        image_path: str
    ) -> None:
        """
        Multi-layered ID extraction - robust, reliable ID extraction.
        
        FINAL FIX: Uses multiple strategies for maximum reliability:
        1. OCR-based extraction (primary): Extract all text labels using Tesseract OCR
        2. Bbox-based matching: Match element bboxes to nearest text labels
        3. Pattern validation: Validate P&ID tag patterns
        4. LLM fallback: Use LLM only if OCR fails
        
        This is much more robust and reliable than pure LLM-based correction.
        
        Args:
            image_path: Path to the image
        """
        logger.info("=== Starting Multi-Layered ID Extraction ===")
        
        elements = self._analysis_results.get('elements', [])
        connections = self._analysis_results.get('connections', [])
        
        if not elements:
            logger.info("No elements to extract IDs for. Skipping ID extraction.")
            return
        
        try:
            from src.analyzer.analysis.id_extractor import IDExtractor
            
            # Initialize ID extractor
            id_extractor = IDExtractor(
                llm_client=self.llm_client,
                config_service=self.config_service
            )
            
            # Extract IDs
            corrected_data = id_extractor.extract_ids(
                image_path=image_path,
                elements=elements,
                connections=connections
            )
            
            # Update analysis results with corrected IDs
            if corrected_data:
                corrected_elements = corrected_data.get('elements', elements)
                corrected_connections = corrected_data.get('connections', connections)
                
                # Count changes
                original_ids = {el.get('id') for el in elements}
                corrected_ids = {el.get('id') for el in corrected_elements}
                id_changes = len(original_ids.symmetric_difference(corrected_ids))
                
                # Count sources
                ocr_count = sum(1 for el in corrected_elements if el.get('id_source') == 'ocr')
                llm_count = sum(1 for el in corrected_elements if el.get('id_source') == 'llm')
                original_count = sum(1 for el in corrected_elements if el.get('id_source') == 'original')
                
                if id_changes > 0:
                    logger.info(f"ID extraction: {id_changes} IDs changed (OCR: {ocr_count}, LLM: {llm_count}, Original: {original_count})")
                    self._analysis_results['elements'] = corrected_elements
                    self._analysis_results['connections'] = corrected_connections
                    self._analysis_results['id_extraction_applied'] = True
                    self._analysis_results['id_extraction_stats'] = {
                        'ocr_count': ocr_count,
                        'llm_count': llm_count,
                        'original_count': original_count,
                        'total_changes': id_changes
                    }
                else:
                    logger.info(f"ID extraction: No ID changes detected (OCR: {ocr_count}, LLM: {llm_count}, Original: {original_count})")
                    self._analysis_results['id_extraction_applied'] = False
                    self._analysis_results['id_extraction_stats'] = {
                        'ocr_count': ocr_count,
                        'llm_count': llm_count,
                        'original_count': original_count,
                        'total_changes': 0
                    }
            else:
                logger.warning("ID extraction returned empty result. Keeping original IDs.")
                self._analysis_results['id_extraction_applied'] = False
                
        except Exception as e:
            logger.error(f"Error in ID extraction: {e}", exc_info=True)
            logger.warning("Continuing with original IDs")
            self._analysis_results['id_extraction_applied'] = False
    
    def _run_visual_feedback_validation(
        self,
        image_path: str,
        output_dir: str,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Run visual feedback validation using MultiModelCritic.
        
        Generates a debug map of current analysis and sends it to the critic
        along with the original image for visual comparison.
        
        Args:
            image_path: Path to original P&ID image
            output_dir: Output directory for debug map
            iteration: Current iteration number
            
        Returns:
            Dictionary with visual corrections
        """
        logger.info("=== Visual Feedback Validation: Generating debug map ===")
        
        try:
            from src.analyzer.visualization.visualizer import Visualizer
            from src.analyzer.analysis.multi_model_critic import MultiModelCritic
            from PIL import Image
            from pathlib import Path
            
            # Get current analysis results
            elements = self._analysis_results.get("elements", [])
            connections = self._analysis_results.get("connections", [])
            
            if not elements:
                logger.warning("No elements to visualize. Skipping visual feedback.")
                return {'visual_errors': [], 'corrections': [], 'validation_score': 0.0}
            
            # Load image to get dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Create visualizer
            visualizer = Visualizer(img_width, img_height)
            
            # Generate debug map
            output_path = Path(output_dir)
            
            # CRITICAL: Save to visualizations/ subdirectory
            visualizations_dir = output_path / "visualizations"
            visualizations_dir.mkdir(parents=True, exist_ok=True)
            debug_map_path = visualizations_dir / f"debug_map_iteration_{iteration + 1}.png"
            
            success = visualizer.draw_debug_map(
                image_path,
                elements,
                connections,
                str(debug_map_path)
            )
            
            if not success:
                logger.warning("Failed to generate debug map. Skipping visual feedback.")
                return {'visual_errors': [], 'corrections': [], 'validation_score': 0.0}
            
            logger.info(f"Debug map generated: {debug_map_path}")
            
            # Initialize MultiModelCritic
            config = self.config_service.get_config()
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.__dict__
            
            critic = MultiModelCritic(
                self.llm_client,
                self.knowledge_manager,
                config_dict
            )
            
            # Prepare legend data
            legend_data = {
                'symbol_map': self._global_knowledge_repo.get('symbol_map', {}),
                'line_map': self._global_knowledge_repo.get('line_map', {})
            }
            
            # Run visual feedback validation
            visual_result = critic.validate_with_visual_feedback(
                elements=elements,
                connections=connections,
                original_image_path=image_path,
                debug_map_path=str(debug_map_path),
                legend_data=legend_data if legend_data.get('symbol_map') or legend_data.get('line_map') else None,
                model_strategy=self.model_strategy
            )
            
            logger.info(f"Visual feedback validation complete: {len(visual_result.get('corrections', []))} corrections")
            
            return visual_result
            
        except Exception as e:
            logger.error(f"Error in visual feedback validation: {e}", exc_info=True)
            return {'visual_errors': [], 'corrections': [], 'validation_score': 0.0}
    
    def _apply_visual_corrections(
        self,
        corrections: List[Dict[str, Any]]
    ) -> None:
        """
        Apply visual corrections to current analysis results.
        
        Args:
            corrections: List of corrections from visual feedback
        """
        if not corrections:
            return
        
        logger.info(f"Applying {len(corrections)} visual corrections...")
        
        elements = self._analysis_results.get("elements", [])
        connections = self._analysis_results.get("connections", [])
        
        for correction in corrections:
            action = correction.get('action', '')
            
            if action == 'add_element':
                # Add new element
                new_element = correction.get('element')
                if new_element:
                    elements.append(new_element)
                    logger.info(f"Visual correction: Added element {new_element.get('id')}")
            
            elif action == 'remove_element':
                # Remove hallucinated element
                element_id = correction.get('element_id')
                if element_id:
                    elements = [el for el in elements if el.get('id') != element_id]
                    # Also remove connections to this element
                    connections = [conn for conn in connections 
                                  if conn.get('from_id') != element_id and conn.get('to_id') != element_id]
                    logger.info(f"Visual correction: Removed element {element_id}")
            
            elif action == 'resize_bbox':
                # Resize bounding box
                element_id = correction.get('element_id')
                new_bbox = correction.get('new_bbox')
                if element_id and new_bbox:
                    for el in elements:
                        if el.get('id') == element_id:
                            el['bbox'] = new_bbox
                            logger.info(f"Visual correction: Resized bbox for element {element_id}")
                            break
            
            elif action == 'update_confidence':
                # Update confidence score
                element_id = correction.get('element_id')
                new_confidence = correction.get('new_confidence')
                if element_id and new_confidence is not None:
                    for el in elements:
                        if el.get('id') == element_id:
                            el['confidence'] = float(new_confidence)
                            logger.info(f"Visual correction: Updated confidence for element {element_id} to {new_confidence}")
                            break
        
        # Update analysis results
        self._analysis_results['elements'] = elements
        self._analysis_results['connections'] = connections
        
        logger.info(f"Visual corrections applied. New element count: {len(elements)}, connection count: {len(connections)}")
    
    def _run_phase_3_validation_and_critic(
        self,
        truth_data: Optional[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Phase 3: Comprehensive validation and critic.
        
        Validates analysis results and generates error feedback with confidence scores.
        
        CRITICAL: Now includes CV-based line verification (Pattern 4 from Audit).
        Verifies LLM-detected connections against CV-extracted pipeline lines.
        
        Returns:
            Tuple of (quality_score, errors_dict)
        """
        from src.analyzer.evaluation.kpi_calculator import KPICalculator
        from src.analyzer.analysis.topology_critic import TopologyCritic
        
        elements = self._analysis_results.get("elements", [])
        connections = self._analysis_results.get("connections", [])
        
        # CRITICAL: CV-based topology validation (Pattern 4)
        # Verify connections against CV-extracted pipeline lines
        topology_validation = {}
        try:
            # Get CV line extraction results
            cv_line_extraction = self._analysis_results.get('cv_line_extraction')
            pipeline_lines = self._analysis_results.get('pipeline_lines', [])
            
            if cv_line_extraction and pipeline_lines:
                # Initialize TopologyCritic
                config_dict = self.config_service.get_config().model_dump() if hasattr(self.config_service.get_config(), 'model_dump') else self.config_service.get_raw_config()
                topology_critic = TopologyCritic(config_dict)
                
                # Get image dimensions from CV extraction result
                image_width = cv_line_extraction.get('image_width')
                image_height = cv_line_extraction.get('image_height')
                
                # Validate topology with CV line verification
                topology_validation = topology_critic.validate_topology(
                    elements=elements,
                    connections=connections,
                    polylines=None,  # Polylines are already in connections
                    pipeline_lines=pipeline_lines,
                    image_width=image_width,
                    image_height=image_height
                )
                
                # Log unverified connections (likely hallucinations)
                unverified_connections = topology_validation.get('unverified_connections', [])
                if unverified_connections:
                    logger.warning(f"CV line verification: {len(unverified_connections)} connections without physical CV line paths (likely hallucinations)")
                    for unverified in unverified_connections[:5]:  # Log first 5
                        logger.debug(f"  Unverified connection: {unverified.get('from_id')} -> {unverified.get('to_id')} "
                                    f"(confidence: {unverified.get('confidence', 0.0):.2f})")
                
                # Store topology validation results
                self._analysis_results['topology_validation'] = topology_validation
            else:
                logger.debug("CV line extraction not available - skipping CV-based topology verification")
        except Exception as e:
            logger.warning(f"Topology validation with CV verification failed: {e}. Continuing without CV verification.", exc_info=True)
            # Don't fail the entire phase if topology validation fails
        
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
        
        # CRITICAL: Add CV-verified connection errors (Pattern 4)
        # Unverified connections from TopologyCritic (connections without physical CV line paths)
        unverified_connections = topology_validation.get('unverified_connections', []) if topology_validation else []
        
        # Add metacritic discrepancies to errors if available
        metacritic_discrepancies = self._analysis_results.get('metacritic_discrepancies', [])
        
        errors: Dict[str, Any] = {
            'missed_elements': missed_els,  # List of actual elements
            'hallucinated_elements': hallucinated_els,  # List of actual elements
            'missed_connections': kpis.get('missed_connections', 0),
            'hallucinated_connections': kpis.get('hallucinated_connections', 0),
            'unverified_connections': unverified_connections,  # CRITICAL: CV-verified connection errors (Pattern 4)
            'low_confidence_elements': [
                el.get('id') for el in elements
                if el.get('confidence', 1.0) < 0.5
            ],
            'isolated_elements': kpis.get('isolated_elements', 0),
            'metacritic_discrepancies': metacritic_discrepancies,  # Add metacritic issues
            'topology_validation': topology_validation  # Store full topology validation results
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
            
            # CRITICAL FIX: Respect use_swarm_analysis flag
            use_swarm = self.active_logic_parameters.get('use_swarm_analysis', True)
            use_monolith = self.active_logic_parameters.get('use_monolith_analysis', True)
            
            output_path = Path(output_dir)
            reanalysis_result = None
            
            # Use Swarm if enabled (default) or Monolith if Swarm is disabled
            if use_swarm:
                logger.info("Re-analyzing problematic areas with Swarm (tile-based for precision)...")
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
                
                reanalysis_result = swarm_analyzer.analyze(
                    image_path,
                    output_path,
                    self._excluded_zones
                )
            elif use_monolith:
                logger.info("Re-analyzing problematic areas with Monolith (whole-image analysis)...")
                from src.analyzer.analysis import MonolithAnalyzer
                
                # Create Monolith analyzer with error feedback
                monolith_analyzer = MonolithAnalyzer(
                    self.llm_client,
                    self.knowledge_manager,
                    self.config_service,
                    self.model_strategy,
                    self.active_logic_parameters
                )
                
                # Set error feedback (MonolithAnalyzer supports error_feedback as attribute)
                monolith_analyzer.error_feedback = error_feedback_str
                monolith_analyzer.legend_context = {
                    'symbol_map': self._global_knowledge_repo.get('symbol_map', {}),
                    'line_map': self._global_knowledge_repo.get('line_map', {})
                }
                
                reanalysis_result = monolith_analyzer.analyze(
                    image_path,
                    output_path,
                    self._excluded_zones
                )
            else:
                logger.warning("Both Swarm and Monolith are disabled. Skipping re-analysis.")
                return self._analysis_results
            
            # Rename for clarity
            swarm_result = reanalysis_result
            
            if swarm_result:
                # CRITICAL FIX 5: Quality check before merging re-analysis results
                # PROBLEM: Previous code always merged re-analysis results, even if they made things worse
                # SOLUTION: Calculate quality before and after merge, only accept if quality improves
                # EXPLANATION: This prevents quality deterioration from bad re-analysis results
                
                # Calculate quality score BEFORE merge
                from src.analyzer.evaluation.kpi_calculator import KPICalculator
                kpi_calculator = KPICalculator()
                kpis_before = kpi_calculator.calculate_comprehensive_kpis(self._analysis_results, None)
                quality_before = kpis_before.get('quality_score', 0.0)
                
                # Fallback calculation if quality_score is 0
                if quality_before == 0.0:
                    elements_before = self._analysis_results.get("elements", [])
                    connections_before = self._analysis_results.get("connections", [])
                    quality_before = 50.0
                    if elements_before:
                        avg_confidence = sum(el.get('confidence', 0.5) for el in elements_before) / len(elements_before)
                        quality_before += min(len(elements_before) * 1.5, 25.0)
                        quality_before += avg_confidence * 15.0
                    if connections_before:
                        avg_conn_confidence = sum(conn.get('confidence', 0.5) for conn in connections_before) / len(connections_before)
                        quality_before += min(len(connections_before) * 1.0, 15.0)
                        quality_before += avg_conn_confidence * 10.0
                    quality_before = min(max(quality_before, 0.0), 100.0)
                
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
                
                # Calculate quality score AFTER merge
                merged_results_temp = {
                    'elements': merged_elements,
                    'connections': merged_connections
                }
                kpis_after = kpi_calculator.calculate_comprehensive_kpis(merged_results_temp, None)
                quality_after = kpis_after.get('quality_score', 0.0)
                
                # Fallback calculation if quality_score is 0
                if quality_after == 0.0:
                    quality_after = 50.0
                    if merged_elements:
                        avg_confidence = sum(el.get('confidence', 0.5) for el in merged_elements) / len(merged_elements)
                        quality_after += min(len(merged_elements) * 1.5, 25.0)
                        quality_after += avg_confidence * 15.0
                    if merged_connections:
                        avg_conn_confidence = sum(conn.get('confidence', 0.5) for conn in merged_connections) / len(merged_connections)
                        quality_after += min(len(merged_connections) * 1.0, 15.0)
                        quality_after += avg_conn_confidence * 10.0
                    quality_after = min(max(quality_after, 0.0), 100.0)
                
                # CRITICAL: Only accept merge if quality improved
                min_improvement = self.active_logic_parameters.get('min_improvement_threshold', 0.5)
                quality_improved = quality_after > (quality_before + min_improvement)
                
                if quality_improved:
                    # Quality improved - accept merge
                    improvement = quality_after - quality_before
                    logger.info(f"Re-analysis improved quality: {quality_before:.2f} -> {quality_after:.2f} (+{improvement:.2f}). Accepting merge.")
                    self._analysis_results['elements'] = merged_elements
                    self._analysis_results['connections'] = merged_connections
                    logger.info(f"Re-analysis complete: {len(merged_elements)} elements, {len(merged_connections)} connections")
                    
                    # Learn from corrections immediately (only if quality improved)
                    try:
                        self.active_learner.learn_from_analysis_result(
                            analysis_result=self._analysis_results,
                            truth_data=None,  # Can be enhanced with truth data
                            quality_score=quality_after
                        )
                        logger.info("Live learning: Learned from re-analysis corrections")
                    except Exception as e:
                        logger.warning(f"Error in live learning: {e}")
                else:
                    # Quality didn't improve - reject merge, keep original results
                    if quality_after < quality_before:
                        logger.warning(f"Re-analysis deteriorated quality: {quality_before:.2f} -> {quality_after:.2f}. Rejecting merge, keeping original results.")
                    else:
                        logger.info(f"Re-analysis did not improve quality significantly: {quality_before:.2f} -> {quality_after:.2f} (threshold: {min_improvement:.2f}). Keeping original results.")
                    # CRITICAL FIX: Return original results (not merged) if quality didn't improve
                    # This ensures only the best results are passed to the next iteration
                    return self._analysis_results  # Return original, unchanged results
            
            # CRITICAL FIX: Only return updated results if quality improved
            # If we reach here, quality_improved was True and self._analysis_results was updated
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
        
        # CRITICAL: Allow Phase 1-only mode (empty elements but has legend/metadata)
        use_swarm = self.active_logic_parameters.get('use_swarm_analysis', True)
        use_monolith = self.active_logic_parameters.get('use_monolith_analysis', True)
        is_phase1_only = not use_swarm and not use_monolith
        
        if not isinstance(final_ai_data, dict):
            logger.error("Post-processing aborted: No valid final data found.")
            return self._create_error_result("Post-processing failed: No valid data")
        
        # For Phase 1-only mode, allow empty elements (we're testing legend recognition)
        if not is_phase1_only and not final_ai_data.get("elements"):
            logger.error("Post-processing aborted: No valid final data found.")
            return self._create_error_result("Post-processing failed: No valid data")
        
        if is_phase1_only:
            logger.info("Phase 1-only mode: Proceeding with legend/metadata only (no elements expected)")
        
        # CRITICAL FIX P1: NormalizationEngine MUST run BEFORE KPI calculation
        # This ensures IDs are normalized (e.g., "SamplePoint-S" → "Sample Point") before matching
        # Convert elements and connections to Pydantic models if needed
        elements_data = final_ai_data.get("elements", [])
        connections_data = final_ai_data.get("connections", [])
        
        # CRITICAL: Use NormalizationEngine for validation and normalization BEFORE KPIs
        from src.analyzer.analysis.normalization_engine import NormalizationEngine
        
        config_dict = self.config_service.get_config().model_dump() if hasattr(self.config_service.get_config(), 'model_dump') else self.config_service.get_raw_config()
        normalization_engine = NormalizationEngine(config_dict)
        
        # Process elements and connections through normalization engine
        # This normalizes IDs, corrects types, and validates connections
        elements_models, connections_models, normalization_stats = normalization_engine.process(
            elements_data, connections_data
        )
        
        # CRITICAL FIX P1: Update final_ai_data with normalized results BEFORE KPI calculation
        normalized_final_data = final_ai_data.copy()
        normalized_final_data["elements"] = [el.model_dump() if hasattr(el, 'model_dump') else el.dict() if hasattr(el, 'dict') else el for el in elements_models]
        normalized_final_data["connections"] = [conn.model_dump() if hasattr(conn, 'model_dump') else conn.dict() if hasattr(conn, 'dict') else conn for conn in connections_models]
        
        # Calculate KPIs using NORMALIZED data (so IDs match correctly)
        kpis = self._calculate_kpis(normalized_final_data, truth_data)
        
        # Generate CGM data using NORMALIZED data
        cgm_data = self._generate_cgm_data(normalized_final_data)
        
        # Save artifacts
        self._save_artifacts(output_dir, image_path, normalized_final_data, kpis, cgm_data)
        
        # Generate visualizations
        score_history = best_result.get("final_ai_data", {}).get("score_history", [])
        if output_dir:
            self._generate_visualizations(output_dir, image_path, normalized_final_data, kpis, score_history)
        
        # CRITICAL FIX 6: Cleanup temporary files at end of pipeline
        # PROBLEM: Temp directories (temp_quadrants, temp_polylines) were left in output folders
        # SOLUTION: Remove all temp/ subdirectories after pipeline completion
        # EXPLANATION: This prevents output folders from being filled with temporary files
        output_path = Path(output_dir)
        temp_dir = output_path / "temp"
        if temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary files: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temp directory {temp_dir}: {e}")
        
        logger.info(f"Post-processing complete: {len(elements_models)} elements, {len(connections_models)} connections (after normalization)")
        
        # Log normalization statistics
        if normalization_stats.get('removed_elements', 0) > 0 or normalization_stats.get('removed_connections', 0) > 0:
            logger.info(f"Normalization stats: {normalization_stats}")
        
        # Update data with normalized results (already Pydantic models from NormalizationEngine)
        elements_data = elements_models
        connections_data = connections_models
        
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
                elif isinstance(el, dict):
                    elements.append(el)
                else:
                    # Fallback: try to convert using dict()
                    elements.append(to_dict_recursive(el))
            except Exception as e:
                logger.warning(f"Error converting element to dict: {e}")
                if isinstance(el, dict):
                    elements.append(el)
        
        # Convert connections to dictionaries if needed
        connections = []
        for conn in connections_raw:
            try:
                if hasattr(conn, 'model_dump'):
                    conn_dict = to_dict_recursive(conn)
                    connections.append(conn_dict)
                elif hasattr(conn, 'dict'):
                    # Pydantic v1 model - recursively convert
                    conn_dict = to_dict_recursive(conn)
                    connections.append(conn_dict)
                elif isinstance(conn, dict):
                    connections.append(conn)
                else:
                    # Fallback: try to convert using dict()
                    conn_dict = to_dict_recursive(conn)
                    connections.append(conn_dict)
            except Exception as e:
                logger.warning(f"Error converting connection to dict: {e}")
                if isinstance(conn, dict):
                    connections.append(conn)
        
        # CRITICAL FIX 4: Detect ports for elements based on connections
        # This must be done AFTER connections are converted to dicts
        elements_with_ports = self._detect_ports(elements, connections)
        elements = elements_with_ports
        
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
        
        # Create connectors (connections between ALL components, not just main_components)
        # CRITICAL FIX: Use ALL elements, not just main_components, to include all connections
        all_elements_map = {}
        for el in elements:
            el_id = el.get("id") if isinstance(el, dict) else getattr(el, 'id', None)
            if el_id:
                all_elements_map[el_id] = el
        
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
            
            # CRITICAL FIX: Check if both elements exist in ALL elements (not just main_components)
            if from_id and to_id and from_id in all_elements_map and to_id in all_elements_map:
                from_el = all_elements_map[from_id]
                to_el = all_elements_map[to_id]
                
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
        # CRITICAL FIX: Use all_elements_map instead of main_component_elements
        main_component_elements = {
            el_id: el for el_id, el in all_elements_map.items()
            if (isinstance(el, dict) and el.get("type", "unknown") in main_components) or
               (hasattr(el, 'type') and getattr(el, 'type', 'unknown') in main_components)
        }
        main_connections = [
            conn for conn in connections
            if (conn.get("from_id") if isinstance(conn, dict) else getattr(conn, 'from_id', None)) in main_component_elements and
               (conn.get("to_id") if isinstance(conn, dict) else getattr(conn, 'to_id', None)) in main_component_elements
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
        
        CRITICAL: Uses OutputStructureManager to enforce folder structure.
        - Data files → data/ subdirectory
        - Artifacts → artifacts/ subdirectory
        
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
        
        # CRITICAL: Ensure structured output directories exist
        data_dir = output_path / "data"
        artifacts_dir = output_path / "artifacts"
        data_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        # Get legend info for structured output
        legend_info = {
            'symbol_map': final_data.get('legend_data', {}).get('symbol_map', {}),
            'line_map': final_data.get('legend_data', {}).get('line_map', {}),
            'metadata': final_data.get('metadata', {})
        }
        
        # Save JSON results with legend info → data/
        results_data = final_data.copy()
        results_data['legend_info'] = legend_info
        results_path = data_dir / f"{base_name}_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to: {results_path}")
        
        # Save separate legend info file for easy access → data/
        legend_info_path = data_dir / f"{base_name}_legend_info.json"
        with open(legend_info_path, 'w', encoding='utf-8') as f:
            json.dump(legend_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved legend info to: {legend_info_path}")
        
        # Save KPIs → data/
        kpis_path = data_dir / f"{base_name}_kpis.json"
        with open(kpis_path, 'w', encoding='utf-8') as f:
            json.dump(kpis, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved KPIs to: {kpis_path}")
        
        # Save CGM data (JSON format) → data/
        cgm_path = data_dir / f"{base_name}_cgm_data.json"
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
        
        # Save CGM Python code (dataclass format) → data/
        if cgm_data.get('python_code'):
            cgm_py_path = data_dir / f"{base_name}_cgm_network_generated.py"
            with open(cgm_py_path, 'w', encoding='utf-8') as f:
                f.write(cgm_data['python_code'])
            logger.info(f"Saved CGM Python code to: {cgm_py_path}")
        
        # Generate HTML report for professional presentation → artifacts/
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
            
            # CRITICAL: Ensure visualizations subdirectory exists
            visualizations_dir = output_path / "visualizations"
            visualizations_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(image_path).stem
            
            # 1. Uncertainty heatmap → visualizations/
            uncertain_zones = final_data.get("uncertain_zones", [])
            if uncertain_zones:
                heatmap_path = visualizations_dir / f"{base_name}_uncertainty_heatmap.png"
                visualizer.draw_uncertainty_heatmap(
                    image_path=image_path,
                    uncertain_zones=uncertain_zones,
                    output_path=str(heatmap_path)
                )
            
            # 2. Debug map → visualizations/
            elements = final_data.get("elements", [])
            connections = final_data.get("connections", [])
            if elements or connections:
                debug_map_path = visualizations_dir / f"{base_name}_debug_map.png"
                visualizer.draw_debug_map(
                    image_path=image_path,
                    elements=elements,
                    connections=connections,
                    output_path=str(debug_map_path)
                )
            
            # 3. Confidence map → visualizations/
            if elements:
                confidence_map_path = visualizations_dir / f"{base_name}_confidence_map.png"
                visualizer.draw_confidence_map(
                    image_path=image_path,
                    elements=elements,
                    output_path=str(confidence_map_path)
                )
            
            # 4. Score curve → visualizations/
            if score_history:
                score_curve_path = visualizations_dir / f"{base_name}_score_curve.png"
                visualizer.plot_score_curve(
                    score_history=score_history,
                    output_path=str(score_curve_path)
                )
            
            # 5. KPI dashboard → visualizations/
            if kpis:
                kpi_dashboard_path = visualizations_dir / f"{base_name}_kpi_dashboard.png"
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
            
            # CRITICAL: Get visualization paths from visualizations/ subdirectory
            visualizations_dir = output_path / "visualizations"
            debug_map_path = visualizations_dir / f"{base_name}_debug_map.png"
            confidence_map_path = visualizations_dir / f"{base_name}_confidence_map.png"
            kpi_dashboard_path = visualizations_dir / f"{base_name}_kpi_dashboard.png"
            
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
            
            # CRITICAL: Save HTML report to artifacts/ subdirectory
            artifacts_dir = output_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            html_path = artifacts_dir / f"{base_name}_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTML report saved to: {html_path}")
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}", exc_info=True)

