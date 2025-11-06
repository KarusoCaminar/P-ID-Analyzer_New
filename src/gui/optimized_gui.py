"""
Optimized GUI - Flotte, responsive GUI mit Threading und Performance-Optimierungen.

Features:
- Non-blocking UI (Threading)
- Progress Updates (Queue-based)
- Performance-Optimierungen
- Responsive Design
- Modern Layout
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import queue
import threading
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta

from src.analyzer.core.pipeline_coordinator import PipelineCoordinator, ProgressCallback
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService

logger = logging.getLogger(__name__)


class GUILogHandler(logging.Handler):
    """
    Custom logging handler that forwards all log messages to the GUI.
    
    Thread-safe implementation using a queue to forward messages.
    """
    
    def __init__(self, queue_update_func: Callable):
        """
        Initialize GUI log handler.
        
        Args:
            queue_update_func: Function to queue GUI updates (thread-safe)
        """
        super().__init__()
        self.queue_update_func = queue_update_func
        self.setFormatter(logging.Formatter(
            '[%(asctime)s - %(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the GUI.
        
        Args:
            record: Log record to emit
        """
        try:
            message = self.format(record)
            # Map logging levels to GUI levels
            level_map = {
                logging.DEBUG: 'INFO',
                logging.INFO: 'INFO',
                logging.WARNING: 'WARNING',
                logging.ERROR: 'ERROR',
                logging.CRITICAL: 'ERROR'
            }
            gui_level = level_map.get(record.levelno, 'INFO')
            # Queue the log message for GUI update
            self.queue_update_func('log', gui_level, message)
        except Exception:
            # Ignore errors in log handler to prevent infinite loops
            pass


class OptimizedProgressCallback(ProgressCallback):
    """Optimized progress callback with queue-based updates."""
    
    def __init__(self, queue_update_func: Callable):
        """
        Initialize progress callback.
        
        Args:
            queue_update_func: Function to queue GUI updates (thread-safe)
        """
        self.queue_update_func = queue_update_func
        self._updates_pending = []
    
    def update_progress(self, value: int, message: str) -> None:
        """Update progress (0-100)."""
        def _update():
            # This will be called in main thread
            pass  # Implement in GUI class
        self.queue_update_func('update_progress', value, message)
    
    def update_status_label(self, text: str) -> None:
        """Update status message."""
        self.queue_update_func('update_status', text)
    
    def report_truth_mode(self, active: bool) -> None:
        """Report truth mode status."""
        self.queue_update_func('report_truth_mode', active)
    
    def report_correction(self, correction_text: str) -> None:
        """Report correction information."""
        self.queue_update_func('report_correction', correction_text)


class OptimizedGUI(tk.Tk):
    """
    Optimized GUI with performance improvements.
    
    Features:
    - Non-blocking operations (Threading)
    - Queue-based updates (Thread-safe)
    - Performance optimizations
    - Responsive UI
    """
    
    def __init__(self):
        """Initialize optimized GUI."""
        super().__init__()
        
        self.title("P&ID Analyzer v2.0 - Optimized")
        self.geometry("1400x900")
        
        # Thread-safe queue for GUI updates
        self.gui_queue = queue.Queue()
        
        # GUI log handler for automatic logging
        self.gui_log_handler = None
        self.file_log_handler = None  # File log handler
        
        # Initialize backend components
        self.config_service = ConfigService()
        self.llm_client = None
        self.knowledge_manager = None
        self.pipeline_coordinator = None
        
        # GUI state
        self.is_processing = False
        self.current_results = None
        self.results_text = None
        self.status_label = None
        self.progress_bar = None
        self.progress_var = None
        self.progress_time_label = None  # Time estimation label
        self.truth_mode_button = None  # Truth Mode button
        self.log_text = None
        self.start_button = None
        self.file_listbox = None
        self.use_monolith_var = None
        self.use_predictive_var = None
        self.use_polyline_var = None
        self.use_self_correction_var = None
        self.duration_var = None
        self.max_cycles_var = None
        self.training_status_text = None
        self.model_selections = {}  # Store model selections per phase
        
        # Time tracking for ETA
        self.phase_start_time = None
        self.phase_durations = {}  # Track phase durations
        self.current_phase = None
        
        # New features for analysis
        self.use_skeleton_extraction_var = tk.BooleanVar(value=False)
        self.use_topology_critic_var = tk.BooleanVar(value=True)
        self.use_legend_consistency_critic_var = tk.BooleanVar(value=True)
        self.use_two_pass_var = tk.BooleanVar(value=False)
        
        # Model Strategy and Self-Correction Parameters
        self.model_strategy_var = None  # StringVar for strategy dropdown
        self.max_iterations_var = None  # IntVar for max iterations slider
        self.early_stop_threshold_var = None  # DoubleVar for early-stop threshold slider
        self.max_iterations_label = None  # Label for max iterations display
        self.early_stop_threshold_label = None  # Label for early-stop threshold display
        
        # Setup
        try:
            self._create_ui()
            self._setup_gui_logging()
            self._initialize_backend()
            self._start_queue_processor()
            
            # Register cleanup on window close
            self.protocol("WM_DELETE_WINDOW", self._on_closing)
            
            logger.info("Optimized GUI initialized")
        except Exception as e:
            logger.error(f"Error initializing GUI: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to initialize GUI: {e}")
    
    def _setup_gui_logging(self):
        """Setup GUI logging handler to capture all log messages and save to file."""
        try:
            from pathlib import Path
            from datetime import datetime
            
            # Create log directory if it doesn't exist
            log_dir = Path("outputs/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file log handler - save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"gui_log_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            
            # Create GUI log handler
            self.gui_log_handler = GUILogHandler(self._queue_update)
            self.gui_log_handler.setLevel(logging.DEBUG)
            
            # Add both handlers to root logger to capture all logs
            root_logger = logging.getLogger()
            root_logger.addHandler(self.gui_log_handler)
            root_logger.addHandler(file_handler)
            
            # Store file handler for cleanup
            self.file_log_handler = file_handler
            
            # Log initial message
            logger.info(f"GUI logging handler configured - Logs saved to: {log_file}")
            self._log_message(f"✓ Logging aktiviert - Logs werden gespeichert in: {log_file}", 'SUCCESS')
            
        except Exception as e:
            logger.error(f"Error setting up GUI logging: {e}", exc_info=True)
            # Try to show error in GUI
            try:
                self._log_message(f"ERROR: Logging setup failed: {e}", 'ERROR')
            except:
                pass
    
    def _initialize_backend(self):
        """Initialize backend components with robust error handling."""
        try:
            import os
            from pathlib import Path
            from dotenv import load_dotenv
            
            # Load .env file if it exists
            load_dotenv()
            
            config = self.config_service.get_raw_config() or {}
            
            # Initialize LLM client
            project_id = config.get('gcp_project_id') or os.getenv('GCP_PROJECT_ID')
            location = config.get('gcp_location') or os.getenv('GCP_LOCATION', 'us-central1')
            
            if not project_id:
                logger.warning("GCP_PROJECT_ID not found - backend features will be limited")
                self._log_message("WARNING: GCP_PROJECT_ID not found. Set in .env file to use full features.", 'WARNING')
                # Don't initialize backend, but GUI can start with limited functionality
                self.pipeline_coordinator = None
                return
            
            try:
                self.llm_client = LLMClient(project_id, location, config)
                logger.info("LLMClient initialized successfully")
                self._log_message("LLMClient initialized successfully", 'SUCCESS')
            except Exception as e:
                logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
                self._log_message(f"ERROR: LLMClient initialization failed: {e}", 'ERROR')
                return
            
            # Initialize Knowledge Manager
            element_type_list_path = config.get('paths', {}).get('element_type_list', 'element_type_list.json')
            learning_db_path = config.get('paths', {}).get('learning_db', 'learning_db.json')
            
            # Convert to Path objects and ensure they exist
            if isinstance(element_type_list_path, str):
                element_type_list_path = Path(element_type_list_path)
            if isinstance(learning_db_path, str):
                learning_db_path = Path(learning_db_path)
            
            # Check if paths exist
            if not element_type_list_path.exists():
                logger.warning(f"Element type list not found: {element_type_list_path}")
                self._log_message(f"WARNING: Element type list not found: {element_type_list_path}", 'WARNING')
            
            try:
                self.knowledge_manager = KnowledgeManager(
                    element_type_list_path=str(element_type_list_path),
                    learning_db_path=str(learning_db_path),
                    llm_handler=self.llm_client,  # KnowledgeManager uses llm_handler parameter
                    config=config
                )
                logger.info("KnowledgeManager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize KnowledgeManager: {e}", exc_info=True)
                self._log_message(f"ERROR: KnowledgeManager initialization failed: {e}", 'ERROR')
                return
            
            # Initialize Pipeline Coordinator
            try:
                self.pipeline_coordinator = PipelineCoordinator(
                    llm_client=self.llm_client,
                    knowledge_manager=self.knowledge_manager,
                    config_service=self.config_service,
                    progress_callback=None  # Will be set per analysis
                )
                logger.info("PipelineCoordinator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PipelineCoordinator: {e}", exc_info=True)
                self._log_message(f"ERROR: PipelineCoordinator initialization failed: {e}", 'ERROR')
                return
            
            logger.info("Backend components initialized successfully")
            self._log_message("Backend initialized successfully", 'SUCCESS')
            
        except Exception as e:
            logger.error(f"Error initializing backend: {e}", exc_info=True)
            self._log_message(f"ERROR: Backend initialization failed: {e}", 'ERROR')
            # Don't show messagebox here to avoid blocking GUI start
    
    def _create_ui(self):
        """Create optimized UI layout."""
        # Main container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top: Status and Progress
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Truth Mode Button (prominent)
        truth_frame = ttk.Frame(status_frame)
        truth_frame.pack(side=tk.LEFT, padx=5)
        
        self.truth_mode_button = tk.Button(
            truth_frame,
            text="❌ Truth Mode: OFF",
            font=("Arial", 10, "bold"),
            bg="#ff4444",
            fg="white",
            relief=tk.RAISED,
            bd=2,
            state=tk.DISABLED  # Only visual indicator
        )
        self.truth_mode_button.pack()
        self._create_tooltip(self.truth_mode_button, 
                           "Truth Mode Status:\n\n"
                           "🟢 ON: Analysiert mit Truth-Daten (Trainingsmodus)\n"
                           "🔴 OFF: Analysiert ohne Truth-Daten (Produktionsmodus)\n\n"
                           "Truth-Modus ermöglicht automatisches Lernen aus Fehlern.")
        
        # Status label
        self.status_label = ttk.Label(status_frame, text="Bereit", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar with time estimation
        progress_frame = ttk.Frame(status_frame)
        progress_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=300,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 2))
        
        # Time estimation label
        self.progress_time_label = ttk.Label(
            progress_frame,
            text="",
            font=("Arial", 8),
            foreground="gray"
        )
        self.progress_time_label.pack()
        
        # Middle: Main content with Notebook
        notebook_frame = ttk.Frame(main_frame)
        notebook_frame.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tabs
        self._create_analysis_tab()
        self._create_training_tab()
        self._create_visualization_tab()
        self._create_info_tab()  # Add Info tab
        
        # Bottom: Log - BIGGER and more visible
        log_frame = ttk.LabelFrame(main_frame, text="Live Log (Alle Ausgaben werden hier angezeigt)", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=15,  # Increased from 8 to 15
            wrap=tk.WORD,
            state='disabled',
            font=("Courier New", 9),
            bg='#1e1e1e',
            fg='#ffffff',
            insertbackground='white'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure log tags with improved colors
        self.log_text.tag_config('INFO', foreground='#a0a0a0', background='#2e2e2e')
        self.log_text.tag_config('WARNING', foreground='#FFA500', background='#2e2e2e')  # Orange
        self.log_text.tag_config('ERROR', foreground='#FF0000', background='#2e2e2e')  # Red
        self.log_text.tag_config('SUCCESS', foreground='#00FF00', background='#2e2e2e')  # Green
        self.log_text.tag_config('PHASE', foreground='#4A9EFF', background='#2e2e2e')  # Blue for phases
    
    def _create_analysis_tab(self):
        """Create analysis tab."""
        analysis_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(analysis_frame, text=" Analyse ")
        
        # File selection
        file_frame = ttk.LabelFrame(analysis_frame, text="Datei-Auswahl", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_listbox = tk.Listbox(file_frame, height=6, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        file_buttons = ttk.Frame(file_frame)
        file_buttons.pack(side=tk.RIGHT)
        
        add_btn = ttk.Button(file_buttons, text="Dateien hinzufügen", command=self._add_files)
        add_btn.pack(pady=2, fill=tk.X)
        self._create_tooltip(add_btn, "Dateien hinzufügen:\n\nFügt P&ID-Bilder zur Analyse-Liste hinzu.\nUnterstützt: PNG, JPG, JPEG")
        
        remove_btn = ttk.Button(file_buttons, text="Dateien entfernen", command=self._remove_files)
        remove_btn.pack(pady=2, fill=tk.X)
        self._create_tooltip(remove_btn, "Dateien entfernen:\n\nEntfernt die ausgewählten Dateien aus der Liste.")
        
        clear_btn = ttk.Button(file_buttons, text="Alle löschen", command=self._clear_files)
        clear_btn.pack(pady=2, fill=tk.X)
        self._create_tooltip(clear_btn, "Alle löschen:\n\nEntfernt alle Dateien aus der Liste.")
        
        # Options
        options_frame = ttk.LabelFrame(analysis_frame, text="Optionen", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.use_monolith_var = tk.BooleanVar(value=True)
        self.use_predictive_var = tk.BooleanVar(value=True)
        self.use_polyline_var = tk.BooleanVar(value=True)
        self.use_self_correction_var = tk.BooleanVar(value=True)
        
        monolith_cb = ttk.Checkbutton(options_frame, text="Monolith Fusion", variable=self.use_monolith_var)
        monolith_cb.grid(row=0, column=0, sticky=tk.W, padx=5)
        self._create_tooltip(monolith_cb, "Monolith Fusion:\n\nKombiniert Swarm- und Monolith-Analyse.\nMonolith-Analyse erkennt große Strukturen,\nSwarm-Analyse erkennt Details.\nEmpfohlen: Aktiviert")
        
        predictive_cb = ttk.Checkbutton(options_frame, text="Predictive Completion", variable=self.use_predictive_var)
        predictive_cb.grid(row=0, column=1, sticky=tk.W, padx=5)
        self._create_tooltip(predictive_cb, "Predictive Completion:\n\nVervollständigt fehlende Verbindungen\nbasierend auf geometrischen Heuristiken.\nEmpfohlen: Aktiviert")
        
        polyline_cb = ttk.Checkbutton(options_frame, text="Polyline Refinement", variable=self.use_polyline_var)
        polyline_cb.grid(row=1, column=0, sticky=tk.W, padx=5)
        self._create_tooltip(polyline_cb, "Polyline Refinement:\n\nVerfeinert Rohrleitungs-Verbindungen\nmit präzisen Polylinien.\nEmpfohlen: Aktiviert")
        
        selfcorrection_cb = ttk.Checkbutton(options_frame, text="Self-Correction", variable=self.use_self_correction_var)
        selfcorrection_cb.grid(row=1, column=1, sticky=tk.W, padx=5)
        self._create_tooltip(selfcorrection_cb, "Self-Correction:\n\nIterative Selbstkorrektur-Schleife.\nAnalysiert Fehler und korrigiert automatisch.\nEmpfohlen: Aktiviert (kann länger dauern)")
        
        # Model Strategy Selection
        strategy_frame = ttk.LabelFrame(analysis_frame, text="Model-Strategie", padding="10")
        strategy_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(strategy_frame, text="Strategie:").pack(side=tk.LEFT, padx=5)
        self.model_strategy_var = tk.StringVar(value="simple_pid_strategy")
        strategy_combo = ttk.Combobox(
            strategy_frame,
            textvariable=self.model_strategy_var,
            values=["simple_pid_strategy", "all_flash", "optimal_swarm_monolith"],
            state='readonly',
            width=30
        )
        strategy_combo.pack(side=tk.LEFT, padx=5)
        self._create_tooltip(strategy_combo, "Model-Strategie:\n\n• simple_pid_strategy: Alle Modelle = Gemini 2.5 Flash (schnell + gut)\n• all_flash: Alle Modelle = Gemini 2.5 Flash (schnell)\n• optimal_swarm_monolith: Swarm=Flash, Monolith=Pro (balanciert)")
        
        # Strategy test button
        test_strategy_btn = ttk.Button(
            strategy_frame,
            text="Strategie testen",
            command=self._test_strategy,
            style="Accent.TButton"
        )
        test_strategy_btn.pack(side=tk.LEFT, padx=5)
        self._create_tooltip(test_strategy_btn, "Strategie testen:\n\nTestet die ausgewählte Strategie mit einem Test-Bild.\nVerwendet automatisch ein Test-Bild aus training_data/simple_pids/\noder fordert zur Auswahl eines Bildes auf.\nZeigt Ergebnisse in der Log-Anzeige.")
        
        # Self-Correction Parameters
        correction_params_frame = ttk.LabelFrame(analysis_frame, text="Self-Correction Parameter", padding="10")
        correction_params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Max Iterations Slider
        max_iter_frame = ttk.Frame(correction_params_frame)
        max_iter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(max_iter_frame, text="Max Iterationen:").pack(side=tk.LEFT, padx=5)
        self.max_iterations_var = tk.IntVar(value=5)
        max_iter_slider = ttk.Scale(
            max_iter_frame,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.max_iterations_var,
            length=200
        )
        max_iter_slider.pack(side=tk.LEFT, padx=5)
        self.max_iterations_label = ttk.Label(max_iter_frame, text="5")
        self.max_iterations_label.pack(side=tk.LEFT, padx=5)
        max_iter_slider.configure(command=lambda v: self.max_iterations_label.config(text=str(int(float(v)))))
        self._create_tooltip(max_iter_slider, "Max Iterationen (1-10):\n\nMaximale Anzahl der Self-Correction-Iterationen.\nEmpfohlen: 3-5 für Simple P&IDs, 5-10 für komplexe Diagramme")
        
        # Early-Stop Threshold Slider
        early_stop_frame = ttk.Frame(correction_params_frame)
        early_stop_frame.pack(fill=tk.X, pady=5)
        ttk.Label(early_stop_frame, text="Early-Stop Threshold:").pack(side=tk.LEFT, padx=5)
        self.early_stop_threshold_var = tk.DoubleVar(value=80.0)
        early_stop_slider = ttk.Scale(
            early_stop_frame,
            from_=70.0,
            to=95.0,
            orient=tk.HORIZONTAL,
            variable=self.early_stop_threshold_var,
            length=200
        )
        early_stop_slider.pack(side=tk.LEFT, padx=5)
        self.early_stop_threshold_label = ttk.Label(early_stop_frame, text="80.0%")
        self.early_stop_threshold_label.pack(side=tk.LEFT, padx=5)
        early_stop_slider.configure(command=lambda v: self.early_stop_threshold_label.config(text=f"{float(v):.1f}%"))
        self._create_tooltip(early_stop_slider, "Early-Stop Threshold (70-95%):\n\nBeendet die Self-Correction-Schleife vorzeitig,\nwenn der Quality Score diesen Wert erreicht.\nEmpfohlen: 80% für Simple P&IDs, 90% für komplexe Diagramme")
        
        # NEW: Advanced Features Section
        advanced_frame = ttk.LabelFrame(analysis_frame, text="Erweiterte Features", padding="10")
        advanced_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1
        skeleton_cb = ttk.Checkbutton(advanced_frame, text="Skeleton Line Extraction", variable=self.use_skeleton_extraction_var)
        skeleton_cb.grid(row=0, column=0, sticky=tk.W, padx=5)
        self._create_tooltip(skeleton_cb, "Skeleton Line Extraction:\n\nTrennt Pipeline-Linien von Symbol-Linien\nmit Skeletonization.\nVorteil: Präzisere Polylines, keine Symbol-Verwechslung\nNachteil: +5-10 Sekunden Verarbeitungszeit\nEmpfohlen: Für komplexe Diagramme")
        
        topology_cb = ttk.Checkbutton(advanced_frame, text="Topology Critic", variable=self.use_topology_critic_var)
        topology_cb.grid(row=0, column=1, sticky=tk.W, padx=5)
        self._create_tooltip(topology_cb, "Topology Critic:\n\nValidiert Graph-Konsistenz:\n- Disconnected nodes\n- Invalid degrees\n- Missing splits/merges\nVorteil: Systematische Topologie-Validierung\nEmpfohlen: Aktiviert")
        
        legend_cb = ttk.Checkbutton(advanced_frame, text="Legend Consistency Critic", variable=self.use_legend_consistency_critic_var)
        legend_cb.grid(row=1, column=0, sticky=tk.W, padx=5)
        self._create_tooltip(legend_cb, "Legend Consistency Critic:\n\nPrüft Konsistenz zwischen Legende\nund erkannten Symbolen:\n- Missing symbols\n- Unexpected symbols\n- Frequency anomalies\nVorteil: Systematische Legend-Validierung\nEmpfohlen: Aktiviert")
        
        two_pass_cb = ttk.Checkbutton(advanced_frame, text="Two-Pass Pipeline", variable=self.use_two_pass_var)
        two_pass_cb.grid(row=1, column=1, sticky=tk.W, padx=5)
        self._create_tooltip(two_pass_cb, "Two-Pass Pipeline:\n\nCoarse → Refine Strategie für große Bilder:\nPass 1: Große Kacheln (1024px) für Übersicht\nPass 2: Kleine Kacheln (512px) für unsichere Bereiche\nVorteil: -60% Kacheln, -40% Zeit für große Bilder\nEmpfohlen: Für Bilder >4000px")
        
        # Model selection per phase
        self._create_model_selection_frame(analysis_frame)
        
        # Actions
        action_frame = ttk.Frame(analysis_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(
            action_frame,
            text="Analyse starten",
            command=self._start_analysis,
            style="Accent.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        self._create_tooltip(self.start_button, "Analyse starten:\n\nStartet die Analyse aller ausgewählten Dateien.\nDie Analyse durchläuft mehrere Phasen:\n1. Pre-Analysis (Metadata, Legende)\n2. Swarm & Monolith Analyse\n3. Self-Correction (optional)\n4. Post-Processing")
        
        cancel_btn = ttk.Button(action_frame, text="Abbrechen", command=self._cancel_analysis)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        self._create_tooltip(cancel_btn, "Abbrechen:\n\nBricht die laufende Analyse ab.\nKann einige Sekunden dauern.")
        
        results_btn = ttk.Button(action_frame, text="Ergebnisse anzeigen", command=self._show_results)
        results_btn.pack(side=tk.LEFT, padx=5)
        self._create_tooltip(results_btn, "Ergebnisse anzeigen:\n\nZeigt die Ergebnisse der letzten Analyse\nin einem separaten Fenster.")
        
        # Results preview
        results_frame = ttk.LabelFrame(analysis_frame, text="Ergebnisse", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, state='disabled')
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_model_selection_frame(self, parent):
        """Create model selection frame with dropdowns for each phase."""
        model_frame = ttk.LabelFrame(parent, text="Model-Auswahl pro Phase", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load available models from config - use get_raw_config() to get dict
        try:
            config = self.config_service.get_raw_config()
            models_config = config.get('models', {})
            
            # Handle Pydantic models - convert to dict if needed
            if hasattr(models_config, 'model_dump'):
                models_config = models_config.model_dump()
            elif hasattr(models_config, 'dict'):
                models_config = models_config.dict()
            
            available_models = list(models_config.keys()) if isinstance(models_config, dict) else []
            
            # Log for debugging
            logger.info(f"Loaded {len(available_models)} models from config: {available_models}")
            self._log_message(f"Loaded {len(available_models)} models from config", 'INFO')
        except Exception as e:
            logger.error(f"Could not load models from config: {e}", exc_info=True)
            self._log_message(f"ERROR: Could not load models: {e}", 'ERROR')
            available_models = []
        
        if not available_models:
            error_label = ttk.Label(model_frame, text="Keine Modelle in Config gefunden - Prüfe config.yaml", foreground="red")
            error_label.pack()
            self._log_message("WARNING: No models found in config.yaml - check models section", 'WARNING')
            return
        
        # Get default strategy from config
        try:
            config = self.config_service.get_config()
            strategies = config.strategies if hasattr(config, 'strategies') else config.get('strategies', {})
            default_strategy = strategies.get('default_flash', {}) if isinstance(strategies, dict) else {}
        except Exception as e:
            logger.warning(f"Could not load default strategy: {e}")
            default_strategy = {}
        
        # Phase definitions
        phases = [
            ('meta_model', 'Meta Model'),
            ('hotspot_model', 'Hotspot Model'),
            ('detail_model', 'Detail Model'),
            ('coarse_model', 'Coarse Model'),
            ('correction_model', 'Correction Model'),
            ('code_gen_model', 'Code Gen Model'),
            ('critic_model_name', 'Critic Model')
        ]
        
        # Create dropdowns in grid layout
        row = 0
        col = 0
        
        for phase_key, phase_label in phases:
            # Label
            ttk.Label(model_frame, text=f"{phase_label}:").grid(
                row=row, column=col*2, sticky=tk.W, padx=5, pady=2
            )
            
            # Dropdown
            var = tk.StringVar()
            # Get default value from strategy
            default_model = default_strategy.get(phase_key, available_models[0] if available_models else "")
            var.set(default_model)
            self.model_selections[phase_key] = var
            
            combo = ttk.Combobox(
                model_frame,
                textvariable=var,
                values=available_models,
                state='readonly',
                width=30
            )
            combo.grid(row=row, column=col*2+1, sticky=tk.W, padx=5, pady=2)
            
            # Add tooltip with model description
            try:
                config = self.config_service.get_config()
                models_config = config.models if hasattr(config, 'models') else config.get('models', {})
                if isinstance(models_config, dict) and default_model in models_config:
                    model_info = models_config[default_model]
                    description = model_info.get('description', '') if isinstance(model_info, dict) else ''
                    if description:
                        self._create_tooltip(combo, f"{phase_label}:\n\n{description}")
            except Exception:
                pass  # Ignore errors in tooltip creation
            
            # Advance to next position
            col += 1
            if col >= 2:  # 2 columns
                col = 0
                row += 1
    
    def _create_training_tab(self):
        """Create training tab."""
        training_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(training_frame, text=" Training ")
        
        info_label = ttk.Label(
            training_frame,
            text="Training Camp: Automatisches Training mit Strategien- und Parameter-Testing",
            font=("Arial", 10, "bold")
        )
        info_label.pack(pady=10)
        
        # Training options
        options_frame = ttk.LabelFrame(training_frame, text="Training-Optionen", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        duration_frame = ttk.Frame(options_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        ttk.Label(duration_frame, text="Dauer (Stunden):").pack(side=tk.LEFT, padx=5)
        self.duration_var = tk.DoubleVar(value=24.0)
        ttk.Spinbox(duration_frame, from_=1.0, to=168.0, textvariable=self.duration_var, width=10).pack(side=tk.LEFT, padx=5)
        
        cycles_frame = ttk.Frame(options_frame)
        cycles_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cycles_frame, text="Max Zyklen (0=unbegrenzt):").pack(side=tk.LEFT, padx=5)
        self.max_cycles_var = tk.IntVar(value=0)
        ttk.Spinbox(cycles_frame, from_=0, to=1000, textvariable=self.max_cycles_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Pretraining section
        pretraining_frame = ttk.LabelFrame(training_frame, text="Symbol Pretraining", padding="10")
        pretraining_frame.pack(fill=tk.X, pady=(0, 10))
        
        pretraining_info = ttk.Label(
            pretraining_frame,
            text="Verarbeitet alle Symbole aus pretraining_symbols/ Verzeichnis:\n"
                 "- Extrahiert Symbole aus PDF-Sammlungen automatisch\n"
                 "- Verarbeitet einzelne Symbol-Bilder\n"
                 "- Prüft auf Duplikate via Embedding-Similarity\n"
                 "- Speichert in Symbol-Library für schnelle Erkennung",
            font=("Arial", 9)
        )
        pretraining_info.pack(pady=5)
        
        ttk.Button(
            pretraining_frame,
            text="Pretraining starten",
            command=self._start_pretraining,
            style="Accent.TButton"
        ).pack(pady=10)
        self._create_tooltip(
            pretraining_frame,
            "Pretraining:\n\nVerarbeitet alle Symbole aus pretraining_symbols/\n"
            "- PDF-Sammlungen werden automatisch segmentiert\n"
            "- Einzelne Symbole werden direkt verarbeitet\n"
            "- Duplikate werden erkannt und vermieden\n"
            "- Verbessert die Erkennungsgenauigkeit erheblich"
        )
        
        # Start training button
        ttk.Button(
            training_frame,
            text="Training Camp starten",
            command=self._start_training_camp,
            style="Accent.TButton"
        ).pack(pady=20)
        
        # Training status
        self.training_status_text = scrolledtext.ScrolledText(training_frame, height=15, state='disabled')
        self.training_status_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
    
    def _create_visualization_tab(self):
        """Create visualization tab."""
        viz_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(viz_frame, text=" Visualisierungen ")
        
        info_label = ttk.Label(
            viz_frame,
            text="Visualisierungen werden automatisch bei jeder Analyse generiert",
            font=("Arial", 10)
        )
        info_label.pack(pady=10)
        
        # Visualization list
        viz_list_frame = ttk.LabelFrame(viz_frame, text="Verfügbare Visualisierungen", padding="10")
        viz_list_frame.pack(fill=tk.BOTH, expand=True)
        
        viz_list = [
            "- Uncertainty Heatmap (uncertainty_heatmap.png)",
            "- Debug Map (debug_map.png)",
            "- Confidence Map (confidence_map.png)",
            "- Score Curve (score_curve.png)",
            "- KPI Dashboard (kpi_dashboard.png)"
        ]
        
        for viz_item in viz_list:
            ttk.Label(viz_list_frame, text=viz_item).pack(anchor=tk.W, pady=2)
    
    def _create_info_tab(self):
        """Create info/help tab with explanations."""
        info_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(info_frame, text=" ℹ️ Info ")
        
        # Create scrollable text widget
        info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, font=("Arial", 10))
        info_text.pack(fill=tk.BOTH, expand=True)
        
        # Info content
        info_content = """
═══════════════════════════════════════════════════════════════════════════════
                    P&ID ANALYZER - BENUTZERHANDBUCH
═══════════════════════════════════════════════════════════════════════════════

📋 INHALT
  1. Analyse-Phasen
  2. Modelle & Strategien
  3. Features & Optionen
  4. Erweiterte Features
  5. Tipps & Best Practices

───────────────────────────────────────────────────────────────────────────────
1. ANALYSE-PHASEN
───────────────────────────────────────────────────────────────────────────────

Die Analyse durchläuft mehrere Phasen:

Phase 1: Pre-Analysis
  • Metadata-Extraktion (Projekt, Titel, Version, Datum)
  • Legend-Erkennung (Symbol-Map, Line-Map)
  • Excluded Zones (Bereiche, die nicht analysiert werden sollen)

Phase 2: Parallel Core Analysis
  • Swarm Analysis: Tile-basierte Detail-Analyse (hohe Precision)
    → Erkennt kleine Details, Symbole, Verbindungen
    → Optimiert für komplexe Diagramme
    → Dynamische Tile-Anzahl basierend auf Bildgröße
  
  • Monolith Analysis: Globale Struktur-Analyse (hohe Recall)
    → Erkennt große Strukturen, Zusammenhänge
    → Optimiert für Übersicht
    → Quadrant-basierte Analyse mit Überlappung

Phase 2c: Fusion
  • Kombiniert Swarm- und Monolith-Ergebnisse
  • Deduplizierung (IoU-basiert)
  • Confidence-Propagation (Verbindungs-Confidence aus Element-Confidence)

Phase 2d: Predictive Completion
  • Vervollständigt fehlende Verbindungen
  • Geometrische Heuristiken (Distanz, Position)
  • Empfohlen: Aktiviert für vollständige Graphen

Phase 2e: Polyline Refinement
  • Extrahiert präzise Polylinien für Verbindungen
  • Option 1: LLM-basiert (Standard)
  • Option 2: Skeleton-basiert (präziser, aber langsamer)
  • Empfohlen: Aktiviert für präzise Koordinaten

Phase 3: Self-Correction Loop
  • Iterative Selbstkorrektur
  • Validiert Ergebnisse und findet Fehler
  • Re-Analyse problematischer Bereiche
  • Plateau-Erkennung (stoppt bei keinem Fortschritt)
  • Empfohlen: Aktiviert für hohe Qualität (kann länger dauern)

Phase 4: Post-Processing
  • KPI-Berechnung (Precision, Recall, F1-Score)
  • CGM-Generierung (Code-Generierung)
  • Visualisierungen (Confidence Maps, Debug Maps)
  • Active Learning (speichert gelernte Patterns)

───────────────────────────────────────────────────────────────────────────────
2. MODELLE & STRATEGIEN
───────────────────────────────────────────────────────────────────────────────

Verfügbare Modelle:
  • Google Gemini 2.5 Flash (schnell, gut für einfache Aufgaben)
  • Google Gemini 2.5 Pro (langsamer, aber präziser)
  • Google Gemini 1.5 Flash (sehr schnell, kosteneffizient)
  • Google Gemini 1.5 Pro (ausgewogen zwischen Geschwindigkeit und Qualität)
  
Alle Modelle werden über Google Vertex AI aufgerufen.

Empfohlene Strategien:

Schnell (Flash):
  • Alle Phasen: Gemini 2.5 Flash
  • Vorteil: Sehr schnell, günstig
  • Nachteil: Etwas weniger präzise
  • Empfohlen für: Einfache Diagramme, schnelle Tests

Balanced (Flash + Pro):
  • Pre-Analysis: Flash
  • Detail-Analyse: Pro
  • Kritiker: Pro
  • Vorteil: Gute Balance zwischen Geschwindigkeit und Qualität
  • Empfohlen für: Standard-Diagramme

Präzise (Pro):
  • Alle Phasen: Gemini 2.5 Pro
  • Vorteil: Höchste Präzision
  • Nachteil: Langsamer, teurer
  • Empfohlen für: Komplexe Diagramme, finale Analyse

───────────────────────────────────────────────────────────────────────────────
3. FEATURES & OPTIONEN
───────────────────────────────────────────────────────────────────────────────

Monolith Fusion:
  • Kombiniert Swarm- und Monolith-Analyse
  • Vorteil: Höhere Recall (verpasst weniger Elemente)
  • Empfohlen: Aktiviert

Predictive Completion:
  • Vervollständigt fehlende Verbindungen basierend auf Geometrie
  • Vorteil: Vollständigere Graphen
  • Empfohlen: Aktiviert

Polyline Refinement:
  • Extrahiert präzise Polylinien für Verbindungen
  • Vorteil: Exakte Koordinaten für Verbindungen
  • Empfohlen: Aktiviert

Self-Correction:
  • Iterative Selbstkorrektur-Schleife
  • Vorteil: Höhere Qualität durch iterative Verbesserung
  • Nachteil: Kann länger dauern (max. 15 Iterationen)
  • Empfohlen: Aktiviert für hohe Qualität

───────────────────────────────────────────────────────────────────────────────
4. ERWEITERTE FEATURES
───────────────────────────────────────────────────────────────────────────────

Skeleton Line Extraction:
  • Trennt Pipeline-Linien von Symbol-Linien mit Skeletonization
  • Vorteil: Präzisere Polylines, keine Symbol-Verwechslung
  • Nachteil: +5-10 Sekunden Verarbeitungszeit
  • Empfohlen: Für komplexe Diagramme mit vielen Symbolen

Topology Critic:
  • Validiert Graph-Konsistenz:
    - Disconnected nodes (isolierte Elemente)
    - Invalid degrees (unmögliche Verbindungsanzahl)
    - Missing splits/merges (fehlende Abzweige)
  • Vorteil: Systematische Topologie-Validierung
  • Gewicht im Quality Score: 20%
  • Empfohlen: Aktiviert

Legend Consistency Critic:
  • Prüft Konsistenz zwischen Legende und erkannten Symbolen:
    - Missing symbols (in Legende aber nicht erkannt)
    - Unexpected symbols (erkannt aber nicht in Legende)
    - Frequency anomalies (ungewöhnliche Häufigkeiten)
  • Vorteil: Systematische Legend-Validierung
  • Gewicht im Quality Score: 10%
  • Empfohlen: Aktiviert

Two-Pass Pipeline:
  • Coarse → Refine Strategie für große Bilder (>4000px):
    Pass 1: Große Kacheln (1024px) für Übersicht
    Pass 2: Kleine Kacheln (512px) nur für unsichere Bereiche
  • Vorteil: -60% Kacheln, -40% Zeit für große Bilder
  • Nachteil: Aktiviert nur für große Bilder
  • Empfohlen: Für Bilder >4000px

───────────────────────────────────────────────────────────────────────────────
5. TIPPS & BEST PRACTICES
───────────────────────────────────────────────────────────────────────────────

Für kleine Diagramme (<3000px):
  • Standard-Einstellungen sind optimal
  • Two-Pass Pipeline nicht nötig
  • Flash-Strategie ausreichend

Für mittlere Diagramme (3000-7000px):
  • Balanced-Strategie empfohlen
  • Self-Correction aktiviert
  • Topology & Legend Critics aktiviert

Für große Diagramme (>7000px):
  • Two-Pass Pipeline aktivieren
  • Pro-Strategie für Detail-Analyse
  • Skeleton Line Extraction für präzise Polylines
  • Alle Critics aktiviert

Für komplexe Diagramme (viele Symbole):
  • Skeleton Line Extraction aktivieren
  • Topology Critic aktiviert
  • Self-Correction mit mehr Iterationen

Für schnelle Tests:
  • Flash-Strategie
  • Self-Correction deaktiviert
  • Skeleton Line Extraction deaktiviert

Für finale Analyse:
  • Pro-Strategie
  • Alle Features aktiviert
  • Self-Correction aktiviert
  • Alle Critics aktiviert

───────────────────────────────────────────────────────────────────────────────
QUALITY SCORE GEWICHTUNG
───────────────────────────────────────────────────────────────────────────────

Der Quality Score wird aus mehreren Quellen berechnet:

  • 40% KPI-based Score (Precision, Recall, F1-Score)
  • 30% Multi-Model Critic (Struktur-Validierung)
  • 20% Topology Critic (Graph-Konsistenz)
  • 10% Legend Consistency Critic (Legend-Konsistenz)

Ziel: Score ≥ 85% für gute Qualität, ≥ 95% für exzellente Qualität

───────────────────────────────────────────────────────────────────────────────
HILFE & SUPPORT
───────────────────────────────────────────────────────────────────────────────

Bei Problemen:
  1. Prüfe Log-Ausgabe für Fehlerdetails
  2. Reduziere Complexity (deaktiviere Features)
  3. Verwende Flash-Strategie für schnelle Tests
  4. Prüfe Config-Parameter in config.yaml

Weitere Informationen:
  • README.md: Grundlegende Dokumentation
  • CHANGELOG.md: Änderungsprotokoll
  • FEATURE_IMPLEMENTATION_MEHRWERT.md: Feature-Details

═══════════════════════════════════════════════════════════════════════════════
"""
        
        info_text.insert('1.0', info_content)
        info_text.config(state='disabled')
    
    def _start_queue_processor(self):
        """Start processing GUI queue (non-blocking)."""
        try:
            # Process up to 10 tasks per cycle for performance
            processed = 0
            while processed < 10:
                try:
                    task = self.gui_queue.get_nowait()
                    self._process_gui_task(task)
                    processed += 1
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error processing GUI queue: {e}", exc_info=True)
        finally:
            self.after(50, self._start_queue_processor)  # Process every 50ms (20 FPS)
    
    def _process_gui_task(self, task):
        """Process a single GUI task."""
        try:
            if isinstance(task, tuple) and len(task) > 0:
                task_type = task[0]
                
                if task_type == 'update_progress':
                    if len(task) >= 3:
                        value, message = task[1], task[2]
                        if self.progress_var:
                            self.progress_var.set(value)
                        if self.status_label:
                            self.status_label.config(text=message)
                        
                        # Update time estimation
                        if self.progress_time_label and value > 0:
                            # Extract phase from message
                            phase = "Analyse"
                            if "Phase" in message:
                                try:
                                    phase = message.split("Phase")[1].split(":")[0].strip()
                                except:
                                    phase = "Analyse"
                            
                            # Calculate ETA based on progress
                            if hasattr(self, 'phase_start_time') and self.phase_start_time:
                                elapsed = time.time() - self.phase_start_time
                                if value > 5:  # Only estimate after 5% progress
                                    total_estimated = elapsed / (value / 100.0)
                                    remaining = max(0, total_estimated - elapsed)
                                    eta_str = f"ETA: {int(remaining)}s" if remaining > 0 else "Fast fertig..."
                                    self.progress_time_label.config(text=f"{eta_str} | Phase: {phase}")
                                else:
                                    self.progress_time_label.config(text=f"Phase: {phase}")
                            else:
                                self.progress_time_label.config(text=f"Phase: {phase}")
                        
                        # Update phase start time when new phase starts
                        if "Phase" in message or "---" in message:
                            self.phase_start_time = time.time()
                            try:
                                phase_name = message.split("Phase")[1].strip() if "Phase" in message else message
                                self.current_phase = phase_name
                            except:
                                self.current_phase = message
                
                elif task_type == 'update_status':
                    if len(task) >= 2:
                        message = task[1]
                        if self.status_label:
                            self.status_label.config(text=message)
                
                elif task_type == 'report_truth_mode':
                    if len(task) >= 2:
                        active = task[1]
                        if self.truth_mode_button:
                            if active:
                                self.truth_mode_button.config(
                                    text="✓ Truth Mode: ON",
                                    bg="#44ff44",
                                    fg="black"
                                )
                            else:
                                self.truth_mode_button.config(
                                    text="❌ Truth Mode: OFF",
                                    bg="#ff4444",
                                    fg="white"
                                )
                
                elif task_type == 'update_progress_with_time':
                    if len(task) >= 4:
                        value, message, current_phase, elapsed_time = task[1], task[2], task[3], task[4]
                        if self.progress_var:
                            self.progress_var.set(value)
                        if self.status_label:
                            self.status_label.config(text=message)
                        if self.progress_time_label:
                            # Calculate ETA
                            if value > 0:
                                total_time = elapsed_time / (value / 100.0)
                                remaining_time = total_time - elapsed_time
                                eta_str = f"ETA: {int(remaining_time)}s" if remaining_time > 0 else "Fast fertig..."
                                phase_str = f" | Phase: {current_phase}" if current_phase else ""
                                self.progress_time_label.config(text=f"{eta_str}{phase_str}")
                            else:
                                self.progress_time_label.config(text="")
                
                elif task_type == 'report_correction':
                    if len(task) >= 2:
                        correction = task[1]
                        self._log_message(f"Correction: {correction}", 'INFO')
                
                elif task_type == 'log':
                    if len(task) >= 3:
                        level, message = task[1], task[2]
                        # Detect phase messages
                        if "Phase" in message or "---" in message:
                            level = 'PHASE'
                        self._log_message(message, level)
                
                # Process any other task types here
        except Exception as e:
            logger.error(f"Error processing GUI task: {e}", exc_info=True)
    
    def _add_files(self):
        """Add files to analysis list."""
        files = filedialog.askopenfilenames(
            title="P&ID Bilder auswählen",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        for file in files:
            self.file_listbox.insert(tk.END, file)
    
    def _remove_files(self):
        """Remove selected files from list."""
        selected = self.file_listbox.curselection()
        for idx in reversed(selected):
            self.file_listbox.delete(idx)
    
    def _clear_files(self):
        """Clear all files from list."""
        self.file_listbox.delete(0, tk.END)
    
    def _start_analysis(self):
        """Start analysis in background thread."""
        files = list(self.file_listbox.get(0, tk.END))
        if not files:
            messagebox.showwarning("Warning", "Bitte wählen Sie mindestens eine Datei aus.")
            return
        
        if self.is_processing:
            messagebox.showinfo("Info", "Analyse läuft bereits.")
            return
        
        # Disable start button
        self.start_button.config(state=tk.DISABLED)
        self.is_processing = True
        self.progress_var.set(0)
        self.phase_start_time = time.time()  # Reset phase start time
        if self.progress_time_label:
            self.progress_time_label.config(text="Starte Analyse...")
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.config(state='disabled')
        
        # Start analysis in background thread
        thread = threading.Thread(
            target=self._run_analysis_worker,
            args=(files,),
            daemon=True
        )
        thread.start()
    
    def _run_analysis_worker(self, files: List[str]):
        """Run analysis in worker thread."""
        try:
            # Check if pipeline coordinator is initialized
            if not self.pipeline_coordinator:
                self._log_message("ERROR: Backend not initialized. GCP_PROJECT_ID not found.", 'ERROR')
                self._queue_update('update_status', "Fehler: Backend nicht initialisiert")
                return
            
            progress_callback = OptimizedProgressCallback(self._queue_update)
            
            # Build model strategy from GUI strategy dropdown
            model_strategy = self._build_model_strategy_from_dropdown()
            
            # Create new pipeline coordinator with selected models
            coordinator = PipelineCoordinator(
                llm_client=self.llm_client,
                knowledge_manager=self.knowledge_manager,
                config_service=self.config_service,
                model_strategy=model_strategy,
                progress_callback=progress_callback
            )
            
            for file_path in files:
                if not self.is_processing:
                    break
                
                self._log_message(f"Processing: {Path(file_path).name}", 'INFO')
                
                try:
                    result = coordinator.process(
                        image_path=file_path,
                        output_dir=None,
                        params_override={
                            'use_monolith_fusion': self.use_monolith_var.get(),
                            'use_predictive_completion': self.use_predictive_var.get(),
                            'use_polyline_refinement': self.use_polyline_var.get(),
                            'use_self_correction': self.use_self_correction_var.get(),
                            # NEW: Advanced features
                            'use_skeleton_line_extraction': self.use_skeleton_extraction_var.get(),
                            'use_topology_critic': self.use_topology_critic_var.get(),
                            'use_legend_consistency_critic': self.use_legend_consistency_critic_var.get(),
                            'two_pass_enabled': self.use_two_pass_var.get(),
                            # NEW: Self-Correction Parameters
                            'max_self_correction_iterations': self.max_iterations_var.get() if self.max_iterations_var else 5,
                            'early_stop_threshold': self.early_stop_threshold_var.get() if self.early_stop_threshold_var else 80.0
                        }
                    )
                    
                    # Store results for results window
                    self.current_results = result
                    
                    # Update results
                    self._update_results(result, file_path)
                    
                    quality_score = result.quality_score if hasattr(result, 'quality_score') else result.get('quality_score', 0.0)
                    self._log_message(f"✓ Completed: {Path(file_path).name} (Score: {quality_score:.2f})", 'SUCCESS')
                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}", exc_info=True)
                    self._log_message(f"✗ Error processing {Path(file_path).name}: {e}", 'ERROR')
        
        finally:
            self.is_processing = False
            self._queue_update('update_status', "Analyse abgeschlossen")
            if self.start_button:
                self.after(0, lambda: self.start_button.config(state=tk.NORMAL))
    
    def _build_model_strategy_from_dropdown(self) -> Dict[str, Any]:
        """Build model strategy dictionary from GUI strategy dropdown."""
        try:
            # Use get_raw_config() to get dict directly
            config = self.config_service.get_raw_config()
            strategies = config.get('strategies', {})
            
            # Handle Pydantic models - convert to dict if needed
            if hasattr(strategies, 'model_dump'):
                strategies = strategies.model_dump()
            elif hasattr(strategies, 'dict'):
                strategies = strategies.dict()
            
            if not isinstance(strategies, dict):
                logger.error(f"Strategies is not a dict: {type(strategies)}")
                self._log_message(f"ERROR: Invalid strategies format in config", 'ERROR')
                return {}
            
            # Get selected strategy from dropdown
            selected_strategy_name = self.model_strategy_var.get() if self.model_strategy_var else "simple_pid_strategy"
            
            self._log_message(f"Looking for strategy: {selected_strategy_name}", 'INFO')
            self._log_message(f"Available strategies: {list(strategies.keys())}", 'INFO')
            
            # Load strategy from config
            if selected_strategy_name in strategies:
                strategy = strategies[selected_strategy_name]
                
                # Handle Pydantic models - convert to dict if needed
                if hasattr(strategy, 'model_dump'):
                    strategy = strategy.model_dump()
                elif hasattr(strategy, 'dict'):
                    strategy = strategy.dict()
                
                if isinstance(strategy, dict):
                    self._log_message(f"Strategy '{selected_strategy_name}' loaded successfully", 'SUCCESS')
                    return strategy
                else:
                    logger.warning(f"Strategy '{selected_strategy_name}' is not a dict: {type(strategy)}")
                    return {}
            
            # Fallback to default_flash
            logger.warning(f"Strategy '{selected_strategy_name}' not found, using 'default_flash'")
            self._log_message(f"WARNING: Strategy '{selected_strategy_name}' not found, using 'default_flash'", 'WARNING')
            fallback = strategies.get('default_flash', {})
            
            # Handle Pydantic models
            if hasattr(fallback, 'model_dump'):
                fallback = fallback.model_dump()
            elif hasattr(fallback, 'dict'):
                fallback = fallback.dict()
            
            return fallback if isinstance(fallback, dict) else {}
            
        except Exception as e:
            logger.error(f"Error building model strategy: {e}", exc_info=True)
            self._log_message(f"ERROR: Failed to build model strategy: {e}", 'ERROR')
            return {}
    
    def _build_model_strategy_from_selections(self) -> Dict[str, Any]:
        """Build model strategy dictionary from GUI model selections (legacy method)."""
        try:
            config = self.config_service.get_config()
            models_config = config.models if hasattr(config, 'models') else config.get('models', {})
            
            model_strategy = {}
            
            # Build strategy from selections
            for phase_key, var in self.model_selections.items():
                selected_model_name = var.get()
                if selected_model_name and selected_model_name in models_config:
                    model_info = models_config[selected_model_name]
                    if isinstance(model_info, dict):
                        model_strategy[phase_key] = model_info
                    elif hasattr(model_info, 'model_dump'):
                        model_strategy[phase_key] = model_info.model_dump()
                    else:
                        model_strategy[phase_key] = model_info
            
            logger.info(f"Built model strategy from GUI selections: {list(model_strategy.keys())}")
            return model_strategy
            
        except Exception as e:
            logger.error(f"Error building model strategy from selections: {e}", exc_info=True)
            # Return empty dict to use default strategy
            return {}
    
    def _update_results(self, result, file_path: str):
        """Update results display."""
        def _update():
            if not self.results_text:
                return
            
            self.results_text.config(state='normal')
            self.results_text.insert(tk.END, f"\n{'='*60}\n")
            self.results_text.insert(tk.END, f"Datei: {Path(file_path).name}\n")
            
            quality_score = result.quality_score if hasattr(result, 'quality_score') else result.get('quality_score', 0.0)
            self.results_text.insert(tk.END, f"Quality Score: {quality_score:.2f}\n")
            
            elements = result.elements if hasattr(result, 'elements') else result.get('elements', [])
            connections = result.connections if hasattr(result, 'connections') else result.get('connections', [])
            
            if elements:
                self.results_text.insert(tk.END, f"Elements: {len(elements)}\n")
            if connections:
                self.results_text.insert(tk.END, f"Connections: {len(connections)}\n")
            
            kpis = result.kpis if hasattr(result, 'kpis') else result.get('kpis', {})
            if kpis:
                self.results_text.insert(tk.END, f"\nKPIs:\n")
                for key, value in kpis.items():
                    self.results_text.insert(tk.END, f"  {key}: {value}\n")
            
            self.results_text.see(tk.END)
            self.results_text.config(state='disabled')
        
        self.after(0, _update)
    
    def _start_pretraining(self):
        """Start pretraining in background thread."""
        if self.is_processing:
            messagebox.showinfo("Info", "Ein Prozess läuft bereits.")
            return
        
        if not self.pipeline_coordinator:
            messagebox.showerror("Error", "Backend nicht initialisiert. Bitte GCP_PROJECT_ID setzen.")
            return
        
        from pathlib import Path
        
        # Get pretraining path from config
        config = self.config_service.get_raw_config()
        pretraining_path = Path(config.get('paths', {}).get('pretraining_symbols', 'training_data/pretraining_symbols'))
        
        if not pretraining_path.exists():
            messagebox.showerror("Error", f"Pretraining-Verzeichnis nicht gefunden: {pretraining_path}\n\nBitte erstellen Sie das Verzeichnis und fügen Sie Symbol-Bilder hinzu.")
            return
        
        # Get model info
        models_config = config.get('models', {})
        model_info = models_config.get('Google Gemini 2.5 Flash', {})
        if not model_info:
            model_info = list(models_config.values())[0] if models_config else {}
        
        # Start pretraining in background thread
        self.is_processing = True
        self._log_message("=== Pretraining gestartet ===", 'INFO')
        self._log_message(f"Verzeichnis: {pretraining_path}", 'INFO')
        
        thread = threading.Thread(
            target=self._run_pretraining_worker,
            args=(pretraining_path, model_info),
            daemon=True
        )
        thread.start()
    
    def _run_pretraining_worker(self, pretraining_path: Path, model_info: Dict[str, Any]):
        """Run pretraining in worker thread."""
        try:
            from src.analyzer.learning.active_learner import ActiveLearner
            
            # Get active learner from pipeline coordinator
            active_learner = self.pipeline_coordinator.active_learner
            
            self._log_message(f"Verarbeite Symbole aus: {pretraining_path}", 'INFO')
            
            # Run pretraining
            report = active_learner.learn_from_pretraining_symbols(
                pretraining_path=pretraining_path,
                model_info=model_info
            )
            
            # Log results
            self._log_message("=== Pretraining abgeschlossen ===", 'SUCCESS')
            self._log_message(f"Dateien verarbeitet: {report.get('symbols_processed', 0)}", 'INFO')
            self._log_message(f"Sammlungen verarbeitet: {report.get('collections_processed', 0)}", 'INFO')
            self._log_message(f"Einzelne Symbole: {report.get('individual_symbols_processed', 0)}", 'INFO')
            self._log_message(f"Neue Symbole gelernt: {report.get('symbols_learned', 0)}", 'SUCCESS')
            self._log_message(f"Symbole aktualisiert: {report.get('symbols_updated', 0)}", 'INFO')
            self._log_message(f"Duplikate gefunden: {report.get('duplicates_found', 0)}", 'INFO')
            
            # Show symbol library stats
            symbol_count = active_learner.symbol_library.get_symbol_count()
            self._log_message(f"Gesamt Symbole in Library: {symbol_count}", 'SUCCESS')
            
            if report.get('errors'):
                self._log_message(f"Warnung: {len(report.get('errors', []))} Fehler aufgetreten", 'WARNING')
                for error in report.get('errors', [])[:5]:
                    self._log_message(f"  - {error}", 'WARNING')
            
            messagebox.showinfo(
                "Pretraining abgeschlossen",
                f"Pretraining erfolgreich!\n\n"
                f"Neue Symbole: {report.get('symbols_learned', 0)}\n"
                f"Duplikate: {report.get('duplicates_found', 0)}\n"
                f"Gesamt in Library: {symbol_count}"
            )
            
        except Exception as e:
            self._log_message(f"Fehler beim Pretraining: {e}", 'ERROR')
            messagebox.showerror("Error", f"Fehler beim Pretraining:\n{e}")
        finally:
            self.is_processing = False
    
    def _start_training_camp(self):
        """Start training camp in background thread."""
        if self.is_processing:
            messagebox.showinfo("Info", "Ein Prozess läuft bereits.")
            return
        
        from src.analyzer.training.training_camp import TrainingCamp
        from pathlib import Path
        
        training_camp = TrainingCamp(
            pipeline_coordinator=self.pipeline_coordinator,
            config_service=self.config_service,
            training_data_dir=Path("training_data"),
            config_path=Path("config.yaml")
        )
        
        duration = self.duration_var.get()
        max_cycles = self.max_cycles_var.get()
        
        # Start training in background thread
        thread = threading.Thread(
            target=self._run_training_worker,
            args=(training_camp, duration, max_cycles),
            daemon=True
        )
        thread.start()
    
    def _run_training_worker(self, training_camp, duration, max_cycles):
        """Run training camp in worker thread."""
        try:
            self._log_message("Training Camp gestartet...", 'INFO')
            
            report = training_camp.run_full_training_camp(
                duration_hours=duration,
                max_cycles=max_cycles,
                sequential=True
            )
            
            self._log_message(f"Training Camp abgeschlossen!", 'SUCCESS')
            self._log_message(f"Best Score: {report.get('best_score', 0.0):.2f}", 'INFO')
            self._log_message(f"Best Strategy: {report.get('best_strategy', 'N/A')}", 'INFO')
            
        except Exception as e:
            logger.error(f"Error in training camp: {e}", exc_info=True)
            self._log_message(f"Training Error: {e}", 'ERROR')
    
    def _cancel_analysis(self):
        """Cancel current analysis."""
        self.is_processing = False
        self._log_message("Analyse abgebrochen", 'WARNING')
    
    def _test_strategy(self):
        """Test the selected strategy with a test image."""
        if self.is_processing:
            messagebox.showinfo("Info", "Ein Prozess läuft bereits.")
            return
        
        if not self.pipeline_coordinator:
            messagebox.showerror("Error", "Backend nicht initialisiert. Bitte GCP_PROJECT_ID setzen.")
            return
        
        # Find test image
        test_image_path = None
        
        # Try to find test image in training_data/simple_pids
        simple_pids_dir = Path("training_data") / "simple_pids"
        if simple_pids_dir.exists():
            images = list(simple_pids_dir.glob("*.png")) + list(simple_pids_dir.glob("*.jpg")) + list(simple_pids_dir.glob("*.jpeg"))
            images = [img for img in images if not any(exclude in img.name.lower() for exclude in ['truth', 'output', 'result', 'cgm', 'temp', 'correction', 'symbol'])]
            if images:
                test_image_path = images[0]
        
        # If no test image found, ask user to select one
        if not test_image_path:
            test_image_path = filedialog.askopenfilename(
                title="Test-Bild auswählen",
                filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
            )
            if not test_image_path:
                return
            test_image_path = Path(test_image_path)
        
        # Get selected strategy
        selected_strategy = self.model_strategy_var.get() if self.model_strategy_var else "simple_pid_strategy"
        
        self._log_message(f"=== Strategie-Test gestartet ===", 'INFO')
        self._log_message(f"Strategie: {selected_strategy}", 'INFO')
        self._log_message(f"Test-Bild: {test_image_path.name}", 'INFO')
        
        # Disable buttons during test
        self.is_processing = True
        self.progress_var.set(0)
        self.phase_start_time = time.time()
        
        # Start test in background thread
        thread = threading.Thread(
            target=self._run_strategy_test_worker,
            args=(test_image_path, selected_strategy),
            daemon=True
        )
        thread.start()
    
    def _run_strategy_test_worker(self, test_image_path: Path, strategy_name: str):
        """Run strategy test in worker thread."""
        try:
            self._log_message(f"Starting strategy test worker...", 'INFO')
            
            # Check if test image exists
            if not test_image_path.exists():
                self._log_message(f"ERROR: Test image not found: {test_image_path}", 'ERROR')
                raise FileNotFoundError(f"Test image not found: {test_image_path}")
            
            self._log_message(f"Test image found: {test_image_path.name}", 'INFO')
            
            # Build model strategy from dropdown
            model_strategy = self._build_model_strategy_from_dropdown()
            self._log_message(f"Model strategy built: {list(model_strategy.keys()) if model_strategy else 'empty'}", 'INFO')
            
            if not model_strategy:
                self._log_message("ERROR: No model strategy available - check config.yaml", 'ERROR')
                raise ValueError("No model strategy available")
            
            # Create new pipeline coordinator with selected strategy
            self._log_message("Creating PipelineCoordinator...", 'INFO')
            coordinator = PipelineCoordinator(
                llm_client=self.llm_client,
                knowledge_manager=self.knowledge_manager,
                config_service=self.config_service,
                model_strategy=model_strategy,
                progress_callback=OptimizedProgressCallback(self._queue_update)
            )
            self._log_message("PipelineCoordinator created successfully", 'SUCCESS')
            
            # Find truth data if available
            truth_path = None
            truth_candidates = [
                test_image_path.parent / f"{test_image_path.stem}_truth.json",
                test_image_path.parent / f"{test_image_path.stem}_truth_cgm.json"
            ]
            for candidate in truth_candidates:
                if candidate.exists():
                    truth_path = candidate
                    break
            
            if truth_path:
                self._log_message(f"Truth-Daten gefunden: {truth_path.name}", 'INFO')
            
            # Run analysis
            self._log_message(f"Starting analysis of: {test_image_path.name}", 'INFO')
            self._log_message(f"Parameters: Monolith={self.use_monolith_var.get()}, Predictive={self.use_predictive_var.get()}, Polyline={self.use_polyline_var.get()}, SelfCorrection={self.use_self_correction_var.get()}", 'INFO')
            
            try:
                result = coordinator.process(
                    image_path=str(test_image_path),
                    output_dir=None,
                    params_override={
                        'truth_data_path': str(truth_path) if truth_path else None,
                        'use_monolith_fusion': self.use_monolith_var.get(),
                        'use_predictive_completion': self.use_predictive_var.get(),
                        'use_polyline_refinement': self.use_polyline_var.get(),
                        'use_self_correction': self.use_self_correction_var.get(),
                        'use_skeleton_line_extraction': self.use_skeleton_extraction_var.get(),
                        'use_topology_critic': self.use_topology_critic_var.get(),
                        'use_legend_consistency_critic': self.use_legend_consistency_critic_var.get(),
                        'two_pass_enabled': self.use_two_pass_var.get(),
                        'max_self_correction_iterations': self.max_iterations_var.get() if self.max_iterations_var else 5,
                        'early_stop_threshold': self.early_stop_threshold_var.get() if self.early_stop_threshold_var else 80.0
                    }
                )
                self._log_message("Analysis completed successfully", 'SUCCESS')
            except Exception as e:
                self._log_message(f"ERROR during analysis: {e}", 'ERROR')
                logger.error(f"Error during analysis: {e}", exc_info=True)
                raise
            
            # Store results
            self.current_results = result
            
            # Update results display
            self._update_results(result, str(test_image_path))
            
            # Log results
            quality_score = result.quality_score if hasattr(result, 'quality_score') else result.get('quality_score', 0.0)
            elements = result.elements if hasattr(result, 'elements') else result.get('elements', [])
            connections = result.connections if hasattr(result, 'connections') else result.get('connections', [])
            
            self._log_message(f"=== Strategie-Test abgeschlossen ===", 'SUCCESS')
            self._log_message(f"Strategie: {strategy_name}", 'INFO')
            self._log_message(f"Quality Score: {quality_score:.2f}%", 'INFO')
            self._log_message(f"Elemente: {len(elements)}", 'INFO')
            self._log_message(f"Verbindungen: {len(connections)}", 'INFO')
            
            # Show success message
            messagebox.showinfo(
                "Strategie-Test abgeschlossen",
                f"Strategie: {strategy_name}\n\n"
                f"Quality Score: {quality_score:.2f}%\n"
                f"Elemente: {len(elements)}\n"
                f"Verbindungen: {len(connections)}"
            )
            
        except Exception as e:
            logger.error(f"Error testing strategy: {e}", exc_info=True)
            self._log_message(f"✗ Fehler beim Strategie-Test: {e}", 'ERROR')
            messagebox.showerror("Error", f"Fehler beim Strategie-Test:\n{e}")
        finally:
            self.is_processing = False
            self._queue_update('update_status', "Strategie-Test abgeschlossen")
            if self.start_button:
                self.after(0, lambda: self.start_button.config(state=tk.NORMAL))
    
    def _show_results(self):
        """Show results in new window with detailed view."""
        if not self.current_results:
            messagebox.showinfo("Info", "Keine Ergebnisse verfügbar.")
            return
        
        # Create results window
        results_window = tk.Toplevel(self)
        results_window.title("P&ID Analyse Ergebnisse")
        results_window.geometry("1200x800")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        result = self.current_results
        
        # Tab 1: Übersicht
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Übersicht")
        
        overview_text = scrolledtext.ScrolledText(overview_frame, wrap=tk.WORD)
        overview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Collect overview data
        quality_score = result.quality_score if hasattr(result, 'quality_score') else result.get('quality_score', 0.0)
        elements = result.elements if hasattr(result, 'elements') else result.get('elements', [])
        connections = result.connections if hasattr(result, 'connections') else result.get('connections', [])
        kpis = result.kpis if hasattr(result, 'kpis') else result.get('kpis', {})
        
        overview_text.insert(tk.END, f"P&ID Analyse Ergebnisse\n")
        overview_text.insert(tk.END, f"{'='*60}\n\n")
        
        overview_text.insert(tk.END, f"Quality Score: {quality_score:.2f}/100\n\n")
        overview_text.insert(tk.END, f"Elemente: {len(elements)}\n")
        overview_text.insert(tk.END, f"Verbindungen: {len(connections)}\n\n")
        
        if kpis:
            overview_text.insert(tk.END, f"KPIs:\n")
            overview_text.insert(tk.END, f"{'-'*60}\n")
            for key, value in kpis.items():
                if isinstance(value, (int, float)):
                    overview_text.insert(tk.END, f"  {key}: {value:.2f}\n")
                else:
                    overview_text.insert(tk.END, f"  {key}: {value}\n")
        
        overview_text.config(state='disabled')
        
        # Tab 2: Elemente
        elements_frame = ttk.Frame(notebook)
        notebook.add(elements_frame, text=f"Elemente ({len(elements)})")
        
        elements_tree = ttk.Treeview(elements_frame, columns=('ID', 'Typ', 'Label', 'Confidence', 'Position'), show='headings', height=20)
        elements_tree.heading('ID', text='ID')
        elements_tree.heading('Typ', text='Typ')
        elements_tree.heading('Label', text='Label')
        elements_tree.heading('Confidence', text='Confidence')
        elements_tree.heading('Position', text='Position (x, y, w, h)')
        
        elements_tree.column('ID', width=150)
        elements_tree.column('Typ', width=150)
        elements_tree.column('Label', width=200)
        elements_tree.column('Confidence', width=100)
        elements_tree.column('Position', width=250)
        
        scrollbar_elements = ttk.Scrollbar(elements_frame, orient=tk.VERTICAL, command=elements_tree.yview)
        elements_tree.configure(yscrollcommand=scrollbar_elements.set)
        
        elements_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_elements.pack(side=tk.RIGHT, fill=tk.Y)
        
        for el in elements[:500]:  # Limit to 500 for performance
            el_dict = el if isinstance(el, dict) else el.model_dump() if hasattr(el, 'model_dump') else el.__dict__ if hasattr(el, '__dict__') else {}
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
                position = f"({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})"
            else:
                position = "N/A"
            
            elements_tree.insert('', tk.END, values=(el_id, el_type, el_label, f"{el_confidence:.2f}", position))
        
        if len(elements) > 500:
            elements_tree.insert('', tk.END, values=('...', f'... und {len(elements) - 500} weitere', '', '', ''))
        
        # Tab 3: Verbindungen
        connections_frame = ttk.Frame(notebook)
        notebook.add(connections_frame, text=f"Verbindungen ({len(connections)})")
        
        connections_tree = ttk.Treeview(connections_frame, columns=('Von', 'Zu', 'Confidence', 'Typ'), show='headings', height=20)
        connections_tree.heading('Von', text='Von')
        connections_tree.heading('Zu', text='Zu')
        connections_tree.heading('Confidence', text='Confidence')
        connections_tree.heading('Typ', text='Typ')
        
        connections_tree.column('Von', width=200)
        connections_tree.column('Zu', width=200)
        connections_tree.column('Confidence', width=100)
        connections_tree.column('Typ', width=150)
        
        scrollbar_conn = ttk.Scrollbar(connections_frame, orient=tk.VERTICAL, command=connections_tree.yview)
        connections_tree.configure(yscrollcommand=scrollbar_conn.set)
        
        connections_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_conn.pack(side=tk.RIGHT, fill=tk.Y)
        
        for conn in connections[:500]:  # Limit to 500 for performance
            conn_dict = conn if isinstance(conn, dict) else conn.model_dump() if hasattr(conn, 'model_dump') else conn.__dict__ if hasattr(conn, '__dict__') else {}
            from_id = conn_dict.get('from_id', 'N/A')
            to_id = conn_dict.get('to_id', 'N/A')
            conn_confidence = conn_dict.get('confidence', 0.0)
            conn_kind = conn_dict.get('kind', 'process')
            
            connections_tree.insert('', tk.END, values=(from_id, to_id, f"{conn_confidence:.2f}", conn_kind))
        
        if len(connections) > 500:
            connections_tree.insert('', tk.END, values=('...', f'... und {len(connections) - 500} weitere', '', ''))
        
        # Tab 4: Metadaten
        metadata_frame = ttk.Frame(notebook)
        notebook.add(metadata_frame, text="Metadaten")
        
        metadata_text = scrolledtext.ScrolledText(metadata_frame, wrap=tk.WORD)
        metadata_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        metadata = result.metadata if hasattr(result, 'metadata') else result.get('metadata', {})
        legend_data = result.legend_data if hasattr(result, 'legend_data') else result.get('legend_data', {})
        
        if metadata:
            metadata_text.insert(tk.END, "Metadaten:\n")
            metadata_text.insert(tk.END, f"{'-'*60}\n")
            for key, value in metadata.items():
                metadata_text.insert(tk.END, f"  {key}: {value}\n")
            metadata_text.insert(tk.END, "\n")
        
        if legend_data:
            metadata_text.insert(tk.END, "Legende:\n")
            metadata_text.insert(tk.END, f"{'-'*60}\n")
            
            symbol_map = legend_data.get('symbol_map', {})
            if symbol_map:
                metadata_text.insert(tk.END, "Symbol-Map:\n")
                for key, value in symbol_map.items():
                    metadata_text.insert(tk.END, f"  {key}: {value}\n")
                metadata_text.insert(tk.END, "\n")
            
            line_map = legend_data.get('line_map', {})
            if line_map:
                metadata_text.insert(tk.END, "Rohrleitungs-Map:\n")
                for key, value in line_map.items():
                    metadata_text.insert(tk.END, f"  {key}: {value}\n")
        
        metadata_text.config(state='disabled')
    
    def _queue_update(self, *args):
        """Queue GUI update (thread-safe)."""
        self.gui_queue.put(args)
    
    def _log_message(self, message: str, level: str = 'INFO'):
        """Log message to GUI log."""
        def _update():
            if not self.log_text:
                return
            
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, f"[{level}] {message}\n", level)
            self.log_text.see(tk.END)
            # Limit log size (keep last 1000 lines) for performance
            lines = self.log_text.get('1.0', tk.END).split('\n')
            if len(lines) > 1000:
                self.log_text.delete('1.0', f'{len(lines) - 1000}.0')
            self.log_text.config(state='disabled')
        
        self.after(0, _update)
    
    def _create_tooltip(self, widget, text: str):
        """Create a tooltip for a widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(
                tooltip,
                text=text,
                background="#ffffe0",
                foreground="black",
                relief=tk.SOLID,
                borderwidth=1,
                font=("Arial", 9),
                justify=tk.LEFT,
                padx=5,
                pady=5
            )
            label.pack()
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
    
    def _on_closing(self):
        """Handle window closing - cleanup resources."""
        try:
            # Remove GUI log handler from root logger
            root_logger = logging.getLogger()
            if self.gui_log_handler:
                root_logger.removeHandler(self.gui_log_handler)
                self.gui_log_handler = None
            
            # Remove file log handler
            if self.file_log_handler:
                root_logger.removeHandler(self.file_log_handler)
                self.file_log_handler.close()
                self.file_log_handler = None
            
            logger.info("GUI closing - cleanup complete")
        except Exception as e:
            logger.error(f"Error during GUI cleanup: {e}", exc_info=True)
        finally:
            # Destroy the window
            self.destroy()


def main():
    """Main entry point for optimized GUI."""
    import os
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s - %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        app = OptimizedGUI()
        app.mainloop()
    except Exception as e:
        logger.error(f"Fatal error in GUI: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"Failed to start GUI: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()

