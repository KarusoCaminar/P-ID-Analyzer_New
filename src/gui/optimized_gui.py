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
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

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
        self.truth_mode_label = None
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
        """Setup GUI logging handler to capture all log messages."""
        try:
            # Create GUI log handler
            self.gui_log_handler = GUILogHandler(self._queue_update)
            self.gui_log_handler.setLevel(logging.DEBUG)
            
            # Add handler to root logger to capture all logs
            root_logger = logging.getLogger()
            root_logger.addHandler(self.gui_log_handler)
            
            logger.info("GUI logging handler configured")
        except Exception as e:
            logger.error(f"Error setting up GUI logging: {e}", exc_info=True)
    
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
            except Exception as e:
                logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
                self._log_message(f"WARNING: LLMClient initialization failed: {e}", 'WARNING')
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
                    llm_handler=self.llm_client,
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
        
        self.status_label = ttk.Label(status_frame, text="Bereit", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100,
            length=300,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.truth_mode_label = ttk.Label(status_frame, text="", foreground="gray")
        self.truth_mode_label.pack(side=tk.LEFT, padx=5)
        
        # Middle: Main content with Notebook
        notebook_frame = ttk.Frame(main_frame)
        notebook_frame.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tabs
        self._create_analysis_tab()
        self._create_training_tab()
        self._create_visualization_tab()
        
        # Bottom: Log
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            wrap=tk.WORD,
            state='disabled',
            font=("Courier New", 9),
            bg='#2e2e2e',
            fg='#ffffff'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure log tags
        self.log_text.tag_config('INFO', foreground='#a0a0a0')
        self.log_text.tag_config('WARNING', foreground='#FFA500')
        self.log_text.tag_config('ERROR', foreground='#FF0000')
        self.log_text.tag_config('SUCCESS', foreground='#00FF00')
    
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
        
        ttk.Button(file_buttons, text="Dateien hinzufügen", command=self._add_files).pack(pady=2, fill=tk.X)
        ttk.Button(file_buttons, text="Dateien entfernen", command=self._remove_files).pack(pady=2, fill=tk.X)
        ttk.Button(file_buttons, text="Alle löschen", command=self._clear_files).pack(pady=2, fill=tk.X)
        
        # Options
        options_frame = ttk.LabelFrame(analysis_frame, text="Optionen", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.use_monolith_var = tk.BooleanVar(value=True)
        self.use_predictive_var = tk.BooleanVar(value=True)
        self.use_polyline_var = tk.BooleanVar(value=True)
        self.use_self_correction_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="Monolith Fusion", variable=self.use_monolith_var).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(options_frame, text="Predictive Completion", variable=self.use_predictive_var).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Checkbutton(options_frame, text="Polyline Refinement", variable=self.use_polyline_var).grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(options_frame, text="Self-Correction", variable=self.use_self_correction_var).grid(row=1, column=1, sticky=tk.W, padx=5)
        
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
        
        ttk.Button(action_frame, text="Abbrechen", command=self._cancel_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Ergebnisse anzeigen", command=self._show_results).pack(side=tk.LEFT, padx=5)
        
        # Results preview
        results_frame = ttk.LabelFrame(analysis_frame, text="Ergebnisse", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, state='disabled')
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
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
                
                elif task_type == 'update_status':
                    if len(task) >= 2:
                        message = task[1]
                        if self.status_label:
                            self.status_label.config(text=message)
                
                elif task_type == 'report_truth_mode':
                    if len(task) >= 2:
                        active = task[1]
                        if self.truth_mode_label:
                            if active:
                                self.truth_mode_label.config(text="✓ Truth Mode", foreground="green")
                            else:
                                self.truth_mode_label.config(text="", foreground="gray")
                
                elif task_type == 'report_correction':
                    if len(task) >= 2:
                        correction = task[1]
                        self._log_message(f"Correction: {correction}", 'INFO')
                
                elif task_type == 'log':
                    if len(task) >= 3:
                        level, message = task[1], task[2]
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
            
            # Update pipeline coordinator with progress callback
            self.pipeline_coordinator.progress_callback = progress_callback
            
            for file_path in files:
                if not self.is_processing:
                    break
                
                self._log_message(f"Processing: {Path(file_path).name}", 'INFO')
                
                try:
                    result = self.pipeline_coordinator.process(
                        image_path=file_path,
                        output_dir=None,
                        params_override={
                            'use_monolith_fusion': self.use_monolith_var.get(),
                            'use_predictive_completion': self.use_predictive_var.get(),
                            'use_polyline_refinement': self.use_polyline_var.get(),
                            'use_self_correction': self.use_self_correction_var.get()
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
    
    def _start_training_camp(self):
        """Start training camp in background thread."""
        if self.is_processing:
            messagebox.showinfo("Info", "Training läuft bereits.")
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
    
    def _on_closing(self):
        """Handle window closing - cleanup resources."""
        try:
            # Remove GUI log handler from root logger
            if self.gui_log_handler:
                root_logger = logging.getLogger()
                root_logger.removeHandler(self.gui_log_handler)
                self.gui_log_handler = None
            
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

