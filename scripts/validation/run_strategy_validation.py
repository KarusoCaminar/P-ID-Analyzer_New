"""
Test-Harness für "Pipeline Isolation & Integration"

Misst F1-Scores für verschiedene Pipeline-Strategien (Feature-Flag-Tests).

Dieses Skript führt die Pipeline mit verschiedenen Konfigurationen aus und
validiert die Ergebnisse gegen Ground Truth-Daten, um die Performance jeder
Komponente isoliert zu messen.
"""

import sys
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.json_encoder import PydanticJSONEncoder, json_dump_safe

# CRITICAL: Live Log Monitoring
sys.path.insert(0, str(project_root))
from scripts.utils.live_log_monitor import monitor_test_logs, LiveLogMonitor

# Load .env file automatically (CRITICAL: Lädt GCP-Credentials automatisch)
try:
    from src.utils.env_loader import load_env_automatically
    if load_env_automatically():
        print(f"[OK] .env Datei automatisch geladen")
    else:
        print(f"[WARNING] .env Datei nicht gefunden")
except (ImportError, Exception) as e:
    # Fallback: Try direct dotenv import
    try:
        from dotenv import load_dotenv
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            print(f"[OK] .env Datei geladen: {env_file}")
    except ImportError:
        pass

# --- Kern-Imports ---
from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.evaluation.kpi_calculator import KPICalculator

# --- Test-Konfiguration ---
# Test-Bilder pro Test konfigurieren
TEST_IMAGES = {
    "Test 1": "training_data/complex_pids/page_1_original.png",  # MIT Legende
    "Test 2": "training_data/simple_pids/Einfaches P&I.png",  # Simple P&ID
    "Test 3": "training_data/simple_pids/Einfaches P&I.png",  # Simple P&ID
    "Test 4": "training_data/complex_pids/page_1_original.png",  # Komplexes Bild
    "Test 5a": "training_data/complex_pids/page_1_original.png",  # Komplexes Bild
    "Test 5b": "training_data/complex_pids/page_1_original.png",  # Komplexes Bild
    "Test 5c": "training_data/complex_pids/page_1_original.png",  # Komplexes Bild
}

TEST_GROUND_TRUTH = {
    "Test 1": "training_data/complex_pids/page_1_original_truth_cgm.json",
    "Test 2": "training_data/simple_pids/Einfaches P&I_truth.json",
    "Test 3": "training_data/simple_pids/Einfaches P&I_truth.json",
    "Test 4": "training_data/complex_pids/page_1_original_truth_cgm.json",
    "Test 5a": "training_data/complex_pids/page_1_original_truth_cgm.json",
    "Test 5b": "training_data/complex_pids/page_1_original_truth_cgm.json",
    "Test 5c": "training_data/complex_pids/page_1_original_truth_cgm.json",
}

# Standard-Pfade (für --image und --ground-truth wenn nicht spezifiziert)
IMAGE_TO_TEST = "training_data/simple_pids/Einfaches P&I.png"
GROUND_TRUTH = "training_data/simple_pids/Einfaches P&I_truth.json"
OUTPUT_DIR_BASE = "outputs/strategy_validation"

# Projekt-Root zum Pfad hinzufügen (für relative Pfade)
project_root = Path(__file__).parent.parent.parent

# Setup Logging
LoggingService.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ground_truth(gt_path: str) -> Optional[Dict[str, Any]]:
    """Lädt die Ground-Truth-Daten."""
    try:
        gt_file = Path(gt_path)
        if not gt_file.exists():
            # Try relative to project root
            gt_file = project_root / gt_path
            if not gt_file.exists():
                logger.warning(f"Ground Truth nicht gefunden: {gt_path}")
                return None
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden der Ground Truth: {e}")
        return None


def validate_test_configuration() -> Dict[str, Any]:
    """
    Validiert die Test-Konfiguration (Bilder und Ground Truth).
    
    Returns:
        Dictionary mit Validierungsergebnissen
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "test_images": {},
        "test_ground_truth": {}
    }
    
    logger.info("=" * 60)
    logger.info("[VALIDATION] Validiere Test-Konfiguration...")
    logger.info("=" * 60)
    
    # Validate test images
    for test_key, image_path in TEST_IMAGES.items():
        image_file = Path(image_path)
        if not image_file.exists():
            # Try relative to project root
            image_file = project_root / image_path
            if not image_file.exists():
                validation_results["valid"] = False
                error_msg = f"Test {test_key}: Bild nicht gefunden: {image_path}"
                validation_results["errors"].append(error_msg)
                validation_results["test_images"][test_key] = {
                    "path": image_path,
                    "exists": False,
                    "error": error_msg
                }
                logger.error(f"[ERROR] {error_msg}")
                continue
        
        validation_results["test_images"][test_key] = {
            "path": str(image_file.resolve()),
            "exists": True,
            "size_mb": image_file.stat().st_size / (1024 * 1024)
        }
        logger.info(f"[OK] Test {test_key}: Bild gefunden: {image_file.resolve()}")
    
    # Validate ground truth files
    for test_key, gt_path in TEST_GROUND_TRUTH.items():
        gt_file = Path(gt_path)
        if not gt_file.exists():
            # Try relative to project root
            gt_file = project_root / gt_path
            if not gt_file.exists():
                validation_results["warnings"].append(
                    f"Test {test_key}: Ground Truth nicht gefunden: {gt_path} (Test läuft ohne Validierung)"
                )
                validation_results["test_ground_truth"][test_key] = {
                    "path": gt_path,
                    "exists": False,
                    "warning": "Ground Truth nicht gefunden"
                }
                logger.warning(f"[WARNING] Test {test_key}: Ground Truth nicht gefunden: {gt_path}")
                continue
        
        # Try to load and validate JSON structure
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            
            elements_count = len(gt_data.get('elements', []))
            connections_count = len(gt_data.get('connections', []))
            
            validation_results["test_ground_truth"][test_key] = {
                "path": str(gt_file.resolve()),
                "exists": True,
                "elements_count": elements_count,
                "connections_count": connections_count,
                "valid": True
            }
            logger.info(f"[OK] Test {test_key}: Ground Truth gefunden: {gt_file.resolve()} ({elements_count} Elemente, {connections_count} Verbindungen)")
        except Exception as e:
            validation_results["warnings"].append(
                f"Test {test_key}: Ground Truth ungültig: {e}"
            )
            validation_results["test_ground_truth"][test_key] = {
                "path": str(gt_file.resolve()),
                "exists": True,
                "valid": False,
                "error": str(e)
            }
            logger.warning(f"[WARNING] Test {test_key}: Ground Truth ungültig: {e}")
    
    logger.info("=" * 60)
    if validation_results["valid"]:
        logger.info("[OK] Test-Konfiguration ist gültig!")
    else:
        logger.error("[ERROR] Test-Konfiguration hat Fehler!")
    logger.info("=" * 60)
    
    return validation_results


def run_test(
    test_name: str, 
    coordinator: PipelineCoordinator, 
    image_path: str,
    gt_data: Optional[Dict[str, Any]],
    param_overrides: Dict[str, Any]
) -> Dict[str, float]:
    """
    Führt einen einzelnen Testlauf aus, wertet ihn aus und gibt KPIs zurück.
    
    Args:
        test_name: Name des Tests
        coordinator: Pipeline Coordinator
        image_path: Pfad zum Testbild
        gt_data: Ground Truth Daten (optional)
        param_overrides: Parameter-Overrides für diesen Test
        
    Returns:
        Dictionary mit KPIs (element_f1, connection_f1, etc.)
    """
    # Create test output directory with clean structure
    test_output_dir = Path(OUTPUT_DIR_BASE) / test_name.replace(" ", "_").replace(":", "")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging for this test - save logs directly to test directory
    test_log_file = test_output_dir / "pipeline.log"
    LoggingService.setup_logging(
        log_level=logging.INFO,
        log_file=test_log_file
    )
    
    # Setup LLM logging for this test
    test_llm_log_dir = test_output_dir / "logs"
    test_llm_log_dir.mkdir(parents=True, exist_ok=True)
    LoggingService.setup_llm_logging(
        log_dir=test_llm_log_dir,
        log_level=logging.DEBUG
    )
    
    logger.info("=" * 60)
    logger.info(f"[START] Starte Test: {test_name}")
    logger.info(f"Overrides: {json.dumps(param_overrides, indent=2)}")
    logger.info(f"Output-Verzeichnis: {test_output_dir}")
    logger.info(f"Log-Datei: {test_log_file}")
    logger.info("=" * 60)
    
    # CRITICAL: Start Live Log Monitoring (während des Tests)
    log_monitor = None
    try:
        log_monitor = monitor_test_logs(
            log_file=test_log_file,
            test_name=test_name,
            output_callback=None  # Use default (print to stdout)
        )
        logger.info(f"[LiveLog] Monitoring gestartet: {test_log_file}")
    except Exception as e:
        logger.warning(f"[LiveLog] Fehler beim Starten des Live-Monitors: {e}")
        log_monitor = None
    
    # 1. Pipeline ausführen
    try:
        # --- KORREKTUR: Circuit Breaker vor jedem Test zurücksetzen ---
        logger.info("Resetting Circuit Breaker before test...")
        if hasattr(coordinator, 'llm_client') and hasattr(coordinator.llm_client, 'retry_handler'):
            if hasattr(coordinator.llm_client.retry_handler, 'circuit_breaker'):
                coordinator.llm_client.retry_handler.circuit_breaker.reset()
                logger.info("Circuit Breaker reset to CLOSED.")
            else:
                logger.warning("Circuit Breaker not found in retry_handler.")
        else:
            logger.warning("LLM client or retry_handler not found. Cannot reset Circuit Breaker.")
        # --- ENDE KORREKTUR ---
        
        # Add test metadata to params_override
        params_with_metadata = {
            **param_overrides,
            'test_name': test_name,
            'test_description': f"Strategy validation test: {test_name}"
        }
        
        result = coordinator.process(
            image_path=image_path,
            output_dir=str(test_output_dir),
            params_override=params_with_metadata
        )
        
        # Convert result to dict if needed
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = result if isinstance(result, dict) else {
                'elements': getattr(result, 'elements', []),
                'connections': getattr(result, 'connections', [])
            }
        
        # Save results JSON to test directory (with datetime serialization support)
        results_file = test_output_dir / "results.json"
        
        # CRITICAL FIX: Use json_dump_safe() instead of json.dump() to handle datetime and Pydantic models
        with open(results_file, 'w', encoding='utf-8') as f:
            json_dump_safe(result_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Ergebnisse gespeichert: {results_file}")
        
    except Exception as e:
        # CRITICAL: Stop log monitor on error
        if log_monitor:
            log_monitor.flush()
            log_monitor.stop()
            logger.info(f"[LiveLog] Monitoring beendet (Fehler)")
        
        logger.error(f"Test '{test_name}' FEHLGESCHLAGEN während der Ausführung: {e}", exc_info=True)
        return {
            "element_f1": 0.0, 
            "connection_f1": 0.0, 
            "element_precision": 0.0,
            "element_recall": 0.0,
            "connection_precision": 0.0,
            "connection_recall": 0.0,
            "error": 1.0
        }
    
    # 2. Ergebnis validieren (wenn Ground Truth verfügbar)
    if gt_data:
        try:
            kpi_calc = KPICalculator()
            kpis = kpi_calc.calculate_comprehensive_kpis(
                analysis_data=result_dict,
                truth_data=gt_data
            )
            
            # Extract F1 scores - KPIs are returned directly, not nested
            f1_elements = kpis.get('element_f1', 0.0)
            f1_connections = kpis.get('connection_f1', 0.0)
            precision_elements = kpis.get('element_precision', 0.0)
            recall_elements = kpis.get('element_recall', 0.0)
            precision_connections = kpis.get('connection_precision', 0.0)
            recall_connections = kpis.get('connection_recall', 0.0)
            
            # CRITICAL: Flush remaining log lines before stopping monitor
            if log_monitor:
                log_monitor.flush()
                log_monitor.stop()
                logger.info(f"[LiveLog] Monitoring beendet")
            
            logger.info(f"[DONE] Test '{test_name}' Abgeschlossen:")
            logger.info(f"  Element F1:    {f1_elements:.4f}")
            logger.info(f"  Element Precision: {precision_elements:.4f}")
            logger.info(f"  Element Recall:    {recall_elements:.4f}")
            logger.info(f"  Connection F1: {f1_connections:.4f}")
            logger.info(f"  Connection Precision: {precision_connections:.4f}")
            logger.info(f"  Connection Recall:    {recall_connections:.4f}")
            logger.info("=" * 60)
            
            # Save KPIs to test directory
            kpis_file = test_output_dir / "kpis.json"
            kpis_data = {
                "element_f1": f1_elements,
                "connection_f1": f1_connections,
                "element_precision": precision_elements,
                "element_recall": recall_elements,
                "connection_precision": precision_connections,
                "connection_recall": recall_connections,
                "full_kpis": kpis
            }
            with open(kpis_file, 'w', encoding='utf-8') as f:
                json_dump_safe(kpis_data, f, indent=2, ensure_ascii=False)
            logger.info(f"KPIs gespeichert: {kpis_file}")
            
            return {
                "element_f1": f1_elements,
                "connection_f1": f1_connections,
                "element_precision": precision_elements,
                "element_recall": recall_elements,
                "connection_precision": precision_connections,
                "connection_recall": recall_connections
            }
        except Exception as e:
            logger.error(f"Fehler bei KPI-Berechnung: {e}", exc_info=True)
            return {
                "element_f1": 0.0,
                "connection_f1": 0.0,
                "error": 1.0
            }
    else:
        # No ground truth - return structural KPIs only
        elements = result_dict.get('elements', [])
        connections = result_dict.get('connections', [])
        logger.info(f"[DONE] Test '{test_name}' Abgeschlossen (ohne Ground Truth):")
        logger.info(f"  Elemente: {len(elements)}")
        logger.info(f"  Verbindungen: {len(connections)}")
        logger.info("=" * 60)
        
        return {
            "element_count": len(elements),
            "connection_count": len(connections)
        }


def get_test_overrides(test_name: str) -> Optional[Dict[str, Any]]:
    """
    Definiert die 'params_override' Dictionaries für jeden Testfall.
    
    Args:
        test_name: Name des Tests
        
    Returns:
        Dictionary mit Parameter-Overrides oder None
    """
    
    # Basis-Deaktivierung: Alle optionalen Phasen aus
    BASE_DEACTIVATED = {
        "use_swarm_analysis": False,
        "use_monolith_analysis": False,
        "use_fusion": False,
        "use_predictive_completion": False,
        "use_polyline_refinement": False,
        "use_self_correction_loop": False,
        "use_post_processing": True,  # Phase 4 immer aktiv (KPIs, CGM)
    }
    
    if test_name == "Test 1: Baseline Phase 1 (Legenden-Erkennung)":
        return {
            **BASE_DEACTIVATED,
            # Nur Phase 1 (Pre-Analysis) läuft
            # Alle anderen Phasen deaktiviert
        }
    
    if test_name == "Test 2: Baseline Simple P&ID (Monolith-All)":
        return {
            **BASE_DEACTIVATED,
            "use_monolith_analysis": True,
            # Monolith findet Elemente + Verbindungen
        }
    
    if test_name == "Test 3: Baseline Swarm-Only":
        return {
            **BASE_DEACTIVATED,
            "use_swarm_analysis": True,
            # Swarm findet nur Elemente
        }
    
    if test_name == "Test 4: Baseline Complex P&ID (Spezialisten-Kette)":
        return {
            **BASE_DEACTIVATED,
            "use_swarm_analysis": True,
            "use_monolith_analysis": True,
            "use_fusion": True,
            # Swarm -> Monolith (Connect-Only) -> Fusion
        }
    
    if test_name == "Test 5a: Test 4 + Predictive (2d)":
        overrides = get_test_overrides("Test 4: Baseline Complex P&ID (Spezialisten-Kette)")
        if overrides:
            # CRITICAL FIX: Test 5a should ONLY test Predictive Completion (Phase 2d)
            # Deactivate other phases to isolate Predictive Completion
            overrides["use_predictive_completion"] = True  # ✅ AKTIVIERT
            overrides["use_polyline_refinement"] = False  # Deaktiviert (nur Predictive testen)
            overrides["use_self_correction_loop"] = False  # Deaktiviert (nur Predictive testen)
        return overrides
    
    if test_name == "Test 5b: Test 4 + Polyline (2e)":
        overrides = get_test_overrides("Test 4: Baseline Complex P&ID (Spezialisten-Kette)")
        if overrides:
            # CRITICAL FIX: Test 5b should ONLY test Polyline Refinement (Phase 2e)
            # Deactivate Predictive to avoid Phase 2d crash
            overrides["use_predictive_completion"] = False  # Deaktiviert (nur Polyline testen)
            overrides["use_polyline_refinement"] = True  # ✅ AKTIVIERT
            overrides["use_self_correction_loop"] = False  # Deaktiviert (nur Polyline testen)
        return overrides
    
    if test_name == "Test 5c: Test 4 + Self-Correction (3)":
        overrides = get_test_overrides("Test 4: Baseline Complex P&ID (Spezialisten-Kette)")
        if overrides:
            # CRITICAL FIX: Test 5c should test Self-Correction (Phase 3)
            # Keep Test 4 baseline (Swarm + Monolith + Fusion + Polyline)
            # But add Self-Correction to fix errors
            overrides["use_predictive_completion"] = False  # Deaktiviert (nur Self-Correction testen)
            overrides["use_polyline_refinement"] = True  # ✅ AKTIVIERT (für Test 4 baseline)
            overrides["use_self_correction_loop"] = True  # ✅ AKTIVIERT (Phase 3)
            # WICHTIGER FIX: Min Score auf 90.0 setzen, damit Phase 3 überhaupt läuft
            overrides["self_correction_min_quality_score"] = 90.0
        return overrides
    
    return None


def main():
    parser = argparse.ArgumentParser(description="P&ID Analyzer Strategy Validation Harness")
    parser.add_argument(
        "--test", 
        type=str, 
        required=True, 
        help="Name des auszuführenden Tests (z.B. 'Test 2', 'Test 4', 'Test 5a', 'all')"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_TO_TEST,
        help="Pfad zum Testbild (Standard: data/input/Einfaches P&I.png)"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=GROUND_TRUTH,
        help="Pfad zur Ground Truth (Standard: data/ground_truth/Einfaches P&I.json)"
    )
    
    args = parser.parse_args()
    
    # --- Create base output directory ---
    output_base = Path(OUTPUT_DIR_BASE)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # --- Validate test configuration ---
    validation_results = validate_test_configuration()
    
    # Save validation results to summaries subdirectory
    summaries_dir = output_base / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    validation_file = summaries_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(validation_file, 'w', encoding='utf-8') as f:
        json_dump_safe(validation_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Validierungsergebnisse gespeichert: {validation_file}")
    
    if not validation_results["valid"]:
        logger.error("Test-Konfiguration ist ungültig. Abbruch.")
        logger.error("Fehler:")
        for error in validation_results["errors"]:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # --- Service-Initialisierung ---
    try:
        config_service = ConfigService()
        config = config_service.get_config().model_dump()
        
        # GCP-Credentials aus Umgebungsvariablen
        project_id = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_LOCATION", "us-central1")
        
        if not project_id:
            logger.error("GCP_PROJECT_ID nicht gesetzt. Abbruch.")
            sys.exit(1)
        
        llm_client = LLMClient(project_id, location, config)
        
        element_type_list = config_service.get_path('element_type_list')
        learning_db = config_service.get_path('learning_db')
        
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list),
            learning_db_path=str(learning_db),
            llm_handler=llm_client,
            config=config
        )
        
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service
        )
        
        logger.info("Services erfolgreich initialisiert")
        
    except Exception as e:
        logger.error(f"Fehler bei der Initialisierung der Services: {e}", exc_info=True)
        sys.exit(1)
    
    # --- Testbild und Ground Truth pro Test bestimmen ---
    # Wenn --test "all", verwenden wir die Standard-Pfade
    # Sonst verwenden wir die Test-spezifischen Pfade
    if args.test.lower() != 'all':
        # Test-spezifische Pfade verwenden
        test_key = args.test
        if test_key in TEST_IMAGES:
            image_path = TEST_IMAGES[test_key]
            gt_path = TEST_GROUND_TRUTH.get(test_key, args.ground_truth)
        else:
            # Fallback zu Standard-Pfaden
            image_path = args.image
            gt_path = args.ground_truth
    else:
        # Für "all" verwenden wir Standard-Pfade (werden pro Test überschrieben)
        image_path = args.image
        gt_path = args.ground_truth
    
    # --- Ground Truth laden ---
    gt_data = load_ground_truth(gt_path)
    if not gt_data:
        logger.warning("Keine Ground Truth verfügbar. Tests werden ohne Validierung ausgeführt.")

    # --- Testbild prüfen ---
    if not Path(image_path).exists():
        # Try relative to project root
        image_path_full = project_root / image_path
        if not image_path_full.exists():
            logger.error(f"Testbild nicht gefunden: {image_path}")
            sys.exit(1)
        image_path = str(image_path_full)
    else:
        image_path = str(Path(image_path).resolve())
    
    # --- Tests ausführen ---
    test_definitions = {
        "Test 1": "Test 1: Baseline Phase 1 (Legenden-Erkennung)",
        "Test 2": "Test 2: Baseline Simple P&ID (Monolith-All)",
        "Test 3": "Test 3: Baseline Swarm-Only",
        "Test 4": "Test 4: Baseline Complex P&ID (Spezialisten-Kette)",
        "Test 5a": "Test 5a: Test 4 + Predictive (2d)",
        "Test 5b": "Test 5b: Test 4 + Polyline (2e)",
        "Test 5c": "Test 5c: Test 4 + Self-Correction (3)",
    }
    
    # Test-Reihenfolge (wichtig für Abhängigkeiten)
    test_order = ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5a", "Test 5b", "Test 5c"]
    
    results_summary = {}
    
    if args.test.lower() == 'all':
        # Alle Tests in der richtigen Reihenfolge
        test_keys_to_run = [key for key in test_order if key in test_definitions]
    elif args.test in test_definitions:
        test_keys_to_run = [args.test]
    else:
        logger.error(f"Test '{args.test}' nicht gefunden. Verfügbar: {list(test_definitions.keys())} oder 'all'")
        sys.exit(1)
    
    logger.info(f"Führe {len(test_keys_to_run)} Test(s) aus...")
    
    for test_key in test_keys_to_run:
        test_name = test_definitions[test_key]
        
        # Test-spezifisches Bild und Ground Truth verwenden
        test_image = TEST_IMAGES.get(test_key, image_path)
        test_gt = TEST_GROUND_TRUTH.get(test_key, args.ground_truth)
        
        # Pfad prüfen und auflösen
        if not Path(test_image).exists():
            test_image_full = project_root / test_image
            if test_image_full.exists():
                test_image = str(test_image_full)
            else:
                logger.error(f"Testbild für {test_key} nicht gefunden: {test_image}")
                continue
        else:
            test_image = str(Path(test_image).resolve())
        
        # Ground Truth für diesen Test laden
        test_gt_data = load_ground_truth(test_gt)
        if not test_gt_data:
            logger.warning(f"Keine Ground Truth für {test_key} verfügbar. Test läuft ohne Validierung.")
        
        overrides = get_test_overrides(test_name)
        if overrides:
            logger.info(f"Verwende Bild: {test_image}")
            logger.info(f"Verwende Ground Truth: {test_gt}")
            kpis = run_test(test_name, coordinator, test_image, test_gt_data, overrides)
            results_summary[test_name] = kpis
        else:
            logger.warning(f"Keine Konfiguration für '{test_name}' gefunden.")
    
    # --- Finale Zusammenfassung ---
    logger.info("=" * 60)
    logger.info("FINALE KPI-ZUSAMMENFASSUNG")
    logger.info("=" * 60)
    
    for test_name, kpis in results_summary.items():
        logger.info(f"[{test_name}]:")
        if "element_f1" in kpis:
            logger.info(f"  Element F1:    {kpis.get('element_f1', 0.0):.4f}")
            logger.info(f"  Element Precision: {kpis.get('element_precision', 0.0):.4f}")
            logger.info(f"  Element Recall:    {kpis.get('element_recall', 0.0):.4f}")
            logger.info(f"  Connection F1: {kpis.get('connection_f1', 0.0):.4f}")
            logger.info(f"  Connection Precision: {kpis.get('connection_precision', 0.0):.4f}")
            logger.info(f"  Connection Recall:    {kpis.get('connection_recall', 0.0):.4f}")
        else:
            logger.info(f"  Elemente: {kpis.get('element_count', 0)}")
            logger.info(f"  Verbindungen: {kpis.get('connection_count', 0)}")
    
    # Save summary to file in summaries subdirectory (to avoid cluttering main folder)
    summaries_dir = Path(OUTPUT_DIR_BASE) / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summaries_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "image_path": image_path,
        "ground_truth": args.ground_truth,
        "results": results_summary,
        "test_configuration": {
            "test_images": TEST_IMAGES,
            "test_ground_truth": TEST_GROUND_TRUTH
        }
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json_dump_safe(summary_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Zusammenfassung gespeichert: {summary_file}")
    
    # CRITICAL FIX: Cleanup ThreadPoolExecutor properly (wait=True to ensure all tasks complete)
    try:
        if hasattr(llm_client, 'close'):
            llm_client.close()
            logger.debug("LLMClient closed successfully")
        elif hasattr(llm_client, 'timeout_executor'):
            llm_client.timeout_executor.shutdown(wait=True, cancel_futures=False)
            logger.debug("ThreadPoolExecutor shut down")
    except Exception as e:
        logger.warning(f"Fehler beim Cleanup: {e}")


if __name__ == "__main__":
    main()

