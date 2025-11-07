"""
Overnight Optimization Script - Automated A/B testing and optimization.

This script runs automated A/B tests between strategies (simple_whole_image vs default_flash)
over an extended period (8 hours by default), calculates KPIs, and generates comprehensive reports.

Features:
- A/B tests between strategies
- Automatic KPI calculation with Ground Truth
- Robust error handling with automatic restart
- Detailed logging and reporting
- HTML and JSON reports
- Configurable duration (default: 8 hours)
"""

import sys
import json
import os
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import argparse
from datetime import datetime, timedelta
from itertools import product

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.json_encoder import PydanticJSONEncoder, json_dump_safe

# Load .env file automatically
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
TEST_IMAGES = {
    "simple": "training_data/simple_pids/Einfaches P&I.png",
    "complex": "training_data/complex_pids/page_1_original.png"
}

TEST_GROUND_TRUTH = {
    "simple": "training_data/simple_pids/Einfaches P&I_truth.json",
    "complex": "training_data/complex_pids/page_1_original_truth_cgm.json"
}

# Parameter-Optimierung
PARAMETER_COMBINATIONS = {
    'iou_match_threshold': [0.3, 0.4, 0.5, 0.6],
    'confidence_threshold': [0.5, 0.6, 0.7, 0.8],
    'self_correction_min_quality_score': [85.0, 90.0, 95.0]
}

# Strategien zu testen
STRATEGIES_TO_TEST = ['simple_whole_image', 'default_flash']

# Setup Logging
LoggingService.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_prerequisites() -> Tuple[bool, List[str]]:
    """
    Validiert Voraussetzungen für nächtlichen Lauf.
    
    Returns:
        Tuple (success, errors)
    """
    errors = []
    
    # Prüfe Test-Bilder
    for key, image_path in TEST_IMAGES.items():
        img_file = project_root / image_path
        if not img_file.exists():
            errors.append(f"Test-Bild nicht gefunden: {image_path}")
    
    # Prüfe Ground Truth
    for key, gt_path in TEST_GROUND_TRUTH.items():
        gt_file = project_root / gt_path
        if not gt_file.exists():
            errors.append(f"Ground Truth nicht gefunden: {gt_path}")
    
    # Prüfe GCP-Credentials
    if not os.getenv('GCP_PROJECT_ID'):
        errors.append("GCP_PROJECT_ID nicht gesetzt")
    
    # Prüfe Viewshots (optional - Warnung, kein Fehler)
    viewshot_dir = project_root / "training_data" / "viewshot_examples"
    if not viewshot_dir.exists():
        logger.warning(f"Viewshot-Verzeichnis nicht gefunden: {viewshot_dir}")
    
    # Prüfe Pre-Training (optional - Warnung, kein Fehler)
    learning_db = project_root / "training_data" / "learning_db.json"
    if not learning_db.exists():
        logger.warning(f"Learning DB nicht gefunden: {learning_db} (Pre-Training sollte ausgeführt werden)")
    
    return len(errors) == 0, errors


def load_ground_truth(gt_path: str) -> Optional[Dict[str, Any]]:
    """Lädt die Ground-Truth-Daten."""
    try:
        gt_file = Path(gt_path)
        if not gt_file.exists():
            gt_file = project_root / gt_path
            if not gt_file.exists():
                logger.warning(f"Ground Truth nicht gefunden: {gt_path}")
                return None
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden der Ground Truth: {e}")
        return None


def get_strategy_config(config: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """
    Lädt Strategie-Konfiguration aus config.yaml.
    
    Args:
        config: Config-Dictionary
        strategy_name: Name der Strategie (z.B. 'simple_whole_image')
        
    Returns:
        Strategie-Konfiguration
    """
    strategies = config.get('strategies', {})
    strategy = strategies.get(strategy_name, {})
    
    if not strategy:
        logger.warning(f"Strategie '{strategy_name}' nicht gefunden, verwende default_flash")
        strategy = strategies.get('default_flash', {})
    
    return strategy


def run_ab_test(
    coordinator: PipelineCoordinator,
    image_path: str,
    gt_data: Optional[Dict[str, Any]],
    strategy_name: str,
    strategy_config: Dict[str, Any],
    parameters: Dict[str, Any],
    output_dir: Path,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Führt einen A/B-Test mit gegebener Strategie und Parametern durch.
    
    Args:
        coordinator: Pipeline Coordinator
        image_path: Pfad zum Testbild
        gt_data: Ground Truth Daten (optional)
        strategy_name: Name der Strategie
        strategy_config: Strategie-Konfiguration
        parameters: Parameter-Overrides
        output_dir: Output-Verzeichnis
        max_retries: Maximale Anzahl Wiederholungsversuche bei Fehlern
        
    Returns:
        Test-Ergebnis Dictionary
    """
    test_output_dir = output_dir / "test_results" / strategy_name
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Erstelle eindeutigen Dateinamen
    image_name = Path(image_path).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = test_output_dir / f"{image_name}_{timestamp}.json"
    
    # Kombiniere Strategie-Konfiguration mit Parametern
    params_override = {
        **strategy_config,
        **parameters,
        'test_name': f"{strategy_name}_{image_name}",
        'test_description': f"A/B Test: {strategy_name} with parameters {parameters}"
    }
    
    # Retry-Logik
    for attempt in range(max_retries):
        try:
            # Circuit Breaker Reset vor jedem Test
            if hasattr(coordinator, 'llm_client') and hasattr(coordinator.llm_client, 'retry_handler'):
                if hasattr(coordinator.llm_client.retry_handler, 'circuit_breaker'):
                    coordinator.llm_client.retry_handler.circuit_breaker.reset()
                    logger.debug(f"Circuit Breaker reset for attempt {attempt + 1}")
            
            logger.info(f"[{strategy_name}] Starte Test (Versuch {attempt + 1}/{max_retries})...")
            logger.info(f"  Bild: {image_path}")
            logger.info(f"  Parameter: {parameters}")
            
            # Führe Analyse durch
            result = coordinator.process(
                image_path=image_path,
                output_dir=str(test_output_dir),
                params_override=params_override
            )
            
            # Konvertiere Ergebnis zu Dict
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            elif hasattr(result, 'dict'):
                result_dict = result.dict()
            else:
                result_dict = result if isinstance(result, dict) else {
                    'elements': getattr(result, 'elements', []),
                    'connections': getattr(result, 'connections', [])
                }
            
            # Berechne KPIs
            kpis = {}
            if gt_data:
                kpi_calc = KPICalculator()
                kpis = kpi_calc.calculate_comprehensive_kpis(
                    analysis_data=result_dict,
                    truth_data=gt_data
                )
                logger.info(f"[{strategy_name}] KPIs berechnet:")
                logger.info(f"  Element F1: {kpis.get('element_f1', 0.0):.4f}")
                logger.info(f"  Connection F1: {kpis.get('connection_f1', 0.0):.4f}")
                logger.info(f"  Quality Score: {kpis.get('quality_score', 0.0):.2f}")
            
            # Speichere Ergebnis
            test_result = {
                'strategy': strategy_name,
                'image_path': image_path,
                'parameters': parameters,
                'timestamp': datetime.now().isoformat(),
                'result': result_dict,
                'kpis': kpis,
                'success': True
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json_dump_safe(test_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[{strategy_name}] Test erfolgreich abgeschlossen")
            logger.info(f"  Ergebnis gespeichert: {result_file}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"[{strategy_name}] Test fehlgeschlagen (Versuch {attempt + 1}/{max_retries}): {e}", exc_info=True)
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                logger.info(f"  Warte {wait_time}s vor erneutem Versuch...")
                time.sleep(wait_time)
            else:
                # Alle Versuche fehlgeschlagen
                error_result = {
                    'strategy': strategy_name,
                    'image_path': image_path,
                    'parameters': parameters,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'success': False
                }
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json_dump_safe(error_result, f, indent=2, ensure_ascii=False)
                
                return error_result
    
    return {'success': False, 'error': 'Max retries exceeded'}


def calculate_kpis(test_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrahiert KPIs aus Test-Ergebnis.
    
    Args:
        test_result: Test-Ergebnis Dictionary
        
    Returns:
        KPI Dictionary
    """
    kpis = test_result.get('kpis', {})
    
    return {
        'element_f1': kpis.get('element_f1', 0.0),
        'element_precision': kpis.get('element_precision', 0.0),
        'element_recall': kpis.get('element_recall', 0.0),
        'connection_f1': kpis.get('connection_f1', 0.0),
        'connection_precision': kpis.get('connection_precision', 0.0),
        'connection_recall': kpis.get('connection_recall', 0.0),
        'graph_edit_distance': kpis.get('normalized_graph_edit_distance', 1.0),
        'graph_similarity_score': kpis.get('graph_similarity_score', 0.0),
        'type_accuracy': kpis.get('type_accuracy', 0.0),
        'quality_score': kpis.get('quality_score', 0.0)
    }


def generate_parameter_combinations() -> List[Dict[str, Any]]:
    """
    Generiert alle Parameter-Kombinationen für Optimierung.
    
    Returns:
        Liste von Parameter-Dictionaries
    """
    param_names = list(PARAMETER_COMBINATIONS.keys())
    param_values = list(PARAMETER_COMBINATIONS.values())
    
    combinations = []
    for combo in product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)
    
    return combinations


def generate_html_report(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    start_time: datetime,
    end_time: datetime
) -> Path:
    """
    Generiert HTML-Report mit allen Ergebnissen.
    
    Args:
        all_results: Liste aller Test-Ergebnisse
        output_dir: Output-Verzeichnis
        start_time: Start-Zeitpunkt
        end_time: End-Zeitpunkt
        
    Returns:
        Pfad zur HTML-Datei
    """
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = reports_dir / f"report_{timestamp}.html"
    
    # Gruppiere Ergebnisse nach Strategie
    results_by_strategy = {}
    for result in all_results:
        strategy = result.get('strategy', 'unknown')
        if strategy not in results_by_strategy:
            results_by_strategy[strategy] = []
        results_by_strategy[strategy].append(result)
    
    # Berechne Statistiken
    stats_by_strategy = {}
    for strategy, results in results_by_strategy.items():
        successful_results = [r for r in results if r.get('success', False)]
        if successful_results:
            kpis_list = [calculate_kpis(r) for r in successful_results]
            
            stats_by_strategy[strategy] = {
                'total_tests': len(results),
                'successful_tests': len(successful_results),
                'avg_element_f1': sum(k.get('element_f1', 0.0) for k in kpis_list) / len(kpis_list) if kpis_list else 0.0,
                'avg_connection_f1': sum(k.get('connection_f1', 0.0) for k in kpis_list) / len(kpis_list) if kpis_list else 0.0,
                'avg_quality_score': sum(k.get('quality_score', 0.0) for k in kpis_list) / len(kpis_list) if kpis_list else 0.0,
                'best_element_f1': max((k.get('element_f1', 0.0) for k in kpis_list), default=0.0),
                'best_connection_f1': max((k.get('connection_f1', 0.0) for k in kpis_list), default=0.0),
                'best_quality_score': max((k.get('quality_score', 0.0) for k in kpis_list), default=0.0)
            }
    
    # Finde beste Parameter-Kombination
    best_result = None
    best_score = 0.0
    for result in all_results:
        if result.get('success', False):
            kpis = calculate_kpis(result)
            score = kpis.get('quality_score', 0.0)
            if score > best_score:
                best_score = score
                best_result = result
    
    # Generiere HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Overnight Optimization Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .best {{ background-color: #ffeb3b; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Overnight Optimization Report</h1>
    <p><strong>Start:</strong> {start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>End:</strong> {end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Duration:</strong> {(end_time - start_time).total_seconds() / 3600:.2f} hours</p>
    <p><strong>Total Tests:</strong> {len(all_results)}</p>
    
    <h2>Strategy Comparison</h2>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Total Tests</th>
            <th>Successful</th>
            <th>Avg Element F1</th>
            <th>Avg Connection F1</th>
            <th>Avg Quality Score</th>
            <th>Best Quality Score</th>
        </tr>
"""
    
    for strategy, stats in stats_by_strategy.items():
        html_content += f"""
        <tr>
            <td>{strategy}</td>
            <td>{stats['total_tests']}</td>
            <td>{stats['successful_tests']}</td>
            <td>{stats['avg_element_f1']:.4f}</td>
            <td>{stats['avg_connection_f1']:.4f}</td>
            <td>{stats['avg_quality_score']:.2f}</td>
            <td>{stats['best_quality_score']:.2f}</td>
        </tr>
"""
    
    html_content += """
    </table>
    
    <h2>Best Configuration</h2>
"""
    
    if best_result:
        best_kpis = calculate_kpis(best_result)
        html_content += f"""
    <table>
        <tr>
            <th>Strategy</th>
            <th>Parameters</th>
            <th>Element F1</th>
            <th>Connection F1</th>
            <th>Quality Score</th>
        </tr>
        <tr class="best">
            <td>{best_result.get('strategy', 'unknown')}</td>
            <td>{json.dumps(best_result.get('parameters', {}), indent=2)}</td>
            <td>{best_kpis.get('element_f1', 0.0):.4f}</td>
            <td>{best_kpis.get('connection_f1', 0.0):.4f}</td>
            <td>{best_kpis.get('quality_score', 0.0):.2f}</td>
        </tr>
    </table>
"""
    
    html_content += """
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Image</th>
            <th>Parameters</th>
            <th>Element F1</th>
            <th>Connection F1</th>
            <th>Quality Score</th>
            <th>Status</th>
        </tr>
"""
    
    for result in all_results:
        status_class = "success" if result.get('success', False) else "error"
        status_text = "Success" if result.get('success', False) else "Error"
        
        kpis = calculate_kpis(result) if result.get('success', False) else {}
        
        html_content += f"""
        <tr class="{status_class}">
            <td>{result.get('strategy', 'unknown')}</td>
            <td>{Path(result.get('image_path', '')).name}</td>
            <td>{json.dumps(result.get('parameters', {}), indent=2)}</td>
            <td>{kpis.get('element_f1', 0.0):.4f}</td>
            <td>{kpis.get('connection_f1', 0.0):.4f}</td>
            <td>{kpis.get('quality_score', 0.0):.2f}</td>
            <td>{status_text}</td>
        </tr>
"""
    
    html_content += """
    </table>
</body>
</html>
"""
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML-Report generiert: {html_file}")
    return html_file


def generate_json_report(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    start_time: datetime,
    end_time: datetime
) -> Path:
    """
    Generiert JSON-Report mit allen Ergebnissen.
    
    Args:
        all_results: Liste aller Test-Ergebnisse
        output_dir: Output-Verzeichnis
        start_time: Start-Zeitpunkt
        end_time: End-Zeitpunkt
        
    Returns:
        Pfad zur JSON-Datei
    """
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = reports_dir / f"report_{timestamp}.json"
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_hours': (end_time - start_time).total_seconds() / 3600,
        'total_tests': len(all_results),
        'successful_tests': len([r for r in all_results if r.get('success', False)]),
        'failed_tests': len([r for r in all_results if not r.get('success', False)]),
        'results': all_results,
        'summary': {}
    }
    
    # Berechne Zusammenfassung
    successful_results = [r for r in all_results if r.get('success', False)]
    if successful_results:
        kpis_list = [calculate_kpis(r) for r in successful_results]
        
        report_data['summary'] = {
            'avg_element_f1': sum(k.get('element_f1', 0.0) for k in kpis_list) / len(kpis_list),
            'avg_connection_f1': sum(k.get('connection_f1', 0.0) for k in kpis_list) / len(kpis_list),
            'avg_quality_score': sum(k.get('quality_score', 0.0) for k in kpis_list) / len(kpis_list),
            'best_quality_score': max((k.get('quality_score', 0.0) for k in kpis_list), default=0.0)
        }
        
        # Finde beste Parameter-Kombination
        best_result = max(successful_results, key=lambda r: calculate_kpis(r).get('quality_score', 0.0))
        report_data['summary']['best_configuration'] = {
            'strategy': best_result.get('strategy'),
            'parameters': best_result.get('parameters'),
            'kpis': calculate_kpis(best_result)
        }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json_dump_safe(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON-Report generiert: {json_file}")
    return json_file


def run_overnight_loop(
    coordinator: PipelineCoordinator,
    config: Dict[str, Any],
    output_dir: Path,
    duration_hours: float = 8.0
) -> List[Dict[str, Any]]:
    """
    Haupt-Loop für nächtlichen Optimierungs-Lauf.
    
    Args:
        coordinator: Pipeline Coordinator
        config: Config-Dictionary
        output_dir: Output-Verzeichnis
        duration_hours: Dauer in Stunden
        
    Returns:
        Liste aller Test-Ergebnisse
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)
    
    logger.info("=" * 60)
    logger.info("STARTE NÄCHTLICHEN OPTIMIERUNGS-LAUF")
    logger.info("=" * 60)
    logger.info(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Ende: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dauer: {duration_hours} Stunden")
    logger.info("=" * 60)
    
    all_results = []
    parameter_combinations = generate_parameter_combinations()
    
    logger.info(f"Strategien zu testen: {STRATEGIES_TO_TEST}")
    logger.info(f"Parameter-Kombinationen: {len(parameter_combinations)}")
    logger.info(f"Test-Bilder: {len(TEST_IMAGES)}")
    logger.info(f"Gesamt-Tests: {len(STRATEGIES_TO_TEST) * len(parameter_combinations) * len(TEST_IMAGES)}")
    
    iteration = 0
    
    while datetime.now() < end_time:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")
        logger.info(f"Verbleibende Zeit: {(end_time - datetime.now()).total_seconds() / 3600:.2f} Stunden")
        
        # Teste jede Strategie
        for strategy_name in STRATEGIES_TO_TEST:
            if datetime.now() >= end_time:
                break
            
            logger.info(f"\n--- Strategie: {strategy_name} ---")
            strategy_config = get_strategy_config(config, strategy_name)
            
            # Teste jede Parameter-Kombination
            for parameters in parameter_combinations:
                if datetime.now() >= end_time:
                    break
                
                # Teste jedes Bild
                for image_key, image_path in TEST_IMAGES.items():
                    if datetime.now() >= end_time:
                        break
                    
                    # Vollständiger Pfad
                    full_image_path = project_root / image_path
                    if not full_image_path.exists():
                        logger.warning(f"Bild nicht gefunden: {full_image_path}")
                        continue
                    
                    # Ground Truth laden
                    gt_path = TEST_GROUND_TRUTH.get(image_key)
                    gt_data = load_ground_truth(gt_path) if gt_path else None
                    
                    # Führe A/B-Test durch
                    result = run_ab_test(
                        coordinator=coordinator,
                        image_path=str(full_image_path),
                        gt_data=gt_data,
                        strategy_name=strategy_name,
                        strategy_config=strategy_config,
                        parameters=parameters,
                        output_dir=output_dir,
                        max_retries=3
                    )
                    
                    all_results.append(result)
                    
                    # Active Learning: Nur für erfolgreiche Tests mit Score > 0.8
                    if result.get('success', False):
                        kpis = calculate_kpis(result)
                        quality_score = kpis.get('quality_score', 0.0)
                        
                        if quality_score > 0.8:
                            logger.info(f"[Active Learning] Test erfolgreich (Score: {quality_score:.2f} > 0.8) - Lernen aktiviert")
                            # TODO: Active Learning Integration hier
                            # coordinator.knowledge_manager.add_correction(...)
                        else:
                            logger.debug(f"[Active Learning] Test erfolgreich, aber Score zu niedrig ({quality_score:.2f} <= 0.8) - Lernen deaktiviert")
                    
                    # ThreadPool Cleanup nach jedem Test
                    try:
                        if hasattr(coordinator, 'llm_client') and coordinator.llm_client:
                            # ThreadPool wird automatisch geschlossen, wenn nicht mehr benötigt
                            pass
                    except Exception as e:
                        logger.warning(f"Fehler beim Cleanup: {e}")
        
        # Kurze Pause zwischen Iterationen
        if datetime.now() < end_time:
            logger.info(f"\nIteration {iteration} abgeschlossen. Warte 60s vor nächster Iteration...")
            time.sleep(60)
    
    logger.info("\n" + "=" * 60)
    logger.info("NÄCHTLICHER OPTIMIERUNGS-LAUF ABGESCHLOSSEN")
    logger.info("=" * 60)
    logger.info(f"Gesamt-Iterationen: {iteration}")
    logger.info(f"Gesamt-Tests: {len(all_results)}")
    logger.info(f"Erfolgreiche Tests: {len([r for r in all_results if r.get('success', False)])}")
    logger.info(f"Fehlgeschlagene Tests: {len([r for r in all_results if not r.get('success', False)])}")
    
    return all_results


def main():
    """Hauptfunktion."""
    parser = argparse.ArgumentParser(description="Overnight Optimization Script")
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Dauer in Stunden (Standard: 8.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/overnight_optimization",
        help="Output-Verzeichnis (Standard: outputs/overnight_optimization)"
    )
    
    args = parser.parse_args()
    
    # Validiere Voraussetzungen
    logger.info("Validiere Voraussetzungen...")
    success, errors = validate_prerequisites()
    
    if not success:
        logger.error("Voraussetzungen nicht erfüllt:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    logger.info("Alle Voraussetzungen erfüllt!")
    
    # Erstelle Output-Verzeichnis
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Logging für nächtlichen Lauf
    log_file = output_dir / "logs" / f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    LoggingService.setup_logging(
        log_level=logging.INFO,
        log_file=str(log_file)
    )
    
    # Service-Initialisierung
    try:
        config_service = ConfigService()
        config = config_service.get_config().model_dump() if hasattr(config_service.get_config(), 'model_dump') else config_service.get_raw_config()
        
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
    
    # Führe nächtlichen Lauf durch
    start_time = datetime.now()
    
    try:
        all_results = run_overnight_loop(
            coordinator=coordinator,
            config=config,
            output_dir=output_dir,
            duration_hours=args.duration
        )
        
        end_time = datetime.now()
        
        # Generiere Reports
        logger.info("\nGeneriere Reports...")
        html_report = generate_html_report(all_results, output_dir, start_time, end_time)
        json_report = generate_json_report(all_results, output_dir, start_time, end_time)
        
        logger.info("\n" + "=" * 60)
        logger.info("REPORTS GENERIERT")
        logger.info("=" * 60)
        logger.info(f"HTML-Report: {html_report}")
        logger.info(f"JSON-Report: {json_report}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nNächtlicher Lauf durch Benutzer abgebrochen")
        end_time = datetime.now()
        
        # Generiere Reports auch bei Abbruch
        if 'all_results' in locals() and all_results:
            logger.info("Generiere Reports für bisherige Ergebnisse...")
            html_report = generate_html_report(all_results, output_dir, start_time, end_time)
            json_report = generate_json_report(all_results, output_dir, start_time, end_time)
            logger.info(f"HTML-Report: {html_report}")
            logger.info(f"JSON-Report: {json_report}")
    
    except Exception as e:
        logger.error(f"Kritischer Fehler im nächtlichen Lauf: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # CRITICAL: Cleanup ThreadPoolExecutor
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

