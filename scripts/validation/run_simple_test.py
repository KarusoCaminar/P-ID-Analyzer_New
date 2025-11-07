"""
Einfacher Test-Runner - Testet eine einzelne Konfiguration um zu prüfen ob alles funktioniert
"""

import sys
import json
import os
import logging
from pathlib import Path
from datetime import datetime

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.json_encoder import json_dump_safe
from src.services.config_service import ConfigService
from src.services.logging_service import LoggingService
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.evaluation.kpi_calculator import KPICalculator

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

# Setup Logging
LoggingService.setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)

# Test-Konfiguration - NUR EIN TEST
TEST_IMAGE = project_root / "training_data" / "simple_pids" / "Einfaches P&I.png"
TEST_GROUND_TRUTH = project_root / "training_data" / "simple_pids" / "Einfaches P&I_truth.json"
STRATEGY = "simple_whole_image"
PARAMETERS = {
    'iou_match_threshold': 0.5,
    'confidence_threshold': 0.6,
    'self_correction_min_quality_score': 90.0
}

def load_ground_truth(gt_path: Path):
    """Lädt Ground Truth."""
    if not gt_path.exists():
        logger.warning(f"Ground Truth nicht gefunden: {gt_path}")
        return None
    with open(gt_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Führt einen einfachen Test durch."""
    logger.info("=" * 70)
    logger.info("EINFACHER TEST-RUNNER")
    logger.info("=" * 70)
    logger.info(f"Bild: {TEST_IMAGE}")
    logger.info(f"Strategie: {STRATEGY}")
    logger.info(f"Parameter: {PARAMETERS}")
    logger.info("=" * 70)
    
    # Prüfe ob Bild existiert
    if not TEST_IMAGE.exists():
        logger.error(f"Test-Bild nicht gefunden: {TEST_IMAGE}")
        sys.exit(1)
    
    # Initialisiere Services
    logger.info("Initialisiere Services...")
    try:
        config_service = ConfigService()
        config = config_service.get_raw_config()
        
        project_id = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_LOCATION", "us-central1")
        
        if not project_id:
            logger.error("GCP_PROJECT_ID nicht gesetzt!")
            sys.exit(1)
        
        logger.info("Erstelle LLM Client...")
        llm_client = LLMClient(project_id, location, config)
        
        logger.info("Erstelle Knowledge Manager...")
        element_type_list = config_service.get_path('element_type_list')
        learning_db = config_service.get_path('learning_db')
        
        knowledge_manager = KnowledgeManager(
            element_type_list_path=str(element_type_list),
            learning_db_path=str(learning_db),
            llm_handler=llm_client,
            config=config
        )
        
        logger.info("Erstelle Pipeline Coordinator...")
        coordinator = PipelineCoordinator(
            llm_client=llm_client,
            knowledge_manager=knowledge_manager,
            config_service=config_service
        )
        
        logger.info("[OK] Services erfolgreich initialisiert")
        
    except Exception as e:
        logger.error(f"[FAIL] Fehler bei Initialisierung: {e}", exc_info=True)
        sys.exit(1)
    
    # Lade Strategie-Config
    logger.info(f"Lade Strategie-Konfiguration: {STRATEGY}")
    strategies = config.get('strategies', {})
    strategy_config = strategies.get(STRATEGY, {})
    
    if not strategy_config:
        logger.error(f"Strategie '{STRATEGY}' nicht gefunden!")
        sys.exit(1)
    
    logger.info(f"[OK] Strategie-Konfiguration geladen: {list(strategy_config.keys())[:5]}...")
    
    # Kombiniere Config mit Parametern
    params_override = {
        **strategy_config,
        **PARAMETERS,
        'test_name': f"{STRATEGY}_simple_test",
        'test_description': f"Simple test: {STRATEGY} with {PARAMETERS}"
    }
    
    # Lade Ground Truth
    gt_data = load_ground_truth(TEST_GROUND_TRUTH)
    if gt_data:
        logger.info(f"[OK] Ground Truth geladen: {len(gt_data.get('elements', []))} Elemente")
    else:
        logger.warning("[WARN] Ground Truth nicht verfügbar - KPI-Berechnung nicht möglich")
    
    # Output-Verzeichnis
    output_dir = project_root / "outputs" / "simple_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Führe Test durch
    logger.info("\n" + "=" * 70)
    logger.info("STARTE TEST")
    logger.info("=" * 70)
    start_time = datetime.now()
    
    try:
        logger.info(f"Starte Analyse um {start_time.strftime('%H:%M:%S')}...")
        
        result = coordinator.process(
            image_path=str(TEST_IMAGE),
            output_dir=str(output_dir),
            params_override=params_override
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        logger.info(f"[OK] Analyse abgeschlossen nach {duration:.2f} Minuten")
        
        # Konvertiere Ergebnis
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = result if isinstance(result, dict) else {
                'elements': getattr(result, 'elements', []),
                'connections': getattr(result, 'connections', [])
            }
        
        logger.info(f"  Gefundene Elemente: {len(result_dict.get('elements', []))}")
        logger.info(f"  Gefundene Verbindungen: {len(result_dict.get('connections', []))}")
        
        # Berechne KPIs
        kpis = {}
        if gt_data:
            logger.info("Berechne KPIs...")
            kpi_calc = KPICalculator()
            kpis = kpi_calc.calculate_comprehensive_kpis(
                analysis_data=result_dict,
                truth_data=gt_data
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("ERGEBNISSE")
            logger.info("=" * 70)
            logger.info(f"Element F1: {kpis.get('element_f1', 0.0):.4f}")
            logger.info(f"Element Precision: {kpis.get('element_precision', 0.0):.4f}")
            logger.info(f"Element Recall: {kpis.get('element_recall', 0.0):.4f}")
            logger.info(f"Connection F1: {kpis.get('connection_f1', 0.0):.4f}")
            logger.info(f"Quality Score: {kpis.get('quality_score', 0.0):.2f}")
            logger.info("=" * 70)
        
        # Speichere Ergebnis
        result_file = output_dir / f"test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        test_result = {
            'strategy': STRATEGY,
            'parameters': PARAMETERS,
            'image_path': str(TEST_IMAGE),
            'duration_minutes': duration,
            'timestamp': datetime.now().isoformat(),
            'result': result_dict,
            'kpis': kpis,
            'success': True
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json_dump_safe(test_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n[OK] Ergebnis gespeichert: {result_file}")
        logger.info("=" * 70)
        logger.info("TEST ERFOLGREICH ABGESCHLOSSEN")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"\n[FAIL] TEST FEHLGESCHLAGEN: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            if hasattr(llm_client, 'close'):
                llm_client.close()
        except:
            pass

if __name__ == "__main__":
    main()

