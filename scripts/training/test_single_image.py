#!/usr/bin/env python3
"""
Schneller Test mit einem einzelnen Bild - Live-Logs
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.learning.symbol_library import SymbolLibrary
from src.analyzer.learning.active_learner import ActiveLearner
from src.services.config_service import ConfigService
from PIL import Image

# Setup logging mit Live-Output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    """Test mit einem einzelnen Bild."""
    logger.info("=== TEST: Einzelnes Bild ===")
    
    # Prüfe .env
    project_id = os.getenv('GCP_PROJECT_ID')
    location = os.getenv('GCP_LOCATION')
    logger.info(f"GCP_PROJECT_ID: {project_id}")
    logger.info(f"GCP_LOCATION: {location}")
    
    if not project_id:
        logger.error("GCP_PROJECT_ID nicht gesetzt!")
        return
    
    try:
        # Initialize services
        logger.info("Initialisiere Services...")
        config_service = ConfigService()
        config = config_service.get_raw_config()
        
        # LLM Client
        logger.info("Initialisiere LLM Client...")
        llm_client = LLMClient(
            project_id=project_id,
            default_location=location,
            config=config
        )
        
        # Get paths from config
        element_type_list_path = config.get('paths', {}).get('element_type_list', 'element_type_list.json')
        learning_db_path = config.get('paths', {}).get('learning_db', 'learning_db.json')
        learned_symbols_images_dir = Path(config.get('paths', {}).get('learned_symbols_images_dir', 'learned_symbols_images'))
        
        # Knowledge Manager
        logger.info("Initialisiere Knowledge Manager...")
        knowledge_manager = KnowledgeManager(
            element_type_list_path=element_type_list_path,
            learning_db_path=learning_db_path,
            llm_handler=llm_client,
            config=config
        )
        
        # Symbol Library
        logger.info("Initialisiere Symbol Library...")
        symbol_library = SymbolLibrary(
            llm_client=llm_client,
            learning_db_path=Path(learning_db_path),
            images_dir=learned_symbols_images_dir
        )
        
        # Active Learner
        logger.info("Initialisiere Active Learner...")
        active_learner = ActiveLearner(
            knowledge_manager=knowledge_manager,
            symbol_library=symbol_library,
            llm_client=llm_client,
            config=config
        )
        
        # Test-Bild laden
        test_image_path = Path("training_data/pretraining_symbols/Pid-symbols-PDF_sammlung.png")
        logger.info(f"Lade Test-Bild: {test_image_path}")
        
        if not test_image_path.exists():
            logger.error(f"Bild nicht gefunden: {test_image_path}")
            return
        
        image = Image.open(test_image_path)
        logger.info(f"Bild geladen: {image.size} (WxH)")
        
        # Model Info (wie in run_pretraining.py)
        models_config = config.get('models', {})
        model_info = models_config.get('Google Gemini 2.5 Flash', {})
        if not model_info:
            # Fallback to first available model
            model_info = list(models_config.values())[0] if models_config else {}
        
        logger.info(f"Model: {model_info.get('id', 'unknown')}")
        
        # Test: Extrahiere Symbole
        logger.info("=== STARTE SYMBOL-EXTRAKTION ===")
        symbols = active_learner._extract_symbols_from_collection(
            collection_image=image,
            collection_path=test_image_path,
            model_info=model_info
        )
        
        logger.info(f"=== ERGEBNIS: {len(symbols)} Symbole extrahiert ===")
        
        for idx, (label, symbol_img, source) in enumerate(symbols[:5]):  # Nur erste 5 zeigen
            logger.info(f"Symbol {idx+1}: {label} ({symbol_img.size})")
        
        if len(symbols) > 5:
            logger.info(f"... und {len(symbols) - 5} weitere")
        
    except Exception as e:
        logger.error(f"FEHLER: {e}", exc_info=True)

if __name__ == "__main__":
    main()

