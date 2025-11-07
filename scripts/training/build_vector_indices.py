"""
Build Vector Indices - Pre-builds vector indices for fast loading.

This script reads learning_db.json and builds optimized .npy files
for fast vector similarity search, eliminating the startup delay.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.env_loader import load_env_automatically

# Load .env
load_env_automatically()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
LEARNING_DB_PATH = project_root / "training_data" / "learning_db.json"
INDICES_DIR = project_root / "training_data" / "indices"
SYMBOL_INDEX_FILE = INDICES_DIR / "symbol_index.npy"
SYMBOL_IDS_FILE = INDICES_DIR / "symbol_ids.json"
SYMBOL_DATA_FILE = INDICES_DIR / "symbol_data.json"
SOLUTION_INDEX_FILE = INDICES_DIR / "solution_index.npy"
SOLUTION_KEYS_FILE = INDICES_DIR / "solution_keys.json"
SOLUTION_VALUES_FILE = INDICES_DIR / "solution_values.json"


def build_vector_indices():
    """Build vector indices from learning_db.json."""
    logger.info("=" * 70)
    logger.info("BUILD VECTOR INDICES")
    logger.info("=" * 70)
    
    # Create indices directory
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load learning database
    if not LEARNING_DB_PATH.exists():
        logger.error(f"Learning database not found: {LEARNING_DB_PATH}")
        logger.error("Run pretraining first: python scripts/training/run_pretraining.py")
        sys.exit(1)
    
    logger.info(f"Loading learning database from: {LEARNING_DB_PATH}")
    file_size_mb = LEARNING_DB_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Database size: {file_size_mb:.2f} MB")
    
    try:
        with open(LEARNING_DB_PATH, 'r', encoding='utf-8') as f:
            learning_database = json.load(f)
        logger.info("✓ Learning database loaded")
    except Exception as e:
        logger.error(f"Error loading learning database: {e}", exc_info=True)
        sys.exit(1)
    
    # Build solution index (text-based embeddings)
    logger.info("\nBuilding solution vector index...")
    learned_solutions = learning_database.get("learned_solutions", {})
    logger.info(f"Found {len(learned_solutions)} learned solutions")
    
    vectors_as_lists = []
    solution_keys_filtered = []
    solution_values_filtered = []
    
    if learned_solutions:
        for key, solution_data in learned_solutions.items():
            embedding = solution_data.get("problem_embedding")
            if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                vectors_as_lists.append(embedding)
                solution_keys_filtered.append(key)
                solution_values_filtered.append(solution_data)
        
        if vectors_as_lists:
            logger.info(f"Creating solution vector array with {len(vectors_as_lists)} entries...")
            solution_vector_index = np.array(vectors_as_lists, dtype=np.float32)
            
            # Save solution index
            np.save(SOLUTION_INDEX_FILE, solution_vector_index)
            logger.info(f"✓ Saved solution index: {SOLUTION_INDEX_FILE} ({solution_vector_index.shape})")
            
            # Save solution keys
            with open(SOLUTION_KEYS_FILE, 'w', encoding='utf-8') as f:
                json.dump(solution_keys_filtered, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved solution keys: {SOLUTION_KEYS_FILE}")
            
            # Save solution values (simplified - only metadata, not full data)
            solution_metadata = [
                {
                    'key': key,
                    'solution_type': val.get('solution_type', 'unknown'),
                    'confidence': val.get('confidence', 0.0)
                }
                for key, val in zip(solution_keys_filtered, solution_values_filtered)
            ]
            with open(SOLUTION_VALUES_FILE, 'w', encoding='utf-8') as f:
                json.dump(solution_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved solution metadata: {SOLUTION_VALUES_FILE}")
        else:
            logger.warning("No valid solution embeddings found")
    else:
        logger.info("No learned solutions in database")
    
    # Build symbol index (visual embeddings)
    logger.info("\nBuilding symbol vector index...")
    symbol_library = learning_database.get("symbol_library", {})
    logger.info(f"Found {len(symbol_library)} symbols in library")
    
    valid_symbols_data = []
    symbol_ids_list = []
    
    # Process symbols with progress tracking
    max_symbols = 10000  # Limit to prevent memory issues
    symbol_count = 0
    
    for sym_id, data in symbol_library.items():
        if symbol_count >= max_symbols:
            logger.warning(f"Symbol library too large ({len(symbol_library)} entries), limiting to {max_symbols}")
            break
        
        # CRITICAL FIX: SymbolLibrary saves embeddings as "embedding", not "visual_embedding"
        # Check both field names for compatibility
        embedding = data.get("visual_embedding") or data.get("embedding")
        if isinstance(data, dict) and isinstance(embedding, list):
            if all(isinstance(x, (int, float)) for x in embedding):
                valid_symbols_data.append(data)
                symbol_ids_list.append(sym_id)
                symbol_count += 1
                
                # Log progress for large datasets
                if symbol_count % 1000 == 0:
                    logger.info(f"  Processed {symbol_count} symbols...")
    
    if valid_symbols_data:
        logger.info(f"Creating symbol vector array with {len(valid_symbols_data)} entries...")
        # CRITICAL FIX: Use "embedding" or "visual_embedding" (check both)
        vectors_as_lists_symbols = [
            data.get("visual_embedding") or data.get("embedding")
            for data in valid_symbols_data
        ]
        symbol_vector_index = np.array(vectors_as_lists_symbols, dtype=np.float32)
        
        # Save symbol index
        np.save(SYMBOL_INDEX_FILE, symbol_vector_index)
        logger.info(f"✓ Saved symbol index: {SYMBOL_INDEX_FILE} ({symbol_vector_index.shape})")
        
        # Save symbol IDs
        with open(SYMBOL_IDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(symbol_ids_list, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved symbol IDs: {SYMBOL_IDS_FILE}")
        
        # Save symbol data (full data for lookup - as per plan)
        symbol_data_file = INDICES_DIR / "symbol_data.json"
        with open(symbol_data_file, 'w', encoding='utf-8') as f:
            json.dump(valid_symbols_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved symbol data: {symbol_data_file}")
    else:
        logger.warning("No valid symbol embeddings found")
    
    logger.info("\n" + "=" * 70)
    logger.info("VECTOR INDICES BUILT SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Indices directory: {INDICES_DIR}")
    logger.info(f"KnowledgeManager will now load in milliseconds instead of minutes!")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        build_vector_indices()
    except Exception as e:
        logger.error(f"Error building vector indices: {e}", exc_info=True)
        sys.exit(1)

