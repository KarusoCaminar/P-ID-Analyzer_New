"""
Diagnose Hang - Findet heraus wo genau das System hängt
"""

import sys
import signal
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.env_loader import load_env_automatically
load_env_automatically()

import logging
logging.basicConfig(level=logging.WARNING)

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

print("=" * 70)
print("DIAGNOSE HANG")
print("=" * 70)

# Stage 1: Config
print("\n[1/4] ConfigService...")
try:
    from src.services.config_service import ConfigService
    cs = ConfigService()
    print("   OK")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

# Stage 2: LLM Client
print("\n[2/4] LLMClient...")
try:
    import os
    from src.analyzer.ai.llm_client import LLMClient
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        print("   FAILED: GCP_PROJECT_ID nicht gesetzt")
        sys.exit(1)
    config = cs.get_raw_config()
    print("   Creating LLMClient (this may take a moment)...")
    llm = LLMClient(project_id, "us-central1", config)
    print("   OK")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Stage 3: Knowledge Manager - THIS IS LIKELY WHERE IT HANGS
print("\n[3/4] KnowledgeManager...")
print("   This is where it usually hangs...")
try:
    from src.analyzer.learning.knowledge_manager import KnowledgeManager
    element_type_list = cs.get_path('element_type_list')
    learning_db = cs.get_path('learning_db')
    
    print(f"   element_type_list: {element_type_list}")
    print(f"   learning_db: {learning_db}")
    print("   Creating KnowledgeManager...")
    
    start = time.time()
    km = KnowledgeManager(
        element_type_list_path=str(element_type_list),
        learning_db_path=str(learning_db),
        llm_handler=llm,
        config=config
    )
    elapsed = time.time() - start
    print(f"   OK (took {elapsed:.2f}s)")
except TimeoutError:
    print("   TIMEOUT - KnowledgeManager creation took too long!")
    sys.exit(1)
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Stage 4: Pipeline Coordinator
print("\n[4/4] PipelineCoordinator...")
try:
    from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
    print("   Creating PipelineCoordinator...")
    coordinator = PipelineCoordinator(
        llm_client=llm,
        knowledge_manager=km,
        config_service=cs
    )
    print("   OK")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL STAGES COMPLETED - NO HANG DETECTED")
print("=" * 70)

