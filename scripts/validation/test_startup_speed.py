"""
Quick Startup Test - Prüft wie schnell das System startet
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.env_loader import load_env_automatically
load_env_automatically()

import logging
logging.basicConfig(level=logging.WARNING)  # Nur Warnings und Errors

print("=" * 70)
print("STARTUP SPEED TEST")
print("=" * 70)

stages = {}

# Stage 1: Config Service
print("\n1. Loading ConfigService...")
start = time.time()
try:
    from src.services.config_service import ConfigService
    cs = ConfigService()
    stages['config'] = time.time() - start
    print(f"   [OK] ConfigService: {stages['config']:.2f}s")
except Exception as e:
    print(f"   [FAIL] ConfigService FAILED: {e}")
    sys.exit(1)

# Stage 2: LLM Client
print("\n2. Creating LLMClient...")
start = time.time()
try:
    import os
    from src.analyzer.ai.llm_client import LLMClient
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        print("   ✗ GCP_PROJECT_ID nicht gesetzt")
        sys.exit(1)
    config = cs.get_raw_config()
    llm = LLMClient(project_id, "us-central1", config)
    stages['llm'] = time.time() - start
    print(f"   [OK] LLMClient: {stages['llm']:.2f}s")
except Exception as e:
    print(f"   [FAIL] LLMClient FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Stage 3: Knowledge Manager
print("\n3. Creating KnowledgeManager...")
start = time.time()
try:
    from src.analyzer.learning.knowledge_manager import KnowledgeManager
    element_type_list = cs.get_path('element_type_list')
    learning_db = cs.get_path('learning_db')
    km = KnowledgeManager(
        element_type_list_path=str(element_type_list),
        learning_db_path=str(learning_db),
        llm_handler=llm,
        config=config
    )
    stages['knowledge'] = time.time() - start
    print(f"   [OK] KnowledgeManager: {stages['knowledge']:.2f}s")
except Exception as e:
    print(f"   [FAIL] KnowledgeManager FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Stage 4: Pipeline Coordinator
print("\n4. Creating PipelineCoordinator...")
start = time.time()
try:
    from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
    coordinator = PipelineCoordinator(
        llm_client=llm,
        knowledge_manager=km,
        config_service=cs
    )
    stages['coordinator'] = time.time() - start
    print(f"   [OK] PipelineCoordinator: {stages['coordinator']:.2f}s")
except Exception as e:
    print(f"   [FAIL] PipelineCoordinator FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
total = sum(stages.values())
print("\n" + "=" * 70)
print("STARTUP SUMMARY")
print("=" * 70)
for stage, duration in stages.items():
    print(f"  {stage:15s}: {duration:6.2f}s ({duration/total*100:5.1f}%)")
print(f"  {'TOTAL':15s}: {total:6.2f}s")
print("=" * 70)

if total > 10:
    print("WARNING: Startup takes more than 10 seconds!")
else:
    print("Startup speed OK")

