# Migration Guide - Alte zu Neue Version

## Übersicht

Dieses Dokument beschreibt die Migration von der alten monolithischen Struktur zur neuen modularen Architektur.

## Was wurde bereits migriert

### 1. Type System (Pydantic Models)
- ✅ `Element`, `Connection`, `Port`, `BBox` als Pydantic Models
- ✅ `PipelineState`, `AnalysisResult` Models
- **Alte Location:** `utils.py` (TypedDict)
- **Neue Location:** `src/analyzer/models/`

### 2. Learning System
- ✅ `KnowledgeManager` refactored
- **Alte Location:** `knowledge_bases.py`
- **Neue Location:** `src/analyzer/learning/knowledge_manager.py`
- **Kompatibilität:** Verwendet weiterhin `learning_db.json` im gleichen Format

### 3. LLM Handler
- ✅ `LLMClient` mit verbesserter Fehlerbehandlung
- **Alte Location:** `llm_handler.py`
- **Neue Location:** `src/analyzer/ai/llm_client.py`
- **Features:** Caching, Retry-Logic, Timeout-Handling

### 4. Configuration
- ✅ Type-safe Config Service
- **Alte Location:** `config.yaml` + `config_validator.py`
- **Neue Location:** `src/services/config_service.py`
- **Kompatibilität:** Liest weiterhin `config.yaml`

## Was noch zu migrieren ist

### 1. Core Processor
- ❌ `core_processor.py` → `src/analyzer/core/pipeline.py`
- **Status:** Grundstruktur erstellt, Logik noch zu migrieren

### 2. Analysis Components
- ❌ Swarm Analyzer → `src/analyzer/analysis/swarm_analyzer.py`
- ❌ Monolith Analyzer → `src/analyzer/analysis/monolith_analyzer.py`
- ❌ Fusion Engine → `src/analyzer/analysis/fusion_engine.py`

### 3. Extraction Components
- ❌ Element Extractor → `src/analyzer/extraction/element_extractor.py`
- ❌ Connection Tracer → `src/analyzer/extraction/connection_tracer.py`
- ❌ Polyline Refiner → `src/analyzer/extraction/polyline_refiner.py`

### 4. Utils
- ❌ Bildverarbeitung → `src/utils/image_utils.py`
- ❌ Graph-Operationen → `src/utils/graph_utils.py`

### 5. Output Components
- ❌ JSON Exporter → `src/analyzer/output/json_exporter.py`
- ❌ CGM Generator → `src/analyzer/output/cgm_generator.py`
- ❌ Visualizer → `src/analyzer/output/visualizer.py`

## Migrations-Strategie

### Option 1: Parallel Run (Empfohlen)
1. Neue Struktur bleibt parallel zur alten
2. Schrittweise Migration einzelner Komponenten
3. Tests nach jeder Migration
4. Alte Code bleibt als Backup

### Option 2: Komplett-Migration
1. Alles auf einmal migrieren
2. Risiko: Breaking Changes
3. Empfohlen nur wenn umfassende Tests vorhanden

## Daten-Migration

### Learning Database
Die `learning_db.json` wird im gleichen Format beibehalten:
- Keine Migration notwendig
- Neue Version ist abwärtskompatibel
- Alte Daten werden automatisch geladen

### Config
Die `config.yaml` bleibt kompatibel:
- Neue Version liest alte Config
- Zusätzliche Validierung durch Pydantic
- Optional: Migration zu neuer Config-Struktur

## API-Änderungen

### Alte API:
```python
from core_processor import Core_Processor
processor = Core_Processor(llm_handler, knowledge_manager, model_strategy, config)
result = processor.run_full_pipeline(image_path)
```

### Neue API:
```python
from src.analyzer.core.pipeline import PipelineCoordinator
from src.services.config_service import ConfigService

config_service = ConfigService("config.yaml")
processor = PipelineCoordinator(config_service)
result = processor.process(image_path)
```

## Breaking Changes

1. **Imports:** Neue Pfade für alle Module
2. **Initialisierung:** Dependency Injection statt direktem Constructor-Call
3. **Return Types:** Pydantic Models statt Dicts

## Schritt-für-Schritt Migration

### Schritt 1: Testen der neuen Komponenten
```bash
# Neue Komponenten testen
python -m pytest tests/test_models.py
python -m pytest tests/test_knowledge_manager.py
```

### Schritt 2: Migration eines kleinen Teils
- Beginne mit einem isolierten Feature
- Teste gründlich
- Dokumentiere Änderungen

### Schritt 3: Graduelle Erweiterung
- Erweitere Migration Schritt für Schritt
- Behalte alte Version als Backup
- Teste nach jedem Schritt

## Rollback-Strategie

Falls Probleme auftreten:
1. Alte Version bleibt verfügbar
2. `learning_db.json` ist kompatibel
3. Config kann zurückgesetzt werden
4. Git-basierte Versionskontrolle

## Hilfe & Support

Bei Fragen zur Migration:
- Siehe `README_REFACTORING.md` für Details
- Code-Kommentare in neuen Modulen
- Typ-Hints für API-Dokumentation

