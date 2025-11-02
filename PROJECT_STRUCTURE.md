# Projekt-Struktur

## Übersicht

```
pid_analyzer_v2/
├── src/                           # Source Code
│   ├── analyzer/                  # Kern-Analyse-Komponenten
│   │   ├── __init__.py
│   │   ├── models/                # Pydantic Datenmodelle
│   │   │   ├── __init__.py
│   │   │   ├── elements.py        # Element, Connection, Port, BBox
│   │   │   └── pipeline.py        # PipelineState, AnalysisResult
│   │   ├── core/                  # Pipeline-Koordinator (in Arbeit)
│   │   ├── analysis/              # Swarm/Monolith Analyzer (in Arbeit)
│   │   ├── extraction/            # Element/Connection Extractor (in Arbeit)
│   │   ├── refinement/            # Semantic Corrector (in Arbeit)
│   │   ├── learning/              # Knowledge Manager & Learning
│   │   │   ├── __init__.py
│   │   │   └── knowledge_manager.py
│   │   ├── output/                # Exporters (in Arbeit)
│   │   └── ai/                    # LLM Client & Embeddings
│   │       ├── __init__.py
│   │       ├── llm_client.py
│   │       ├── prompt_manager.py
│   │       └── embedding_service.py
│   ├── interfaces/                # ABC Interfaces
│   │   ├── __init__.py
│   │   ├── processor.py
│   │   ├── analyzer.py
│   │   └── exporter.py
│   ├── services/                  # Services
│   │   ├── __init__.py
│   │   ├── config_service.py
│   │   ├── cache_service.py
│   │   └── logging_service.py
│   └── utils/                     # Hilfsfunktionen (in Arbeit)
│
├── training_data/                 # Trainings- und Testdaten
│   ├── simple_pids/
│   ├── complex_pids/
│   └── Testbilder (Debugging_Phase)/
│
├── pretraining_symbols/           # Symbol-Vortraining
│
├── outputs/                       # Analyse-Ergebnisse
│
├── temp_tiles/                    # Temporäre Tile-Dateien
├── temp_symbols_for_embeddings/   # Temporäre Symbol-Dateien
├── .pni_analyzer_cache/           # LLM Cache
│
├── config.yaml                    # Haupt-Konfiguration
├── element_type_list.json         # Element-Typen-Definition
├── learning_db.json               # Lern-Datenbank
│
├── requirements.txt               # Python-Abhängigkeiten
├── pyproject.toml                 # Projekt-Konfiguration
├── .gitignore                     # Git-Ignore-Regeln
├── .env.example                   # Environment-Variablen Beispiel
│
├── README.md                      # Haupt-Dokumentation
├── SETUP.md                       # Setup-Anleitung
├── PROJECT_STRUCTURE.md           # Diese Datei
├── MIGRATION_GUIDE.md             # Migrations-Guide
├── IMPLEMENTATION_STATUS.md       # Implementierungs-Status
└── README_REFACTORING.md          # Refactoring-Dokumentation
```

## Datei-Beschreibungen

### Core Module

- **src/analyzer/models/**: Pydantic Models für type-safe Datenstrukturen
- **src/analyzer/learning/**: Knowledge Manager mit Vector-Indexing
- **src/analyzer/ai/**: LLM Client mit Caching und Retry-Logic

### Services

- **src/services/config_service.py**: Type-safe Configuration Management
- **src/services/cache_service.py**: Disk-Cache für LLM-Responses
- **src/services/logging_service.py**: Zentralisiertes Logging

### Interfaces

- **src/interfaces/**: ABC Interfaces für alle Komponenten

### Data

- **config.yaml**: Haupt-Konfiguration (YAML)
- **element_type_list.json**: Definition aller P&ID Element-Typen
- **learning_db.json**: Persistente Lern-Datenbank mit Embeddings

## Status

### ✅ Implementiert

- Modulare Verzeichnisstruktur
- Pydantic Models
- Knowledge Manager (refactored)
- LLM Client (refactored)
- Config Service
- Cache Service
- Logging Service

### 🚧 In Arbeit

- Pipeline Coordinator
- Analysis Components
- Extraction Components
- Utils

### ❌ Ausstehend

- Testing
- GUI/CLI
- Documentation
- Performance Optimization


