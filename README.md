# 🚀 P&ID Analyzer v2.0

Professionelles KI-System für die automatisierte Analyse von P&ID (Piping & Instrumentation Diagram) Diagrammen mit Google Gemini AI.

## 📋 Übersicht

Der P&ID Analyzer ist ein hochmodernes System zur automatischen Erkennung und Extraktion von Komponenten, Verbindungen und Topologie aus P&ID-Diagrammen. Das System verwendet eine modulare Pipeline-Architektur mit mehreren Analyse-Phasen und unterstützt sowohl einfache als auch komplexe Diagramme.

### 🎯 Hauptfunktionen

- ✅ **Automatische Element-Erkennung**: Pumpen, Ventile, Sensoren, Mischer, etc.
- ✅ **Verbindungs-Analyse**: Automatische Erkennung von Pipeline-Verbindungen
- ✅ **Topologie-Validierung**: Graph-basierte Konsistenzprüfung
- ✅ **Legenden-Erkennung**: Automatische Extraktion von Symbol-Mappings
- ✅ **Selbstkorrektur**: Iterative Verbesserung der Analyse-Ergebnisse
- ✅ **CGM-Generierung**: Python dataclass und JSON-Output
- ✅ **Comprehensive KPIs**: Precision, Recall, F1-Score, Quality Score
- ✅ **Active Learning**: Kontinuierliche Verbesserung durch Lernen aus Fehlern

---

## 🏗️ Pipeline-Architektur

Das System verwendet eine modulare Phase-basierte Architektur:

### **Phase 0: Complexity Analysis (CV-basiert)**
- **Zweck**: Schnelle Komplexitätserkennung für Strategie-Auswahl
- **Prozess**: CV-basierte Multi-Metrik-Analyse (Edge Density, Object Density, Junctions)
- **Output**: Strategie-Name (`simple_pid_strategy` oder `optimal_swarm_monolith`)
- **Datei**: `src/analyzer/core/pipeline_coordinator.py` → `_run_phase_0_complexity_analysis()`

### **Phase 1: Pre-Analysis**
- **1.1 Metadata Extraction**: Extrahiert Titel, Projekt, Datum, Version
- **1.2 Legend Extraction**: Erkennt und extrahiert Symbol- und Line-Mappings aus der Legende
- **Output**: Global Knowledge Repository (Metadata, Symbol-Map, Line-Map)
- **Datei**: `src/analyzer/core/pipeline_coordinator.py` → `_run_phase_1_pre_analysis()`

### **Phase 2: Core Analysis**

#### **2a: Swarm Analysis (Element-Erkennung)**
- **Zweck**: Tile-basierte Detail-Analyse für Element-Erkennung
- **Prozess**: Bild wird in Kacheln aufgeteilt, jede Kachel wird einzeln analysiert
- **Output**: Liste aller erkannten Elemente (Symbole, Text-Labels)
- **Datei**: `src/analyzer/analysis/swarm_analyzer.py`
- **Besonderheit**: Ignoriert Verbindungen (nur Element-Erkennung)

#### **2b: Guard Rails (Inference Rules)**
- **Zweck**: Bereinigung und Anreicherung der Element-Liste
- **Prozess**: 
  - SamplePoint-S: `id == 'S'` → `type = 'Sample Point'`
  - ISA-Supply: `'isa' in id/label` → `type = 'Source'`
  - Confidence-Boost für alle Elemente
- **Output**: Bereinigte Element-Liste
- **Datei**: `src/analyzer/core/pipeline_coordinator.py` → Guard Rails Logic

#### **2c: Monolith Analysis (Verbindungs-Erkennung)**
- **Zweck**: Globale Struktur-Analyse für Verbindungs-Erkennung
- **Prozess**: Analysiert das gesamte Bild (oder Quadranten) und erkennt Verbindungen zwischen Elementen
- **Input**: Element-Liste von Swarm (als JSON)
- **Output**: Liste aller erkannten Verbindungen
- **Datei**: `src/analyzer/analysis/monolith_analyzer.py`
- **Besonderheit**: Nutzt Element-Liste als Knowledge Base

#### **2d: Fusion Engine**
- **Zweck**: Kombiniert Swarm- und Monolith-Ergebnisse
- **Prozess**: 
  - Deduplizierung (IoU-basiert)
  - Confidence-Propagation
  - Element-Merging
- **Output**: Kombinierte Element- und Verbindungs-Liste
- **Datei**: `src/analyzer/analysis/fusion_engine.py`

#### **2e: Predictive Completion**
- **Zweck**: Vervollständigt fehlende Verbindungen
- **Prozess**: Geometrische Heuristiken (Distanz, Position)
- **Output**: Vervollständigte Verbindungs-Liste
- **Datei**: `src/analyzer/core/pipeline_coordinator.py` → `_run_phase_2d_predictive_completion()`

#### **2f: Polyline Refinement**
- **Zweck**: Extrahiert präzise Polylinien für Verbindungen
- **Prozess**: 
  - Option 1: LLM-basiert (Standard)
  - Option 2: Skeleton-basiert (präziser, aber langsamer)
- **Output**: Polylinien-Koordinaten für jede Verbindung
- **Datei**: `src/analyzer/core/pipeline_coordinator.py` → `_run_phase_2e_polyline_refinement()`

### **Phase 3: Self-Correction Loop**
- **Zweck**: Iterative Selbstkorrektur der Analyse-Ergebnisse
- **Prozess**:
  1. **Topology Critic**: Validiert Graph-Konsistenz (Disconnected nodes, Invalid degrees, Missing splits/merges)
  2. **Legend Consistency Critic**: Prüft Konsistenz zwischen Legende und erkannten Symbolen
  3. **Multi-Model Critic**: Visuelle Feedback-Validierung
  4. **Re-Analyse**: Problematische Bereiche werden erneut analysiert
  5. **Plateau-Erkennung**: Stoppt bei keinem Fortschritt
- **Output**: Verbesserte Analyse-Ergebnisse
- **Datei**: `src/analyzer/core/pipeline_coordinator.py` → `_run_phase_3_self_correction()`
- **Max. Iterationen**: 15 (konfigurierbar)

### **Phase 4: Post-Processing**
- **4.1 KPI-Berechnung**: Precision, Recall, F1-Score, Quality Score
- **4.2 CGM-Generierung**: Python dataclass und JSON-Output
- **4.3 Visualisierungen**: Confidence Maps, Debug Maps, Score Curves
- **4.4 Active Learning**: Speichert gelernte Patterns
- **Output**: Finale Analyse-Ergebnisse, Visualisierungen, Reports
- **Datei**: `src/analyzer/core/pipeline_coordinator.py` → `_run_phase_4_post_processing()`

---

## 📁 Projektstruktur

```
pid_analyzer_v2/
├── src/                          # Haupt-Code
│   ├── analyzer/                 # Analyse-Komponenten
│   │   ├── core/                 # Pipeline Coordinator
│   │   ├── analysis/             # Swarm, Monolith, Fusion
│   │   ├── learning/             # Knowledge Manager, Active Learner
│   │   ├── evaluation/           # KPI Calculator
│   │   ├── output/               # CGM Generator
│   │   ├── visualization/        # Visualizer
│   │   └── models/               # Pydantic Models
│   ├── services/                 # Services (Config, Logging)
│   ├── utils/                    # Utilities
│   ├── gui/                      # GUI (tkinter)
│   └── interfaces/               # ABC Interfaces
├── scripts/                      # Skripte
│   ├── validation/               # Test-Skripte
│   ├── training/                 # Training-Skripte
│   ├── utilities/                # Utility-Skripte
│   └── utils/                    # Script-Utilities
├── training_data/                # Training-Daten
│   ├── simple_pids/              # Einfache P&IDs
│   ├── complex_pids/             # Komplexe P&IDs
│   ├── viewshot_examples/        # Viewshot-Beispiele
│   └── learning_db.json          # Learning Database
├── outputs/                      # Output-Ordner
│   ├── live_test/                # Live-Test-Outputs
│   ├── overnight_optimization/   # Overnight-Test-Outputs
│   └── ...
├── docs/                         # Dokumentation
├── config.yaml                   # Haupt-Konfigurationsdatei
├── requirements.txt              # Python-Dependencies
├── run_cli.py                    # CLI-Starter
├── run_gui.py                    # GUI-Starter
└── README.md                     # Diese Datei
```

---

## 🚀 Schnellstart

### 1. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 2. Umgebungsvariablen setzen

Erstelle `.env` Datei im Projekt-Root:

```bash
GCP_PROJECT_ID=dein_project_id
GCP_LOCATION=us-central1
```

### 3. Vector-Indizes erstellen (Optional, aber empfohlen)

```bash
python scripts/training/build_vector_indices.py
```

Dies beschleunigt den Startup erheblich.

### 4. Erste Analyse starten

**CLI:**
```bash
python run_cli.py path/to/image.png
```

**GUI:**
```bash
python run_gui.py
```

**Live-Test (mit Log-Monitoring):**
```bash
python scripts/validation/run_live_test.py
```

---

## 📚 Wichtige Skripte

### Test-Skripte (`scripts/validation/`)

- **`run_live_test.py`** ⭐ **HAUPT-TEST-SKRIPT**
  - Führt einen vollständigen Test mit Live-Log-Monitoring durch
  - Verwendet strukturierte Output-Ordner
  - Zeigt Logs live im Terminal an

- **`run_simple_test.py`**
  - Einfacher Test-Runner für schnelle Tests
  - Testet eine einzelne Konfiguration

- **`run_strategy_validation.py`**
  - Strategy-Validation-Tests
  - Führt mehrere Strategien nacheinander aus
  - Berechnet KPIs für jede Strategie

- **`run_overnight_optimization.py`**
  - Overnight A/B Testing
  - Führt automatische A/B-Tests zwischen Strategien durch
  - Generiert umfassende Reports

### Monitoring-Skripte (`scripts/validation/`)

- **`monitor_overnight.py`**: Überwacht Overnight-Prozess
- **`watchdog_overnight.py`**: Watchdog für Overnight
- **`auto_guardian.py`**: Auto-Guardian für Überwachung
- **`continuous_monitor.py`**: Kontinuierlicher Monitor
- **`test_startup_speed.py`**: Startup-Speed-Test
- **`diagnose_hang.py`**: Diagnose-Tool für Hangs

### Training-Skripte (`scripts/training/`)

- **`build_vector_indices.py`**: Erstellt Vektor-Indizes (schneller Startup)
- **`run_pretraining.py`**: Symbol-Pretraining
- **`run_pretraining_stepwise.py`**: Stepwise Pretraining

### Utility-Skripte (`scripts/utilities/`)

- **`backup_learning_db.py`**: Backup der Learning DB
- **`restore_learning_db.py`**: Restore der Learning DB
- **`reset_learning_db.py`**: Reset der Learning DB
- **`cleanup_outputs.py`**: Aufräumen der Output-Ordner
- **`extract_viewshots_from_pretraining_pdf.py`**: Viewshot-Extraktion (PDF)
- **`extract_viewshots_from_uni_bilder.py`**: Viewshot-Extraktion (Uni)

---

## 🔧 Hauptkomponenten

### **PipelineCoordinator** (`src/analyzer/core/pipeline_coordinator.py`)
- **Zweck**: Orchestriert alle Pipeline-Phasen
- **Funktionen**: 
  - Phase 0-4 Ausführung
  - Progress-Callbacks
  - Error-Handling
  - Output-Generierung

### **SwarmAnalyzer** (`src/analyzer/analysis/swarm_analyzer.py`)
- **Zweck**: Tile-basierte Element-Erkennung
- **Funktionen**:
  - Bild-Tiling
  - Parallele Kachel-Analyse
  - Element-Erkennung
  - Viewshot-Integration

### **MonolithAnalyzer** (`src/analyzer/analysis/monolith_analyzer.py`)
- **Zweck**: Globale Verbindungs-Erkennung
- **Funktionen**:
  - Ganzbild-Analyse
  - Quadrant-basierte Analyse
  - Verbindungs-Erkennung
  - Port-Detection

### **FusionEngine** (`src/analyzer/analysis/fusion_engine.py`)
- **Zweck**: Kombiniert Swarm- und Monolith-Ergebnisse
- **Funktionen**:
  - IoU-basierte Deduplizierung
  - Confidence-Propagation
  - Element-Merging

### **KnowledgeManager** (`src/analyzer/learning/knowledge_manager.py`)
- **Zweck**: Verwaltet statisches und dynamisches Wissen
- **Funktionen**:
  - Element-Type-Resolution
  - Similarity-Search
  - Learning-Database-Management
  - Vector-Index-Loading

### **ActiveLearner** (`src/analyzer/learning/active_learner.py`)
- **Zweck**: Kontinuierliches Lernen aus Fehlern
- **Funktionen**:
  - Pattern-Erkennung
  - Correction-Learning
  - Knowledge-Update

### **KPICalculator** (`src/analyzer/evaluation/kpi_calculator.py`)
- **Zweck**: Berechnet Comprehensive KPIs
- **Funktionen**:
  - Precision, Recall, F1-Score
  - Quality Score
  - Connection-Matching
  - Element-Matching

### **CGMGenerator** (`src/analyzer/output/cgm_generator.py`)
- **Zweck**: Generiert CGM-Daten (Python dataclass + JSON)
- **Funktionen**:
  - Network-Instanz-Generierung
  - Connector-Generierung
  - System-Flow-Generierung

---

## ⚙️ Konfiguration

Die Haupt-Konfigurationsdatei ist `config.yaml`. Wichtige Bereiche:

### **Strategien** (`strategies/`)
- `simple_whole_image`: Einfache P&IDs (Monolith-Only)
- `default_flash`: Flash-Strategie (schnell)
- `optimal_swarm_monolith`: Optimale Strategie (Swarm + Monolith)

### **Modelle** (`model_strategy/`)
- `swarm_model`: Modell für Swarm-Analyse
- `monolith_model`: Modell für Monolith-Analyse
- `critic_model`: Modell für Critics

### **Logik-Parameter** (`logic_parameters/`)
- `use_swarm_analysis`: Swarm-Analyse aktivieren
- `use_monolith_analysis`: Monolith-Analyse aktivieren
- `use_fusion`: Fusion aktivieren
- `use_phase_3`: Self-Correction aktivieren
- `max_iterations`: Max. Iterationen für Self-Correction

---

## 📊 Output-Struktur

Alle Outputs folgen einer standardisierten Struktur:

```
outputs/
  {test_type}/                    # z.B. live_test, overnight_optimization
    YYYYMMDD_HHMMSS/              # Timestamp für jeden Testlauf
      logs/                       # Log-Dateien
        test.log
      visualizations/             # Visualisierungen
        {image_name}_score_curve.png
        {image_name}_confidence_map.png
        {image_name}_debug_map.png
      data/                       # Daten (JSON, Python)
        {image_name}_results.json
        {image_name}_cgm_data.json
        {image_name}_cgm_network_generated.py
        {image_name}_kpis.json
      artifacts/                  # Artefakte (Config, Reports)
        config_snapshot.yaml
        {image_name}_report.html
      temp/                       # Temporäre Dateien
        temp_quadrants/
        temp_polylines/
      README.md                   # Erklärt die Struktur
```

---

## 📖 Dokumentation

Alle Dokumentationen finden Sie im **[docs/](docs/)** Ordner:

- **[Pipeline-Dokumentation](docs/PIPELINE_PROCESS_DETAILED.md)**: Detaillierte Prozessbeschreibung
- **[Output-Struktur](docs/OUTPUT_STRUCTURE_STANDARD.md)**: Gold Standard für Output-Ordner
- **[Overnight-Optimization](docs/OVERNIGHT_OPTIMIZATION_GUIDE.md)**: Anleitung für Overnight-Tests
- **[Test-Strategie](tests/STRATEGY_VALIDATION.md)**: Strategy-Validation-Tests

---

## 🎯 Features im Detail

### **Graphentheorie (NetworkX)**
- Graph-basierte Repräsentation der P&ID-Topologie
- Split/Merge-Detection
- Pipeline-Flow-Analyse

### **CGM Format**
- Python dataclass-Format für Code-Generierung
- JSON-Format für Daten-Austausch
- System-Flow-Generierung

### **Active Learning**
- Kontinuierliches Lernen aus Fehlern
- Pattern-Erkennung
- Knowledge-Update

### **Comprehensive KPIs**
- Element-Precision, Recall, F1-Score
- Connection-Precision, Recall, F1-Score
- Quality Score (gewichteter Durchschnitt)
- Hallucination-Detection

---

## 🔍 Troubleshooting

### Problem: Startup hängt
**Lösung**: Führe `python scripts/training/build_vector_indices.py` aus, um Vektor-Indizes zu erstellen.

### Problem: GCP-Credentials fehlen
**Lösung**: Erstelle `.env` Datei mit `GCP_PROJECT_ID` und `GCP_LOCATION`.

### Problem: Output-Ordner unorganisiert
**Lösung**: Das System verwendet automatisch strukturierte Output-Ordner. Prüfe `docs/OUTPUT_STRUCTURE_STANDARD.md`.

### Problem: Tests schlagen fehl
**Lösung**: Prüfe Logs in `outputs/{test_type}/YYYYMMDD_HHMMSS/logs/test.log`.

---

## 📝 License

[Lizenz-Informationen hier einfügen]

---

## 👥 Contributors

[Contributor-Informationen hier einfügen]

---

**Status:** ✅ System ist PRODUCTION-READY und vollständig dokumentiert!

**Letzte Aktualisierung:** 2025-11-07
