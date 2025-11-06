# Wichtige Dateien: Programm-Ablauf

**Datum:** 2025-11-06  
**Status:** ✅ Aktuell

## 🎯 Übersicht

Diese Datei listet alle wichtigen Dateien auf, in denen das Programm abläuft, damit der Code extern geprüft werden kann.

## 📂 Entry Points (Haupteinstiegspunkte)

### 1. **GUI-Modus**
- **Datei:** `run_gui.py`
- **Beschreibung:** Startet die GUI-Anwendung
- **Wichtig:** Haupteinstiegspunkt für Benutzer

### 2. **CLI-Modus**
- **Datei:** `run_cli.py`
- **Beschreibung:** Startet die CLI-Anwendung
- **Wichtig:** Haupteinstiegspunkt für Kommandozeile

## 🔧 Core Pipeline (Kern-Pipeline)

### 1. **Pipeline Coordinator** (Haupt-Orchestrator)
- **Datei:** `src/analyzer/core/pipeline_coordinator.py`
- **Beschreibung:** Haupt-Orchestrator für die gesamte Pipeline
- **Wichtig:** 
  - Orchestriert alle Phasen (Phase 0-4)
  - Verwaltet Swarm → Guard Rails → Monolith → Fusion Sequenz
  - **Kritische Methoden:**
    - `process()` - Haupt-Methode
    - `_run_phase_2_sequential_analysis()` - Phase 2 Sequenz
    - `_run_phase_2c_fusion()` - Fusion-Logik
    - `_apply_guard_rails()` - Guard Rails Integration

### 2. **Swarm Analyzer** (Element-Erkennung)
- **Datei:** `src/analyzer/analysis/swarm_analyzer.py`
- **Beschreibung:** Spezialist für Element-Erkennung
- **Wichtig:** 
  - Findet alle Elemente (Symbols, Text Labels)
  - Gibt leere `connections: []` Liste zurück
  - Wird in Phase 2a verwendet

### 3. **Monolith Analyzer** (Verbindungs-Erkennung)
- **Datei:** `src/analyzer/analysis/monolith_analyzer.py`
- **Beschreibung:** Spezialist für Verbindungs-Erkennung
- **Wichtig:**
  - Findet alle Verbindungen zwischen Elementen
  - Nutzt Element-Liste als Input (`element_list_json`)
  - Gibt leere `elements: []` Liste zurück
  - Wird in Phase 2c verwendet
  - **Kritische Methoden:**
    - `analyze()` - Haupt-Methode
    - `_analyze_whole_image()` - Whole-Image-Analyse
    - `_process_quadrants_parallel()` - Quadranten-Analyse
    - `_calculate_optimal_quadrant_strategy()` - Adaptive Strategie

### 4. **Guard Rails** (Inference Rules)
- **Datei:** `src/analyzer/core/pipeline_coordinator.py` (Methode `_apply_guard_rails`)
- **Beschreibung:** Bereinigt und ergänzt Element-Liste
- **Wichtig:**
  - Findet fehlende Elemente (SamplePoint-S, ISA-Supply)
  - Bereinigt Element-Typen
  - Wird in Phase 2b verwendet (nach Swarm, vor Monolith)

## 🤖 AI/LLM Integration

### 1. **LLM Client**
- **Datei:** `src/analyzer/ai/llm_client.py`
- **Beschreibung:** Verwaltet alle LLM-API-Calls
- **Wichtig:**
  - Caching
  - Error Handling
  - Circuit Breaker
  - Retry-Logik

### 2. **Knowledge Manager**
- **Datei:** `src/analyzer/learning/knowledge_manager.py`
- **Beschreibung:** Verwaltet Element-Typen und Lern-Datenbank
- **Wichtig:**
  - Element-Typ-Validierung
  - Synonym-Auflösung
  - Lern-Datenbank-Verwaltung

### 3. **Active Learner**
- **Datei:** `src/analyzer/learning/active_learner.py`
- **Beschreibung:** Verwaltet Active Learning Loop
- **Wichtig:**
  - Integriert neues Wissen
  - Symbol Library Verwaltung

## 🎨 GUI

### 1. **Optimized GUI**
- **Datei:** `src/gui/optimized_gui.py`
- **Beschreibung:** Haupt-GUI-Implementierung
- **Wichtig:**
  - Tkinter-basierte GUI
  - Threading für non-blocking UI
  - Queue-basierte Updates
  - Logging-Integration

## ⚙️ Konfiguration

### 1. **Config Service**
- **Datei:** `src/services/config_service.py`
- **Beschreibung:** Verwaltet Konfiguration aus `config.yaml`
- **Wichtig:**
  - Lädt und validiert Konfiguration
  - Bietet Zugriff auf Model-Strategien
  - Bietet Zugriff auf Prompts

### 2. **Config File**
- **Datei:** `config.yaml`
- **Beschreibung:** Haupt-Konfigurationsdatei
- **Wichtig:**
  - Model-Strategien (z.B. `simple_pid_strategy`)
  - Prompts (z.B. `monolithic_analysis_prompt_template`)
  - Logic Parameters (z.B. `use_swarm_analysis`, `use_monolith_analysis`)

## 🔗 Utilities

### 1. **Connection Reasoning** (Chain-of-Thought)
- **Datei:** `src/utils/connection_reasoning.py`
- **Beschreibung:** Chain-of-Thought Reasoning für Verbindungen
- **Wichtig:**
  - Validiert Verbindungen
  - Findet Splits/Merges
  - Findet fehlende Elemente
  - Markiert dangling connections

### 2. **Graph Utils**
- **Datei:** `src/utils/graph_utils.py`
- **Beschreibung:** Graph-Manipulation und Matching
- **Wichtig:**
  - IoU-Berechnung
  - Duplikat-Fusion (mit Levenshtein-Distanz)
  - Polyline-Matching

### 3. **Line Extractor**
- **Datei:** `src/analyzer/analysis/line_extractor.py`
- **Beschreibung:** CV-basierte Linien-Extraktion
- **Wichtig:**
  - Text-Removal (OCR + CV)
  - Adaptive Thresholds
  - Gap-Bridging
  - Skeletonization

### 4. **Complexity Analyzer**
- **Datei:** `src/utils/complexity_analyzer.py`
- **Beschreibung:** CV-basierte Komplexitäts-Analyse
- **Wichtig:**
  - Multi-Metrik-Ansatz (Edge Density, Object Density, Junctions)
  - Wird in Phase 0 verwendet für Strategie-Auswahl

## 📊 Evaluation

### 1. **KPI Calculator**
- **Datei:** `src/analyzer/evaluation/kpi_calculator.py`
- **Beschreibung:** Berechnet KPIs (F1-Score, Precision, Recall)
- **Wichtig:**
  - Element-KPIs
  - Connection-KPIs
  - Quality Score

### 2. **Visualizer**
- **Datei:** `src/analyzer/visualization/visualizer.py`
- **Beschreibung:** Generiert Visualisierungen
- **Wichtig:**
  - Debug Maps
  - Confidence Maps
  - KPI Dashboards

## 🧪 Test Scripts

### 1. **Smoke Test**
- **Datei:** `scripts/smoke_test_gui.py`
- **Beschreibung:** Smoke Test für GUI und Pipeline

### 2. **Simple PID Test**
- **Datei:** `scripts/test_simple_pid_no_truth.py`
- **Beschreibung:** Test für Simple PID ohne Truth Data

### 3. **Uni Images Test**
- **Datei:** `scripts/test_uni_images.py`
- **Beschreibung:** Test für Uni-Bilder mit Truth Data

### 4. **Full Pipeline Test**
- **Datei:** `scripts/test_full_pipeline_features.py`
- **Beschreibung:** Test für alle Pipeline-Features

### 5. **Parameter Tuning**
- **Datei:** `scripts/run_parameter_tuning.py`
- **Beschreibung:** Optuna-basierte Parameter-Optimierung

## 📋 Zusammenfassung: Kritische Dateien für Code-Review

### **Muss geprüft werden:**
1. `src/analyzer/core/pipeline_coordinator.py` - Haupt-Orchestrator
2. `src/analyzer/analysis/monolith_analyzer.py` - Monolith-Verbindungs-Erkennung
3. `src/analyzer/analysis/swarm_analyzer.py` - Swarm-Element-Erkennung
4. `config.yaml` - Konfiguration

### **Sollte geprüft werden:**
5. `src/analyzer/ai/llm_client.py` - LLM-Integration
6. `src/utils/connection_reasoning.py` - Chain-of-Thought Reasoning
7. `src/utils/graph_utils.py` - Graph-Manipulation
8. `src/analyzer/analysis/line_extractor.py` - CV-Linien-Extraktion

### **Kann geprüft werden:**
9. `src/gui/optimized_gui.py` - GUI
10. `src/services/config_service.py` - Config Service
11. `src/analyzer/learning/knowledge_manager.py` - Knowledge Manager
12. `src/analyzer/evaluation/kpi_calculator.py` - KPI-Berechnung

---

**Status:** ✅ **Alle wichtigen Dateien dokumentiert**

