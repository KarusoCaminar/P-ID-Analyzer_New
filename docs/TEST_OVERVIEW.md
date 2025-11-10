# 🧪 Test Overview - Alle verfügbaren Tests

**Datum:** 2025-11-07  
**Status:** Übersicht aller verfügbaren Test-Skripte und -Strategien

---

## 📋 Test-Kategorien

### **1. Live Test Scripts (Empfohlen für einzelne Testläufe)**

#### **`run_live_test.py`** ⭐ **HAUPT-TEST-SKRIPT**
- **Zweck:** Führt einen einzelnen Testlauf mit Live-Log-Monitoring durch
- **Features:**
  - Live-Log-Monitoring im Terminal
  - Strukturierte Ausgabe (OutputStructureManager)
  - Unterstützt verschiedene Bilder: `simple`, `complex`, `uni`
  - Strategie-Auswahl: `--strategy` (z.B. `hybrid_fusion`, `simple_whole_image`)
  - KPI-Berechnung mit Ground Truth
  - Speichert alle Ergebnisse in `outputs/live_test/`
- **Verwendung:**
  ```bash
  python scripts/validation/run_live_test.py --image uni --strategy hybrid_fusion
  ```
- **Output:** `outputs/live_test/{timestamp}/`

#### **`run_simple_test.py`**
- **Zweck:** Schneller Testlauf für einfache Validierung
- **Features:**
  - Minimaler Testlauf
  - Keine Live-Logs
  - Schnellere Ausführung
- **Verwendung:**
  ```bash
  python scripts/validation/run_simple_test.py
  ```

---

### **2. Parameter Tuning Tests**

#### **`run_parameter_tuning.py`** ⚙️ **PARAMETER-OPTIMIERUNG**
- **Zweck:** Optimiert `adaptive_threshold_factor`, `adaptive_threshold_min`, `adaptive_threshold_max`
- **Features:**
  - Testet verschiedene Parameter-Kombinationen
  - Berechnet Connection F1-Score für jede Kombination
  - Worker Pool für Parallelisierung
  - Live-Log-Monitoring
  - Speichert Ergebnisse in `outputs/parameter_tuning/`
- **Verwendung:**
  ```bash
  python scripts/validation/run_parameter_tuning.py
  ```
- **Output:** `outputs/parameter_tuning/{timestamp}/`

#### **`monitor_parameter_tuning.py`**
- **Zweck:** Überwacht den Parameter-Tuning-Prozess
- **Features:**
  - Zeigt Fortschritt an
  - Beste Parameter-Kombination
  - Aktueller Status
- **Verwendung:**
  ```bash
  python scripts/validation/monitor_parameter_tuning.py
  ```

#### **`show_parameter_tuning_status.py`**
- **Zweck:** Zeigt Status nach Abschluss des Parameter-Tuning
- **Features:**
  - Statistik
  - Beste Parameter
  - KPIs
  - Warnungen (z.B. Connection F1 = 0.0)

---

### **3. Strategy Validation Tests**

#### **`run_strategy_validation.py`**
- **Zweck:** Testet verschiedene Strategien (z.B. `hybrid_fusion`, `simple_whole_image`)
- **Features:**
  - Vergleicht verschiedene Strategien
  - Berechnet KPIs für jede Strategie
  - Speichert Ergebnisse für Vergleich
- **Verwendung:**
  ```bash
  python scripts/validation/run_strategy_validation.py
  ```

#### **`run_strategy_validation_with_logs.py`**
- **Zweck:** Wie `run_strategy_validation.py`, aber mit detaillierten Logs
- **Features:**
  - Detaillierte Logs
  - Live-Monitoring
  - Bessere Fehlerdiagnose

---

### **4. Overnight Optimization Tests**

#### **`run_overnight_optimization.py`** 🌙 **ÜBERNACHT-OPTIMIERUNG**
- **Zweck:** A/B-Testing und Optimierung über Nacht
- **Features:**
  - Testet verschiedene Strategien
  - Parameter-Optimierung
  - KPI-Berechnung
  - Active Learning
  - Automatische Wiederholung bei Fehlern
- **Verwendung:**
  ```bash
  python scripts/validation/run_overnight_optimization.py
  ```

#### **`monitor_overnight.py`**
- **Zweck:** Überwacht den Overnight-Optimization-Prozess
- **Features:**
  - Fortschrittsanzeige
  - Log-Überwachung
  - Fehler-Erkennung

#### **`watchdog_overnight.py`**
- **Zweck:** Watchdog für Overnight-Optimization
- **Features:**
  - Startet Prozess neu bei Absturz
  - Überwacht Logs
  - Automatische Fehlerbehandlung

---

### **5. API Rate Limit Tests**

#### **`test_api_rate_limit.py`** 🚀 **RATE-LIMIT-TEST**
- **Zweck:** Testet API Rate Limits für verschiedene Modelle/Regionen
- **Features:**
  - Testet Flash und Pro Modelle
  - Verschiedene Worker-Anzahlen (15, 20, 25, 30, 40, 50)
  - Verschiedene Regionen (us-central1, europe-west3)
  - Berechnet maximale RPM
  - DSQ-Optimierung
- **Verwendung:**
  ```bash
  python scripts/validation/test_api_rate_limit.py
  ```
- **Output:** `outputs/rate_limit_test/rate_limit_test_results_{timestamp}.json`

#### **`analyze_rate_limit_results.py`**
- **Zweck:** Analysiert Rate-Limit-Test-Ergebnisse
- **Features:**
  - Empfehlungen für Config
  - Beste Worker-Anzahl
  - Beste Region
  - Maximale RPM

---

### **6. Performance Tests**

#### **`test_startup_speed.py`**
- **Zweck:** Misst Startup-Zeit verschiedener Komponenten
- **Features:**
  - KnowledgeManager Startup-Zeit
  - LLMClient Startup-Zeit
  - PipelineCoordinator Startup-Zeit
  - Gesamt-Startup-Zeit

---

### **7. Unit Tests**

#### **`tests/unit/test_*.py`**
- **Zweck:** Unit Tests für einzelne Komponenten
- **Verfügbare Tests:**
  - `test_pipeline_coordinator.py` - Pipeline Coordinator Tests
  - `test_swarm_analyzer.py` - Swarm Analyzer Tests
  - `test_monolith_analyzer.py` - Monolith Analyzer Tests
  - `test_fusion_engine.py` - Fusion Engine Tests
  - `test_line_extractor.py` - Line Extractor Tests
  - `test_kpi_calculator.py` - KPI Calculator Tests
  - `test_cgm_generator.py` - CGM Generator Tests
  - `test_complexity_analyzer.py` - Complexity Analyzer Tests
- **Verwendung:**
  ```bash
  pytest tests/unit/
  ```

---

### **8. Test Harness Utilities**

#### **`src/utils/test_harness.py`** 🔧 **TEST-HARNESS-UTILITIES**
- **Zweck:** Utilities für Test-Harness (Zwischenergebnisse, Config-Snapshots)
- **Features:**
  - `save_intermediate_result()` - Speichert Zwischenergebnisse nach jeder Phase
  - `save_config_snapshot()` - Speichert Config-Snapshot
  - `save_test_metadata()` - Speichert Test-Metadaten
  - Verwendet OutputStructureManager für strukturierte Ausgabe
- **Verwendung:** Wird automatisch von PipelineCoordinator verwendet

---

## 🎯 Empfohlene Test-Reihenfolge

### **1. Einfacher Testlauf (Schnell)**
```bash
python scripts/validation/run_live_test.py --image simple --strategy simple_whole_image
```
- **Zweck:** Schneller Test, um zu prüfen, ob alles funktioniert
- **Dauer:** ~3-5 Minuten

### **2. Komplexer Testlauf (Vollständig)**
```bash
python scripts/validation/run_live_test.py --image uni --strategy hybrid_fusion
```
- **Zweck:** Vollständiger Testlauf mit komplexem Bild
- **Dauer:** ~10-20 Minuten
- **Features:**
  - Live-Log-Monitoring
  - KPI-Berechnung
  - Strukturierte Ausgabe

### **3. Parameter Tuning (Optimierung)**
```bash
python scripts/validation/run_parameter_tuning.py
```
- **Zweck:** Optimiert `adaptive_threshold_factor`, `adaptive_threshold_min`, `adaptive_threshold_max`
- **Dauer:** ~1-2 Stunden (36 Kombinationen)
- **Monitoring:**
  ```bash
  python scripts/validation/monitor_parameter_tuning.py
  ```

### **4. Strategy Validation (Vergleich)**
```bash
python scripts/validation/run_strategy_validation.py
```
- **Zweck:** Vergleicht verschiedene Strategien
- **Dauer:** ~30-60 Minuten
- **Output:** Vergleich verschiedener Strategien

### **5. Overnight Optimization (Langzeit)**
```bash
python scripts/validation/run_overnight_optimization.py
```
- **Zweck:** Langzeit-Optimierung über Nacht
- **Dauer:** ~8-12 Stunden
- **Monitoring:**
  ```bash
  python scripts/validation/monitor_overnight.py
  ```

---

## 📊 Test-Output-Struktur

### **Live Test Output:**
```
outputs/live_test/{timestamp}/
├── data/
│   ├── test_result.json
│   └── output_phase_*.json
├── artifacts/
│   ├── config_snapshot.yaml
│   └── test_metadata.json
├── visualizations/
│   ├── debug_map.png
│   ├── score_curve.png
│   └── kpi_dashboard.png
├── logs/
│   └── test.log
└── README.md
```

### **Parameter Tuning Output:**
```
outputs/parameter_tuning/{timestamp}/
├── data/
│   ├── parameter_tuning_results.json
│   └── parameter_tuning_summary.json
├── logs/
│   └── parameter_tuning.log
└── README.md
```

---

## 🔍 Monitoring & Debugging

### **Live Log Monitoring:**
```bash
# Python Monitor
python scripts/validation/monitor_live_test.py

# PowerShell Monitor
scripts/validation/watch_live_test.ps1
```

### **Log Analysis:**
```bash
# Zeige letzten 100 Zeilen
tail -n 100 outputs/live_test/{timestamp}/logs/test.log

# Suche nach Fehlern
grep -i error outputs/live_test/{timestamp}/logs/test.log
```

---

## ⚙️ Test-Konfiguration

### **Verfügbare Strategien:**
- `simple_whole_image` - Einfache P&IDs (Monolith-only)
- `hybrid_fusion` - Komplexe P&IDs (Swarm + Monolith + Fusion)
- `optimal_swarm_monolith` - Balanced (Standard)
- `quality_focused` - Maximale Qualität
- `default_flash` - Schnellste Strategie (nur Swarm)

### **Verfügbare Bilder:**
- `simple` - Einfaches P&ID (`Einfaches P&I.png`)
- `complex` - Komplexes P&ID (`page_1_original.png`)
- `uni` - Uni-Bild (`Verfahrensfließbild_Uni.png`)

---

## 🎯 Nächste Schritte

1. **Starte einfachen Testlauf:**
   ```bash
   python scripts/validation/run_live_test.py --image simple --strategy simple_whole_image
   ```

2. **Starte komplexen Testlauf:**
   ```bash
   python scripts/validation/run_live_test.py --image uni --strategy hybrid_fusion
   ```

3. **Überwache Live-Logs:**
   ```bash
   python scripts/validation/monitor_live_test.py
   ```

4. **Analysiere Ergebnisse:**
   - Check `outputs/live_test/{timestamp}/data/test_result.json`
   - Check `outputs/live_test/{timestamp}/visualizations/kpi_dashboard.png`

---

## 📝 Wichtige Hinweise

### **Test Harness:**
- **Automatisch aktiviert:** Test Harness wird automatisch von `PipelineCoordinator` verwendet
- **Zwischenergebnisse:** Werden in `data/output_phase_*.json` gespeichert
- **Config-Snapshots:** Werden in `artifacts/config_snapshot.yaml` gespeichert

### **Optimierungen:**
- **Worker:** 30 (Flash-optimiert)
- **RPM:** 500 (Flash-optimiert)
- **Region:** us-central1 (2.5x schneller)

### **Erwartete Performance:**
- **Flash:** 500-530 RPM (stabil)
- **Pro:** 24 RPM (stabil bei 15 Workers)
- **Pipeline:** ~40-50% Zeitersparnis mit optimierten Einstellungen

