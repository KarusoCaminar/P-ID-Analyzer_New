# Output Folder Fix: All Files in One Folder

**Datum:** 2025-11-06  
**Status:** ✅ Fix implementiert

## 🎯 Problem

Die Ergebnisse wurden in verschiedenen Ordnern gespeichert:
- **Haupt-Ergebnisse:** `outputs/{base_name}_output_{timestamp}/`
- **Pipeline-Logs:** `outputs/{base_name}_output_{timestamp}/logs/`
- **LLM-Logs:** `outputs/logs/` (global, nicht im Output-Ordner)
- **Debug-Informationen:** `outputs/debug/` (global, nicht im Output-Ordner)

**Problem:** Alle Dateien eines Runs waren nicht in einem einzigen Ordner.

## ✅ Lösung

Alle Dateien werden jetzt in einem einzigen Output-Ordner gespeichert:

### **Neue Ordnerstruktur:**

```
outputs/
└── {base_name}_output_{timestamp}/
    ├── {base_name}_results.json          # Ergebnisse
    ├── {base_name}_kpis.json              # KPIs
    ├── {base_name}_cgm_data.json          # CGM-Daten
    ├── {base_name}_legend_info.json       # Legend-Info
    ├── {base_name}_report.html            # HTML-Report
    ├── {base_name}_debug_map.png          # Debug-Map
    ├── {base_name}_confidence_map.png     # Confidence-Map
    ├── {base_name}_kpi_dashboard.png      # KPI-Dashboard
    ├── {base_name}_score_curve.png        # Score-Kurve
    ├── logs/                              # ALLE Logs
    │   ├── pipeline_{timestamp}.log       # Pipeline-Log
    │   └── llm_calls_{timestamp}.log      # LLM-Log
    └── debug/                              # ALLE Debug-Informationen
        ├── prompt-{request_id}.txt        # LLM-Prompts
        ├── response-{request_id}.txt       # LLM-Responses
        ├── circuit-state.json              # Circuit-Breaker-State
        └── workflow-debug.json             # Workflow-Debug
```

## 🔧 Implementierte Änderungen

### **1. LLM-Debug-Verzeichnis** ✅

**Datei:** `src/analyzer/core/pipeline_coordinator.py` (Zeile 230-235)

**Änderung:**
- **Vorher:** `outputs/debug/` (global)
- **Nachher:** `{output_dir}/debug/` (im Output-Ordner)

**Code:**
```python
# CRITICAL: Set LLM client debug directory to output directory (everything in one folder)
output_path = Path(final_output_dir)
debug_dir = output_path / "debug"
debug_dir.mkdir(parents=True, exist_ok=True)
self.llm_client.debug_dir = debug_dir
logger.info(f"LLM debug directory set to: {debug_dir}")
```

### **2. LLM-Logs** ✅

**Datei:** `src/analyzer/core/pipeline_coordinator.py` (Zeile 240-249)

**Änderung:**
- **Vorher:** `outputs/logs/` (global)
- **Nachher:** `{output_dir}/logs/` (im Output-Ordner)

**Code:**
```python
# Setup enhanced LLM logging if enabled (now that we have output_dir)
llm_logging_enabled = self.active_logic_parameters.get('llm_logging_enabled', True)
if llm_logging_enabled:
    from src.services.logging_service import LoggingService
    # Use output directory for LLM logs (everything in one folder)
    log_dir = output_path / "logs"
    log_level_str = self.active_logic_parameters.get('llm_log_level', 'DEBUG')
    log_level = getattr(logging, log_level_str, logging.DEBUG)
    LoggingService.setup_llm_logging(log_dir=log_dir, log_level=log_level)
    logger.info(f"Enhanced LLM logging enabled: {log_dir}")
```

### **3. Circuit-Breaker-State** ✅

**Datei:** `src/analyzer/core/pipeline_coordinator.py` (Zeile 256-262)

**Änderung:**
- **Vorher:** `outputs/debug/circuit-state.json` (global)
- **Nachher:** `{output_dir}/debug/circuit-state.json` (im Output-Ordner)

**Code:**
```python
# CRITICAL: Save reset state to file in output directory
circuit_state_path = debug_dir / 'circuit-state.json'
try:
    self.llm_client.retry_handler.circuit_breaker.save_state(circuit_state_path)
    logger.info(f"Circuit breaker reset state saved to: {circuit_state_path}")
except Exception as e:
    logger.warning(f"Could not save circuit breaker reset state: {e}")
```

### **4. Logging-Dokumentation** ✅

**Datei:** `src/analyzer/core/pipeline_coordinator.py` (Zeile 643-680)

**Änderung:**
- Docstring erweitert, um zu dokumentieren, dass alle Logs im Output-Ordner gespeichert werden

**Code:**
```python
def _setup_output_directory_logging(self, output_dir: str) -> None:
    """Setup logging to output directory.
    
    Creates a log directory in the output directory and configures
    file logging to save all pipeline logs there.
    
    CRITICAL: All logs (pipeline, LLM, debug) are saved in the output directory
    to keep everything in one folder.
    ...
    """
    ...
    logger.info(f"CRITICAL: All output files (logs, debug, results) are in: {output_path}")
```

## 📊 Vorteile

### **1. Alles in einem Ordner:**
- ✅ Alle Dateien eines Runs sind im selben Ordner
- ✅ Einfache Analyse: Alle Informationen an einem Ort
- ✅ Keine verstreuten Dateien: Keine globalen `outputs/logs/` oder `outputs/debug/` mehr

### **2. Nachvollziehbarkeit:**
- ✅ Jeder Run hat seinen eigenen vollständigen Ordner
- ✅ Alle Kommunikation (LLM-Prompts, Responses) ist im `debug/` Ordner
- ✅ Alle Logs (Pipeline, LLM) sind im `logs/` Ordner

### **3. Einfache Archivierung:**
- ✅ Einfach den gesamten Output-Ordner archivieren
- ✅ Keine Suche nach verstreuten Dateien
- ✅ Vollständige Reproduzierbarkeit

## ✅ Status

**Alle Änderungen implementiert:**
- ✅ LLM-Debug-Verzeichnis wird dynamisch gesetzt
- ✅ LLM-Logs werden im Output-Ordner gespeichert
- ✅ Circuit-Breaker-State wird im Output-Ordner gespeichert
- ✅ Pipeline-Logs bleiben im Output-Ordner
- ✅ Alle Visualisierungen bleiben im Output-Ordner
- ✅ Alle JSON-Dateien bleiben im Output-Ordner

**Bereit für Tests:**
- ✅ Code-Änderungen abgeschlossen
- ✅ Linter-Fehler behoben
- ⏳ Wartet auf neuen Testlauf zur Verifizierung

---

**Status:** ✅ **Fix implementiert - Bereit für Verifizierung**

