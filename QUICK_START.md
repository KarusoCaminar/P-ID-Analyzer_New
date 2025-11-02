# 🚀 Quick Start Guide - P&ID Analyzer v2.0

## ✅ System ist Bereit für erste Tests!

Das System ist vollständig integriert und einsatzbereit.

## 📋 Voraussetzungen

### 1. Umgebungsvariablen

Erstelle eine `.env` Datei im Projekt-Root:

```bash
GCP_PROJECT_ID=dein_project_id
GCP_LOCATION=us-central1
```

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 3. Config-Datei prüfen

Stelle sicher, dass `config.yaml` existiert und korrekt konfiguriert ist.

## 🎯 Schnellstart

### CLI (Kommandozeile)

```bash
# Einzelnes Bild analysieren
python run_cli.py path/to/image.png

# Mit Output-Verzeichnis
python run_cli.py path/to/image.png --output-dir outputs/my_results

# Mit verbose Logging
python run_cli.py path/to/image.png --verbose
```

**Oder direkt mit src.analyzer.cli:**

```bash
python -m src.analyzer.cli path/to/image.png
```

### GUI (Graphische Oberfläche)

```bash
python run_gui.py
```

**Oder direkt:**

```bash
python -m src.gui.optimized_gui
```

## 📊 Erwartete Ausgaben

### CLI Output

```
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.cli] Starting analysis of: image.png
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.core.pipeline_coordinator] Initialized pipeline for: image.png
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.cli] Progress: Phase 1: Pre-analysis... (10%)
...
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.cli] ============================================================
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.cli] Analysis Complete!
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.cli] ============================================================
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.cli] Image: image.png
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.cli] Elements detected: 42
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.cli] Connections detected: 38
[2024-XX-XX XX:XX:XX - INFO - src.analyzer.cli] Quality score: 85.50
```

### Output-Verzeichnis

Nach erfolgreicher Analyse findest du:

```
outputs/
  image_results/
    ├── image_results.json          # Vollständige Analyse-Ergebnisse
    ├── image_kpis.json              # KPIs
    ├── image_cgm_data.json          # CGM JSON Format
    ├── image_cgm_network_generated.py  # CGM Python Code (dataclass)
    ├── image_debug_map.png          # Debug-Visualisierung
    ├── image_confidence_map.png     # Confidence-Map
    ├── image_uncertainty_heatmap.png  # Uncertainty Heatmap
    └── ...
```

## 🔧 Erste Tests

### Test 1: Einfaches P&ID Diagramm

```bash
# Test mit einfachem Diagramm aus training_data
python run_cli.py training_data/simple_pids/Einfaches\ P\&I.png
```

### Test 2: Mit Truth-Data (für KPI-Berechnung)

Platziere eine `*_truth.json` oder `*_truth_cgm.json` Datei neben dem Bild:

```
training_data/simple_pids/
  ├── Einfaches P&I.png
  ├── Einfaches P&I_truth.json  # Optional
  └── Einfaches P&I_truth_cgm.json  # Optional
```

### Test 3: GUI verwenden

```bash
python run_gui.py
```

1. Klicke auf "Bild auswählen"
2. Wähle ein P&ID Bild
3. Klicke auf "Analyse starten"
4. Sieh dir die Ergebnisse im GUI an

## ✅ System-Check

Um zu prüfen ob alles funktioniert:

```bash
python -c "
from src.analyzer.core.pipeline_coordinator import PipelineCoordinator
from src.analyzer.ai.llm_client import LLMClient
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.services.config_service import ConfigService
print('[OK] Alle Module importiert')
print('[OK] System bereit für Tests!')
"
```

## 🐛 Troubleshooting

### Problem: `GCP_PROJECT_ID not set`

**Lösung:** Erstelle `.env` Datei mit:
```
GCP_PROJECT_ID=dein_project_id
```

### Problem: `Configuration file not found`

**Lösung:** Stelle sicher, dass `config.yaml` im Projekt-Root existiert.

### Problem: `No module named 'src.analyzer...'`

**Lösung:** Starte vom Projekt-Root aus oder füge das Projekt zum PYTHONPATH hinzu:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Problem: Import-Fehler mit vertexai

**Lösung:** Installiere Dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Weitere Dokumentation

- **GRAPH_THEORY_IMPLEMENTATION.md**: Graphentheorie & Mathematik
- **ERROR_HANDLING_OPTIMIZATION.md**: Error Handling & API-Call-Minimierung
- **PRODUCTION_READY.md**: Production-Ready Features
- **PERFORMANCE_OPTIMIZATION.md**: Performance-Optimierungen

## 🎉 Bereit zum Starten!

Das System ist vollständig integriert und einsatzbereit für erste Tests.

**Viel Erfolg mit den ersten Analysen!** 🚀
