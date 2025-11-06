# 🚀 P&ID Analyzer v2.0

Professionelles KI-System für P&ID Diagramm-Analyse

## 📋 Schnellstart

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

### 3. Erste Analyse starten

**CLI:**
```bash
python run_cli.py path/to/image.png
```

**GUI:**
```bash
python run_gui.py
```

## 📚 Vollständige Dokumentation

Alle Dokumentationen finden Sie im **[docs/](docs/README.md)** Ordner:

- **[Schnellstart & Anleitung](docs/README.md)** - Vollständige Dokumentation
- **[Wichtige Dateien](docs/IMPORTANT_FILES.md)** - Code-Review Guide
- **[Pipeline-Dokumentation](docs/PIPELINE_PROCESS_DETAILED.md)** - Detaillierte Prozessbeschreibung
- **[Test-Strategie](tests/STRATEGY_VALIDATION.md)** - Strategy Validation Tests

## 🎯 Features

- ✅ Graphentheorie (NetworkX)
- ✅ Split/Merge Detection
- ✅ Pipeline Flow Analysis
- ✅ CGM Format (Python dataclass + JSON)
- ✅ AI Data Format mit Koordinaten
- ✅ Error Handling & API-Call-Minimierung
- ✅ Performance-Optimierungen
- ✅ Active Learning
- ✅ Comprehensive KPIs

## 🔧 Wichtige Dateien

- **run_cli.py**: CLI Start-Script
- **run_gui.py**: GUI Start-Script
- **config.yaml**: Haupt-Config-Datei
- **requirements.txt**: Python Dependencies
- **.env**: Umgebungsvariablen (muss erstellt werden)

## 📖 Weitere Informationen

Für detaillierte Informationen, Anleitungen und Dokumentation siehe **[docs/README.md](docs/README.md)**.

---

**Status:** ✅ System ist STARTBEREIT für erste Tests!
