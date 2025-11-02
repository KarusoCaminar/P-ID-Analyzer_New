# 🤖 Automatisierter Test - P&ID Analyzer v2.0

## 🎯 Automatisierter System-Test

Das System kann jetzt automatisch getestet werden!

## 🚀 Schnellstart

### Mit automatischer Test-Bild-Suche:

```bash
python run_automated_test.py
```

Das Script sucht automatisch nach Test-Bildern in:
- `training_data/simple_pids/`
- `training_data/`

### Mit eigenem Test-Bild:

```bash
python run_automated_test.py path/to/image.png
```

## 📋 Was wird getestet?

### 1. Environment Check ✅
- Prüft `.env` Datei
- Validiert `GCP_PROJECT_ID` und `GCP_LOCATION`

### 2. Backend Initialization ✅
- ConfigService
- LLMClient
- KnowledgeManager
- PipelineCoordinator

### 3. Automatische Analyse ✅
- Startet vollständige Pipeline
- Progress-Tracking
- Status-Updates

### 4. Ergebnis-Validierung ✅
- Elemente erkannt?
- Verbindungen erkannt?
- Quality Score vorhanden?
- CGM Daten generiert?
- KPIs berechnet?

## 📊 Test-Output

```
============================================================
P&ID Analyzer v2.0 - Automated Test
============================================================

=== Checking Environment Variables ===
[OK] GCP_PROJECT_ID: koretex-zugang
[OK] GCP_LOCATION: us-central1

=== Finding Test Image ===
[OK] Found test image: training_data/simple_pids/Einfaches P&I.png

=== Testing Backend Initialization ===
[OK] ConfigService initialized
[OK] LLMClient initialized
[OK] KnowledgeManager initialized
[OK] PipelineCoordinator initialized

=== Running Analysis ===
Image: training_data/simple_pids/Einfaches P&I.png
Progress: Phase 1: Pre-analysis... (10%)
...
[OK] Analysis completed!

=== Validating Results ===
[OK] Elements detected: 42
[OK] Connections detected: 38
[OK] Quality score: 85.50
[OK] CGM data generated:
    Components: 15
    Connectors: 38
    Splits: 3
    Merges: 2
    Flows: 8
[OK] KPIs calculated: 12 metrics

============================================================
Test Summary
============================================================
[OK] Environment: Passed
[OK] Backend Init: Passed
[OK] Analysis: Passed
[OK] Validation: Passed

[SUCCESS] Automated test completed successfully!
```

## ⚠️ Wichtig: .env Datei erstellen

Bevor du den Test startest, erstelle eine `.env` Datei:

```
GCP_PROJECT_ID=koretex-zugang
GCP_PROJECT_NUMBER=748084370989
GCP_LOCATION=us-central1
```

Siehe `SETUP_ENV.md` für Details.

## 🎯 Verwendung

### Vollständiger Test mit automatischer Bild-Suche:

```bash
python run_automated_test.py
```

### Test mit eigenem Bild:

```bash
python run_automated_test.py training_data/simple_pids/Einfaches\ P\&I.png
```

### Mit Truth-Data (für vollständige KPIs):

Platziere eine `*_truth.json` oder `*_truth_cgm.json` Datei neben dem Bild:
```
training_data/simple_pids/
├── Einfaches P&I.png
├── Einfaches P&I_truth.json  # Optional
└── Einfaches P&I_truth_cgm.json  # Optional
```

## ✅ Exit Codes

- **0**: Test erfolgreich
- **1**: Test fehlgeschlagen

## 🔧 Troubleshooting

### Problem: `GCP_PROJECT_ID not set`

**Lösung:** Erstelle `.env` Datei mit:
```
GCP_PROJECT_ID=koretex-zugang
GCP_PROJECT_NUMBER=748084370989
GCP_LOCATION=us-central1
```

### Problem: `No test image found`

**Lösung:** Gib ein Test-Bild als Argument an:
```bash
python run_automated_test.py path/to/image.png
```

### Problem: `Backend initialization failed`

**Lösung:** 
1. Prüfe `.env` Datei
2. Prüfe GCP-Credentials
3. Prüfe `config.yaml`

## 📚 Weitere Dokumentation

- **START_HERE.md**: Start-Anleitung
- **QUICK_START.md**: Schnellstart-Guide
- **SETUP_ENV.md**: .env Datei Einrichtung

## 🎉 Bereit zum Testen!

Starte den automatisierten Test:

```bash
python run_automated_test.py
```

**Viel Erfolg!** 🚀


