# 🧪 Test-Strategie: Vollständige Dokumentation

**Datum:** 2025-11-06  
**Status:** ✅ Konfiguriert und bereit

---

## 📋 Übersicht

Diese Dokumentation beschreibt die vollständige Test-Strategie für die P&ID-Analyse-Pipeline. Alle Tests werden automatisch validiert, ausgeführt und die Ergebnisse werden in einer sauberen Ordnerstruktur unter `outputs/strategy_validation/` gespeichert.

---

## 🎯 Test-Ziele

Die Test-Strategie dient dazu:

1. **Pipeline-Komponenten isoliert zu testen** - Jede Phase einzeln validieren
2. **Performance zu messen** - F1-Scores, Precision, Recall für Elemente und Verbindungen
3. **Fehlerquellen zu identifizieren** - Welche Komponente verursacht welche Fehler?
4. **Strategien zu vergleichen** - Monolith vs. Swarm vs. Fusion
5. **Verbesserungen zu validieren** - Predictive, Polyline, Self-Correction

---

## 📊 Test-Übersicht

### Test 1: Baseline Phase 1 (Legenden-Erkennung)
- **Ziel:** Nur Phase 1 (Pre-Analysis) testen
- **Bild:** `training_data/complex_pids/page_1_original.png` (MIT Legende)
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Aktiviert:** Nur Phase 1
- **Deaktiviert:** Alle anderen Phasen
- **Erwartung:** Legende wird erkannt, keine Elemente/Verbindungen

### Test 2: Baseline Simple P&ID (Monolith-All)
- **Ziel:** Monolith-Analyse auf einfachem Bild testen
- **Bild:** `training_data/simple_pids/Einfaches P&I.png` (Simple P&ID)
- **Ground Truth:** `training_data/simple_pids/Einfaches P&I_truth.json`
- **Aktiviert:** Phase 2 (Monolith), Phase 4 (Post-Processing)
- **Deaktiviert:** Swarm, Fusion, Predictive, Polyline, Self-Correction
- **Erwartung:** Gute F1-Scores für einfaches Bild

### Test 3: Baseline Swarm-Only
- **Ziel:** Swarm-Analyse isoliert testen
- **Bild:** `training_data/simple_pids/Einfaches P&I.png` (Simple P&ID)
- **Ground Truth:** `training_data/simple_pids/Einfaches P&I_truth.json`
- **Aktiviert:** Phase 2 (Swarm), Phase 4 (Post-Processing)
- **Deaktiviert:** Monolith, Fusion, Predictive, Polyline, Self-Correction
- **Erwartung:** Swarm findet Elemente, aber keine Verbindungen

### Test 4: Baseline Complex P&ID (Spezialisten-Kette)
- **Ziel:** Vollständige Pipeline auf komplexem Bild testen
- **Bild:** `training_data/complex_pids/page_1_original.png` (Komplexes Bild)
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Aktiviert:** Phase 2 (Swarm + Monolith), Phase 2c (Fusion), Phase 4 (Post-Processing)
- **Deaktiviert:** Predictive, Polyline, Self-Correction
- **Erwartung:** Beste F1-Scores durch Kombination von Swarm + Monolith + Fusion

### Test 5a: Test 4 + Predictive (2d)
- **Ziel:** Predictive Completion testen
- **Bild:** `training_data/complex_pids/page_1_original.png` (Komplexes Bild)
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Aktiviert:** Wie Test 4 + Phase 2d (Predictive)
- **Erwartung:** Verbesserte Recall durch Predictive Completion

### Test 5b: Test 4 + Polyline (2e)
- **Ziel:** Polyline Refinement testen
- **Bild:** `training_data/complex_pids/page_1_original.png` (Komplexes Bild)
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Aktiviert:** Wie Test 4 + Phase 2e (Polyline)
- **Erwartung:** Verbesserte Precision durch Polyline Refinement

### Test 5c: Test 4 + Self-Correction (3)
- **Ziel:** Self-Correction Loop testen
- **Bild:** `training_data/complex_pids/page_1_original.png` (Komplexes Bild)
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Aktiviert:** Wie Test 4 + Phase 3 (Self-Correction)
- **Erwartung:** Verbesserte F1-Scores durch iterative Korrektur

---

## 📁 Output-Ordnerstruktur

Alle Tests speichern ihre Ergebnisse in einer sauberen Ordnerstruktur:

```
outputs/strategy_validation/
├── validation_YYYYMMDD_HHMMSS.json          # Validierungsergebnisse
├── summary_YYYYMMDD_HHMMSS.json            # Finale Zusammenfassung aller Tests
│
├── Test_1_Baseline_Phase_1_(Legenden-Erkennung)/
│   ├── pipeline.log                        # Pipeline-Logs (alle Phasen)
│   ├── logs/                               # LLM-Logs
│   │   └── llm_calls_YYYYMMDD_HHMMSS.log
│   ├── results.json                        # Vollständige Analyse-Ergebnisse
│   ├── kpis.json                           # KPIs (falls Ground Truth verfügbar)
│   └── [weitere Pipeline-Outputs]          # Debug-Maps, Visualisierungen, etc.
│
├── Test_2_Baseline_Simple_PID_(Monolith-All)/
│   ├── pipeline.log
│   ├── logs/
│   ├── results.json
│   ├── kpis.json
│   └── [weitere Pipeline-Outputs]
│
├── Test_3_Baseline_Swarm-Only/
│   └── ...
│
├── Test_4_Baseline_Complex_PID_(Spezialisten-Kette)/
│   └── ...
│
├── Test_5a_Test_4_+_Predictive_(2d)/
│   └── ...
│
├── Test_5b_Test_4_+_Polyline_(2e)/
│   └── ...
│
└── Test_5c_Test_4_+_Self-Correction_(3)/
    └── ...
```

### Dateien pro Test-Ordner

1. **`pipeline.log`** - Alle Pipeline-Logs (Phase 0-4)
2. **`logs/llm_calls_*.log`** - Alle LLM-Aufrufe (Requests/Responses)
3. **`results.json`** - Vollständige Analyse-Ergebnisse (Elements, Connections)
4. **`kpis.json`** - KPIs (F1, Precision, Recall) wenn Ground Truth verfügbar
5. **Weitere Pipeline-Outputs:**
   - `*_debug_map.png` - Debug-Visualisierungen
   - `*_confidence_map.png` - Confidence-Maps
   - `*_score_curve.png` - Score-Kurven
   - `*_kpi_dashboard.png` - KPI-Dashboards
   - `*_report.html` - HTML-Reports
   - `output_phase_*.json` - Zwischenergebnisse pro Phase

---

## 🚀 Test-Ausführung

### Voraussetzungen

1. **GCP-Credentials setzen:**
   ```powershell
   $env:GCP_PROJECT_ID='dein_project_id'
   $env:GCP_LOCATION='us-central1'
   ```

2. **Test-Konfiguration validieren:**
   - Das Skript validiert automatisch alle Bilder und Ground Truth-Dateien
   - Fehler werden vor Test-Start gemeldet

### Einzelnen Test ausführen

```bash
# Test 2 (empfohlen zum Start)
python scripts/validation/run_strategy_validation.py --test "Test 2"

# Test 4 (vollständige Pipeline)
python scripts/validation/run_strategy_validation.py --test "Test 4"

# Test 5c (mit Self-Correction)
python scripts/validation/run_strategy_validation.py --test "Test 5c"
```

### Alle Tests ausführen

```bash
python scripts/validation/run_strategy_validation.py --test all
```

**Laufzeit:** ~20-30 Minuten für alle 7 Tests

---

## 📊 Validierung

### Automatische Validierung

Das Skript validiert automatisch:

1. **Bilder:** Existieren alle Test-Bilder?
2. **Ground Truth:** Existieren alle Ground Truth-Dateien?
3. **JSON-Struktur:** Sind Ground Truth-Dateien gültig?
4. **Elemente/Verbindungen:** Wie viele Elemente/Verbindungen in Ground Truth?

**Validierungsergebnisse werden gespeichert:**
- `outputs/strategy_validation/validation_YYYYMMDD_HHMMSS.json`

### Manuelle Validierung

Nach Test-Ausführung:

1. **Ergebnisse prüfen:** `outputs/strategy_validation/Test_X_*/results.json`
2. **KPIs prüfen:** `outputs/strategy_validation/Test_X_*/kpis.json`
3. **Logs prüfen:** `outputs/strategy_validation/Test_X_*/pipeline.log`
4. **Zusammenfassung prüfen:** `outputs/strategy_validation/summary_*.json`

---

## 📈 Datenanalyse

### Nach Test-Ausführung

Alle Daten sind in `outputs/strategy_validation/` gespeichert:

1. **Zusammenfassung:** `summary_YYYYMMDD_HHMMSS.json`
   - Enthält alle KPIs aller Tests
   - Ermöglicht Vergleich zwischen Tests

2. **Einzelne Test-Ergebnisse:** `Test_X_*/kpis.json`
   - Detaillierte KPIs pro Test
   - Vollständige KPI-Struktur

3. **Logs:** `Test_X_*/pipeline.log` und `Test_X_*/logs/`
   - Vollständige Pipeline-Logs
   - LLM-Aufrufe für Debugging

### Datenanalyse-Skripte

```python
# Beispiel: Zusammenfassung laden
import json
from pathlib import Path

summary_file = Path("outputs/strategy_validation/summary_*.json")
with open(summary_file, 'r') as f:
    summary = json.load(f)

# KPIs vergleichen
for test_name, kpis in summary['results'].items():
    print(f"{test_name}:")
    print(f"  Element F1: {kpis.get('element_f1', 0.0):.4f}")
    print(f"  Connection F1: {kpis.get('connection_f1', 0.0):.4f}")
```

---

## 🔧 Pipeline-Abstimmung

### Nach Test-Ausführung

1. **Ergebnisse analysieren:**
   - Welche Tests haben die besten F1-Scores?
   - Welche Komponenten verbessern die Performance?
   - Welche Komponenten verschlechtern die Performance?

2. **Parameter anpassen:**
   - IoU-Thresholds
   - Confidence-Thresholds
   - Self-Correction-Parameter

3. **Erneut testen:**
   - Tests mit angepassten Parametern ausführen
   - Ergebnisse vergleichen

4. **Iterativ verbessern:**
   - Test → Analyse → Anpassung → Test
   - Bis optimale Performance erreicht ist

---

## ✅ Checkliste

### Vor Test-Ausführung

- [ ] GCP-Credentials gesetzt
- [ ] Test-Bilder vorhanden
- [ ] Ground Truth-Dateien vorhanden
- [ ] Output-Verzeichnis erstellt (`outputs/strategy_validation/`)

### Nach Test-Ausführung

- [ ] Alle Tests erfolgreich abgeschlossen
- [ ] Ergebnisse in `outputs/strategy_validation/` gespeichert
- [ ] Logs verfügbar
- [ ] KPIs berechnet (wenn Ground Truth verfügbar)
- [ ] Zusammenfassung erstellt

---

## 🎯 Nächste Schritte

1. **Tests ausführen:** Starte mit Test 2 (einfachster Test)
2. **Ergebnisse analysieren:** Prüfe KPIs und Logs
3. **Pipeline abstimmen:** Passe Parameter basierend auf Ergebnissen an
4. **Erneut testen:** Validiere Verbesserungen
5. **Finale Pipeline:** Optimale Konfiguration festlegen

---

**Status:** ✅ **Bereit für Test-Ausführung**

