# 🧪 Test-Strategie: Erklärung & Auswertung

**Datum:** 2025-11-06  
**Status:** ✅ Vollständig dokumentiert

---

## 📋 Übersicht

Die **Strategy Validation Tests** messen die Performance jeder Pipeline-Komponente isoliert, bevor sie kombiniert werden. Jeder Testlauf wird gegen Ground Truth-Daten validiert, um `element_f1` und `connection_f1` zu erhalten.

---

## 🗂️ Wie werden Ergebnisse gespeichert?

### 1. Test-Ausführung

Jeder Test erstellt ein eigenes Ausgabe-Verzeichnis:

```
outputs/strategy_validation/
├── Test_1_Baseline_Phase_1/
│   ├── results.json              # Vollständige Analyse-Ergebnisse
│   ├── pipeline.log              # Pipeline-Log
│   ├── llm_calls.log             # LLM-Aufrufe
│   ├── kpis.json                 # KPIs (falls Ground Truth verfügbar)
│   └── ...
├── Test_2_Baseline_Simple_PID/
│   └── ...
├── Test_4_Baseline_Complex_PID/
│   └── ...
└── summary_20250101_120000.json  # Finale Zusammenfassung
```

### 2. Zusammenfassung

Nach allen Tests wird eine **Zusammenfassung** erstellt:

**Datei:** `outputs/strategy_validation/summary_YYYYMMDD_HHMMSS.json`

**Struktur:**
```json
{
  "timestamp": "2025-01-01T12:00:00",
  "image_path": "data/input/Einfaches P&I.png",
  "ground_truth": "data/ground_truth/Einfaches P&I.json",
  "results": {
    "Test 2: Baseline Simple P&ID (Monolith-All)": {
      "element_f1": 0.8500,
      "connection_f1": 0.8200,
      "element_precision": 0.8800,
      "element_recall": 0.8200,
      "connection_precision": 0.8500,
      "connection_recall": 0.7900
    },
    "Test 4: Baseline Complex P&ID (Spezialisten-Kette)": {
      "element_f1": 0.7800,
      "connection_f1": 0.6500,
      ...
    }
  }
}
```

---

## 📊 Wie können Ergebnisse ausgewertet werden?

### 1. Manuelle Auswertung

#### Vergleichstabelle erstellen

Öffnen Sie die `summary_*.json` Datei und erstellen Sie eine Vergleichstabelle:

| Test | Element F1 | Connection F1 | Element Precision | Connection Precision | Bemerkungen |
|------|------------|---------------|-------------------|----------------------|-------------|
| Test 2 (Monolith-All) | 0.85 | 0.82 | 0.88 | 0.85 | Sauber, hohe Scores |
| Test 4 (Spezialisten-Kette) | 0.78 | 0.65 | 0.80 | 0.70 | Sauber, aber unvollständig |
| Test 5a (+ Predictive) | 0.75 | 0.60 | 0.78 | 0.65 | **Verschlechterung!** |
| Test 5b (+ Polyline) | 0.79 | 0.68 | 0.81 | 0.72 | **Verbesserung!** |
| Test 5c (+ Self-Correction) | 0.78 | 0.66 | 0.80 | 0.71 | Phase 3 läuft nicht |

#### Entscheidungen treffen

- **Test 5a verschlechtert F1:** Predictive Completion deaktivieren oder Parameter anpassen
- **Test 5b verbessert F1:** Polyline Refinement behalten
- **Test 5c läuft nicht:** `self_correction_min_quality_score` anpassen

### 2. Automatische Auswertung

#### Skript: `scripts/utilities/analyze_test_results.py`

```bash
python scripts/utilities/analyze_test_results.py outputs/strategy_validation/summary_*.json
```

**Ausgabe:**
- Vergleichstabelle
- Beste Konfiguration
- Empfehlungen für Parameter-Anpassungen

---

## 🔧 Wie können Parameter angepasst werden?

### 1. Direkt in `config.yaml`

Öffnen Sie `config.yaml` und passen Sie die Parameter an:

```yaml
logic_parameters:
  # Predictive Completion
  use_predictive_completion: false  # Deaktiviert nach Test 5a
  
  # Polyline Refinement
  use_polyline_refinement: true     # Aktiviert nach Test 5b
  
  # Self-Correction
  use_self_correction_loop: true
  self_correction_min_quality_score: 90.0  # Erhöht nach Test 5c
```

### 2. Über `params_override` im Test-Skript

Im Test-Skript können Parameter zur Laufzeit überschrieben werden:

```python
params_override = {
    "use_predictive_completion": False,
    "use_polyline_refinement": True,
    "self_correction_min_quality_score": 90.0
}
```

### 3. Automatisch über Training Camp

Das **Training Camp** kann Parameter automatisch optimieren (siehe unten).

---

## 🎓 Zusammenhang mit Training Camp

### Was ist das Training Camp?

Das **Training Camp** (`src/analyzer/training/training_camp.py`) ist ein automatisiertes System zur Parameter-Optimierung:

1. **Strategien testen:** Testet verschiedene Strategien (Simple P&ID, Complex P&ID, etc.)
2. **Parameter optimieren:** Testet verschiedene Parameter-Kombinationen
3. **Beste Parameter speichern:** Speichert die besten Parameter automatisch

### Wie funktioniert es?

#### 1. Konfiguration in `config.yaml`

```yaml
training_camp:
  duration_hours: 3
  max_cycles: 0  # 0 = unbegrenzt (nur Zeitlimit)
  
  # Parameter-Kombinationen für Hyperparameter-Optimierung
  parameter_combinations:
    - {}  # Basis-Parameter
    - min_quality_to_keep_bbox: 0.6
    - visual_symbol_similarity_threshold: 0.80
    - min_quality_to_keep_bbox: 0.4
      visual_symbol_similarity_threshold: 0.90
    # ... weitere Kombinationen
```

#### 2. Ausführung

```bash
# Über GUI
python run_gui.py
# → "Training Camp" Button

# Über Skript
python scripts/training/run_automated_testcamp.py
```

#### 3. Ergebnisse

Das Training Camp erstellt:
- **`training_report.csv`** - CSV mit allen Test-Ergebnissen
- **`training_report_detail.json`** - Detaillierte JSON-Ergebnisse
- **Beste Parameter** werden automatisch in `config.yaml` gespeichert

### Wann Training Camp verwenden?

**Nach Strategy Validation Tests:**

1. **Strategy Validation Tests ausführen** → Identifiziert problematische Phasen
2. **Parameter manuell anpassen** → Basierend auf Test-Ergebnissen
3. **Training Camp ausführen** → Optimiert Parameter automatisch
4. **Ergebnisse validieren** → Strategy Validation Tests erneut ausführen

---

## 📈 Workflow: Von Tests zu optimierten Parametern

### Schritt 1: Strategy Validation Tests

```bash
# Alle Tests ausführen
python scripts/validation/run_strategy_validation.py --test all
```

**Ergebnis:** `outputs/strategy_validation/summary_*.json`

### Schritt 2: Ergebnisse analysieren

```bash
# Automatische Auswertung
python scripts/utilities/analyze_test_results.py outputs/strategy_validation/summary_*.json
```

**Ergebnis:** Vergleichstabelle + Empfehlungen

### Schritt 3: Parameter manuell anpassen

Basierend auf Analyse-Ergebnissen:
- Problematische Phasen deaktivieren
- Parameter anpassen
- `config.yaml` aktualisieren

### Schritt 4: Training Camp (optional)

```bash
# Training Camp ausführen
python scripts/training/run_automated_testcamp.py
```

**Ergebnis:** Optimierte Parameter in `config.yaml`

### Schritt 5: Validierung

```bash
# Tests erneut ausführen
python scripts/validation/run_strategy_validation.py --test all
```

**Ergebnis:** Verbesserte F1-Scores

---

## 🔍 Beispiel-Workflow

### 1. Initiale Tests

```bash
python scripts/validation/run_strategy_validation.py --test all
```

**Ergebnis:**
- Test 2: Element F1 = 0.85, Connection F1 = 0.82 ✅
- Test 4: Element F1 = 0.78, Connection F1 = 0.65 ✅
- Test 5a: Element F1 = 0.75, Connection F1 = 0.60 ❌ (Verschlechterung!)
- Test 5b: Element F1 = 0.79, Connection F1 = 0.68 ✅ (Verbesserung!)
- Test 5c: Element F1 = 0.78, Connection F1 = 0.66 (Phase 3 läuft nicht)

### 2. Analyse

**Entscheidungen:**
- Test 5a: Predictive Completion deaktivieren (Verschlechterung)
- Test 5b: Polyline Refinement behalten (Verbesserung)
- Test 5c: `self_correction_min_quality_score` auf 90.0 erhöhen

### 3. Parameter anpassen

```yaml
logic_parameters:
  use_predictive_completion: false  # Deaktiviert
  use_polyline_refinement: true     # Aktiviert
  use_self_correction_loop: true
  self_correction_min_quality_score: 90.0  # Erhöht
```

### 4. Training Camp (optional)

```bash
python scripts/training/run_automated_testcamp.py
```

**Ergebnis:** Weitere Parameter-Optimierungen

### 5. Validierung

```bash
python scripts/validation/run_strategy_validation.py --test all
```

**Ergebnis:** Verbesserte F1-Scores

---

## 📝 Zusammenfassung

### Ergebnisse speichern
- ✅ Jeder Test erstellt eigenes Verzeichnis
- ✅ Zusammenfassung in `summary_*.json`
- ✅ Vollständige Logs und KPIs

### Ergebnisse auswerten
- ✅ Manuelle Auswertung (Vergleichstabelle)
- ✅ Automatische Auswertung (`analyze_test_results.py`)
- ✅ Entscheidungen basierend auf F1-Scores

### Parameter anpassen
- ✅ Direkt in `config.yaml`
- ✅ Über `params_override` im Test-Skript
- ✅ Automatisch über Training Camp

### Training Camp
- ✅ Automatische Parameter-Optimierung
- ✅ Testet verschiedene Parameter-Kombinationen
- ✅ Speichert beste Parameter automatisch
- ✅ **Verwendung:** Nach Strategy Validation Tests

---

**Status:** ✅ **Vollständig dokumentiert und bereit für Tests**

