# Nächtlicher Optimierungs-Lauf - Anleitung

**Datum:** 2025-11-06  
**Status:** ✅ Implementiert und bereit

---

## Übersicht

Dieses Dokument beschreibt, wie Sie den nächtlichen Optimierungs-Lauf starten, der automatisch A/B-Tests zwischen verschiedenen Strategien durchführt, KPIs berechnet und am Morgen einen detaillierten Report liefert.

---

## Voraussetzungen

### 1. Viewshots validieren (Manuell)

**Zweck:** Sicherstellen, dass die Viewshots von hoher Qualität sind.

**Anweisung:**
1. Öffnen Sie den Ordner `training_data/viewshot_examples/`
2. Blättern Sie stichprobenartig durch die Bilder in den Unterordnern (z.B. `pump/`, `valve/`, etc.)
3. Prüfen Sie, ob jedes Bild ein sauberes, klar erkennbares Symbol-Beispiel ist

**Kriterium:** Jedes Bild sollte ein sauberes, klar erkennbares Beispiel für das Symbol sein. Wenn hier fehlerhafte (z.B. halb abgeschnittene oder falsche) Bilder enthalten sind, wird der SwarmAnalyzer falsch trainiert.

**Dauer:** 5-10 Minuten

### 2. Pre-Training ausführen

**Zweck:** `learning_db.json` mit Vektor-Embeddings aller Symbole füllen.

**Befehl:**
```bash
python scripts/training/run_pretraining.py
```

**Erwartung:** Die `learning_db.json` wird erstellt/aktualisiert. Die SymbolLibrary ist jetzt "geladen" und kann für die Ähnlichkeitssuche in der Hauptpipeline verwendet werden.

**Dauer:** 10-15 Minuten

### 3. Test-Konfiguration validieren

**Dateien:**
- `training_data/simple_pids/Einfaches P&I.png` + `Einfaches P&I_truth.json`
- `training_data/complex_pids/page_1_original.png` + `page_1_original_truth_cgm.json`

**Aktion:** Das Skript validiert automatisch, ob alle Test-Bilder und Ground Truth-Dateien vorhanden sind.

---

## Nächtlichen Lauf starten

### Basis-Befehl

```bash
python scripts/validation/run_overnight_optimization.py
```

**Standard-Dauer:** 8 Stunden

### Optionen

```bash
# Andere Dauer (z.B. 4 Stunden)
python scripts/validation/run_overnight_optimization.py --duration 4.0

# Anderes Output-Verzeichnis
python scripts/validation/run_overnight_optimization.py --output-dir outputs/my_overnight_test
```

### Vollständige Optionen

```bash
python scripts/validation/run_overnight_optimization.py \
    --duration 8.0 \
    --output-dir outputs/overnight_optimization
```

---

## Was passiert während des Laufs?

### 1. A/B-Tests

Das Skript führt automatisch A/B-Tests zwischen folgenden Strategien durch:
- **`simple_whole_image`**: Monolith-Analyse auf ganzem Bild (beste für einfache P&IDs)
- **`default_flash`**: Swarm-Baseline (Standard-Strategie)

### 2. Parameter-Optimierung

Für jede Strategie werden verschiedene Parameter-Kombinationen getestet:
- **IoU-Threshold:** [0.3, 0.4, 0.5, 0.6]
- **Confidence-Threshold:** [0.5, 0.6, 0.7, 0.8]
- **Self-Correction-Min-Score:** [85.0, 90.0, 95.0]

### 3. Test-Bilder

Jede Strategie wird auf folgenden Bildern getestet:
- **Simple P&ID:** `training_data/simple_pids/Einfaches P&I.png`
- **Complex P&ID:** `training_data/complex_pids/page_1_original.png`

### 4. KPI-Berechnung

Für jeden Test werden automatisch KPIs berechnet:
- Element F1-Score, Precision, Recall
- Connection F1-Score, Precision, Recall
- Graph-Edit-Distance (GED)
- Type Accuracy
- Quality Score

### 5. Active Learning

**Aktivierung:** Nur für erfolgreiche Tests (Score > 0.8)

**Zweck:** System lernt nur aus validierten, guten Ergebnissen, um schlechte Korrekturen zu vermeiden.

### 6. Error-Handling

- **Automatischer Neustart:** Bei Fehlern wird der Test automatisch neu gestartet (max. 3 Versuche)
- **Circuit Breaker Reset:** Vor jedem Test wird Circuit Breaker zurückgesetzt
- **ThreadPool Cleanup:** Nach jedem Test wird ThreadPoolExecutor ordnungsgemäß geschlossen

---

## Ausgabe-Struktur

```
outputs/overnight_optimization/
├── logs/
│   ├── overnight_YYYYMMDD_HHMMSS.log
│   └── errors_YYYYMMDD_HHMMSS.log
├── reports/
│   ├── report_YYYYMMDD_HHMMSS.html
│   └── report_YYYYMMDD_HHMMSS.json
└── test_results/
    ├── simple_whole_image/
    │   ├── Einfaches_P&I_YYYYMMDD_HHMMSS.json
    │   └── page_1_original_YYYYMMDD_HHMMSS.json
    └── default_flash/
        ├── Einfaches_P&I_YYYYMMDD_HHMMSS.json
        └── page_1_original_YYYYMMDD_HHMMSS.json
```

---

## Report-Analyse am Morgen

### HTML-Report

**Datei:** `outputs/overnight_optimization/reports/report_YYYYMMDD_HHMMSS.html`

**Inhalt:**
- Vergleich zwischen `simple_whole_image` und `default_flash`
- KPIs pro Strategie und Test-Bild
- Beste Parameter-Kombination
- Verbesserungs-Historie
- Empfehlungen für nächste Schritte

**Öffnen:** Öffnen Sie die HTML-Datei in Ihrem Browser.

### JSON-Report

**Datei:** `outputs/overnight_optimization/reports/report_YYYYMMDD_HHMMSS.json`

**Inhalt:** Vollständige Daten für spätere Analyse

**Format:** Strukturiertes JSON mit allen KPIs, Parametern und Ergebnissen

---

## KPIs verstehen

### Element F1-Score

**Bedeutung:** Wie gut werden Elemente erkannt?

**Berechnung:** `2 * (Precision * Recall) / (Precision + Recall)`

**Erwartung:**
- **Einfaches P&ID:** > 0.95 (nahe 1.0) für `simple_whole_image`
- **Komplexes P&ID:** > 0.80 für `default_flash`

### Connection F1-Score

**Bedeutung:** Wie gut werden Verbindungen erkannt?

**Berechnung:** `2 * (Precision * Recall) / (Precision + Recall)`

**Erwartung:**
- **Einfaches P&ID:** > 0.90 für `simple_whole_image`
- **Komplexes P&ID:** > 0.70 für `default_flash`

### Graph-Edit-Distance (GED)

**Bedeutung:** Strukturelle Ähnlichkeit zwischen Analyse und Ground Truth

**Berechnung:** Normalisierte Anzahl von Operationen (Add/Delete Edges) zum Transformieren des Analyse-Graphen in den Ground-Truth-Graphen

**Erwartung:**
- **Normalized GED:** < 0.2 (niedriger ist besser)
- **Graph Similarity Score:** > 0.8 (höher ist besser)

### Type Accuracy

**Bedeutung:** Wie genau werden Element-Typen erkannt?

**Berechnung:** Anteil der korrekt erkannten Element-Typen

**Erwartung:**
- **Einfaches P&ID:** > 0.95
- **Komplexes P&ID:** > 0.85

**Wichtig:** Dank `NormalizationEngine` sollte der Typ "S" jetzt "Sample Point" sein und nicht mehr "pipe".

### Quality Score

**Bedeutung:** Gesamt-Qualitäts-Score (0-100)

**Berechnung:** Kombination aus Element F1, Connection F1, Type Accuracy, etc.

**Erwartung:**
- **Einfaches P&ID:** > 90 für `simple_whole_image`
- **Komplexes P&ID:** > 75 für `default_flash`

---

## Interpretation der Ergebnisse

### 1. Strategie-Vergleich

**Frage:** Welche Strategie ist besser?

**Antwort:** Vergleichen Sie die durchschnittlichen F1-Scores:
- **Element F1:** Höher ist besser
- **Connection F1:** Höher ist besser
- **Quality Score:** Höher ist besser

**Erwartung:** `simple_whole_image` sollte für einfache P&IDs besser sein als `default_flash`.

### 2. Beste Parameter-Kombination

**Frage:** Welche Parameter-Kombination liefert die besten Ergebnisse?

**Antwort:** Suchen Sie im Report nach der besten Parameter-Kombination (höchster Quality Score).

**Verwendung:** Diese Parameter können in `config.yaml` als Standard gesetzt werden.

### 3. Active Learning

**Frage:** Hat das System aus erfolgreichen Tests gelernt?

**Antwort:** Prüfen Sie `training_data/learning_db.json`:
- Neue Einträge in `learned_visual_corrections`?
- Neue Typ-Aliases?
- Verbesserte Embeddings?

**Konsequenz:** Wenn Sie den nächtlichen Lauf ein zweites Mal laufen lassen, sollte die Pipeline diese Fehler proaktiv vermeiden, was zu einer (leichten) Verbesserung der Scores führen müsste.

---

## Troubleshooting

### Problem: GCP_PROJECT_ID nicht gesetzt

**Lösung:**
```bash
# Windows PowerShell
$env:GCP_PROJECT_ID='dein_project_id'
$env:GCP_LOCATION='us-central1'

# Windows CMD
set GCP_PROJECT_ID=dein_project_id
set GCP_LOCATION=us-central1

# Oder .env Datei erstellen
GCP_PROJECT_ID=dein_project_id
GCP_LOCATION=us-central1
```

### Problem: Test-Bilder nicht gefunden

**Lösung:** Stellen Sie sicher, dass folgende Dateien existieren:
- `training_data/simple_pids/Einfaches P&I.png`
- `training_data/simple_pids/Einfaches P&I_truth.json`
- `training_data/complex_pids/page_1_original.png`
- `training_data/complex_pids/page_1_original_truth_cgm.json`

### Problem: Ground Truth nicht gefunden

**Lösung:** Das Skript läuft trotzdem, aber ohne KPI-Berechnung. Stellen Sie sicher, dass die Ground Truth-Dateien vorhanden sind.

### Problem: Skript stürzt ab

**Lösung:** 
1. Prüfen Sie die Logs in `outputs/overnight_optimization/logs/`
2. Prüfen Sie, ob GCP-Credentials korrekt gesetzt sind
3. Prüfen Sie, ob genug Speicherplatz vorhanden ist
4. Prüfen Sie, ob die API-Quoten nicht überschritten wurden

### Problem: Keine Verbesserung nach mehreren Läufen

**Lösung:**
1. Prüfen Sie, ob Active Learning aktiviert ist (nur für Score > 0.8)
2. Prüfen Sie `learning_db.json` auf neue Einträge
3. Prüfen Sie, ob die Parameter-Kombinationen sinnvoll sind
4. Erwägen Sie, die Parameter-Ranges anzupassen

---

## Nächste Schritte

### 1. Ergebnisse analysieren

- Öffnen Sie den HTML-Report
- Vergleichen Sie die Strategien
- Identifizieren Sie die beste Parameter-Kombination

### 2. Parameter anpassen

- Setzen Sie die beste Parameter-Kombination in `config.yaml`
- Führen Sie erneut einen nächtlichen Lauf durch
- Vergleichen Sie die Ergebnisse

### 3. Iterativ verbessern

- Test → Analyse → Anpassung → Test
- Bis optimale Performance erreicht ist

---

## Erfolgskriterien

1. ✅ **Skript läuft 8 Stunden ohne Absturz**
2. ✅ **A/B-Tests werden korrekt durchgeführt**
3. ✅ **KPIs werden automatisch berechnet**
4. ✅ **HTML-Report wird am Morgen generiert**
5. ✅ **Beste Parameter-Kombination wird identifiziert**
6. ✅ **System lernt aus erfolgreichen Tests (Score > 0.8)**

---

**Status:** ✅ **Bereit für nächtlichen Optimierungs-Lauf**

