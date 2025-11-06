# 🚀 Strategy Validation Tests: Start-Anleitung

**Datum:** 2025-11-06

---

## ⏱️ Laufzeit-Schätzung

### Einzelner Test

- **Test 1 (Phase 1 only):** ~15-30 Sekunden
- **Test 2 (Monolith-All):** ~60-120 Sekunden (1-2 Minuten) ⭐ **START HIER**
- **Test 3 (Swarm-Only):** ~90-150 Sekunden (1.5-2.5 Minuten)
- **Test 4 (Complex P&ID):** ~120-240 Sekunden (2-4 Minuten)
- **Test 5a/5b/5c:** ~150-300 Sekunden (2.5-5 Minuten)

### Alle Tests (7 Tests)

- **Minimum:** ~10-15 Minuten
- **Typisch:** ~20-30 Minuten
- **Maximum:** ~40-60 Minuten

**Faktoren:**
- LLM-API-Latenz (kann variieren)
- Bildgröße (größere Bilder = mehr Tiles)
- Komplexität (mehr Elemente = mehr LLM-Calls)

---

## 🚀 Test starten mit Live-Logs

### Schritt 1: GCP-Credentials setzen

**Windows PowerShell:**
```powershell
$env:GCP_PROJECT_ID='dein_project_id'
$env:GCP_LOCATION='us-central1'
```

**Windows CMD:**
```cmd
set GCP_PROJECT_ID=dein_project_id
set GCP_LOCATION=us-central1
```

**Oder .env Datei erstellen** (im Projekt-Root):
```
GCP_PROJECT_ID=dein_project_id
GCP_LOCATION=us-central1
```

### Schritt 2: Test starten

**Einzelnen Test (z.B. Test 2):**
```bash
python scripts/validation/run_strategy_validation.py --test "Test 2"
```

**Alle Tests:**
```bash
python scripts/validation/run_strategy_validation.py --test all
```

**Logs werden live im Terminal angezeigt!**

---

## 📊 Was Sie in den Logs sehen

### 1. Initialisierung
```
[2025-11-06 12:00:00 - INFO] Services erfolgreich initialisiert
[2025-11-06 12:00:00 - INFO] Ground Truth geladen: 15 elements, 12 connections
```

### 2. Test-Start
```
============================================================
🚀 Starte Test: Test 2: Baseline Simple P&ID (Monolith-All)
Overrides: {
  "use_swarm_analysis": false,
  "use_monolith_analysis": true,
  ...
}
============================================================
```

### 3. Pipeline-Ausführung
```
[2025-11-06 12:00:05 - INFO] Phase 0: Complexity Analysis...
[2025-11-06 12:00:10 - INFO] Phase 1: Pre-analysis...
[2025-11-06 12:00:15 - INFO] Phase 2: Simple P&ID analysis...
[2025-11-06 12:00:45 - INFO] Monolith analysis complete: 15 elements, 12 connections
[2025-11-06 12:00:50 - INFO] Phase 4: Post-processing...
```

### 4. Test-Ergebnis
```
🏁 Test 'Test 2: Baseline Simple P&ID (Monolith-All)' Abgeschlossen:
  Element F1:    0.8500
  Element Precision: 0.8800
  Element Recall:    0.8200
  Connection F1: 0.8200
  Connection Precision: 0.8500
  Connection Recall:    0.7900
============================================================
```

### 5. Finale Zusammenfassung
```
============================================================
📈 FINALE KPI-ZUSAMMENFASSUNG
============================================================
[Test 2: Baseline Simple P&ID (Monolith-All)]:
  Element F1:    0.8500
  Connection F1: 0.8200
...
Zusammenfassung gespeichert: outputs/strategy_validation/summary_20250101_120000.json
```

---

## 📁 Wo werden Ergebnisse gespeichert?

### 1. Test-Verzeichnisse
```
outputs/strategy_validation/
├── Test_2_Baseline_Simple_PID/
│   ├── results.json          # Vollständige Analyse-Ergebnisse
│   ├── pipeline.log          # Pipeline-Log
│   ├── llm_calls.log         # LLM-Aufrufe
│   └── kpis.json             # KPIs (falls Ground Truth verfügbar)
└── ...
```

### 2. Zusammenfassung
```
outputs/strategy_validation/summary_YYYYMMDD_HHMMSS.json
```

**Inhalt:**
```json
{
  "timestamp": "2025-01-01T12:00:00",
  "image_path": "data/input/Einfaches P&I.png",
  "ground_truth": "data/ground_truth/Einfaches P&I.json",
  "results": {
    "Test 2: Baseline Simple P&ID (Monolith-All)": {
      "element_f1": 0.8500,
      "connection_f1": 0.8200,
      ...
    }
  }
}
```

---

## 💡 Empfehlung

**Starten Sie mit Test 2** (~1-2 Minuten):
- Schnellster Test
- Gute Baseline
- Zeigt, ob alles funktioniert

```bash
python scripts/validation/run_strategy_validation.py --test "Test 2"
```

**Dann alle Tests ausführen** (~20-30 Minuten):
```bash
python scripts/validation/run_strategy_validation.py --test all
```

---

**Status:** ✅ **Bereit für Tests mit Live-Logs**

