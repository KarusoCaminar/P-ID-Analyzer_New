# 🧪 Pretraining Test Guide

**Datum:** 2025-11-06  
**Status:** ✅ Konfiguriert und bereit

---

## 📋 Übersicht

Dieses Dokument beschreibt, wie das Pretraining (Symbolvortraining) getestet und ausgewertet wird, **BEVOR** Viewshots generiert werden.

---

## 🎯 Was macht Pretraining?

Das Pretraining-Skript (`run_pretraining.py`):

1. **Verarbeitet alle Bilder** in `training_data/pretraining_symbols/`
2. **Erkennt automatisch**, ob Bilder Sammlungen (groß) oder einzelne Symbole sind
3. **Extrahiert Symbole** aus Sammlungen automatisch (z.B. aus PDF-Sammlung)
4. **Integriert Symbole** in die Symbol-Bibliothek mit Duplikat-Prüfung
5. **Speichert** in Learning Database (`learning_db.json`)

---

## 🧪 Pretraining testen

### Voraussetzungen

1. **GCP-Credentials setzen:**
   ```powershell
   $env:GCP_PROJECT_ID='dein_project_id'
   $env:GCP_LOCATION='us-central1'
   ```

2. **Pretraining-Verzeichnis prüfen:**
   - `training_data/pretraining_symbols/` sollte existieren
   - Enthält Symbol-Bilder (z.B. `Pid-symbols-PDF_sammlung.png`)

### Test ausführen

```bash
# Test-Skript ausführen (mit Auswertung)
python scripts/training/test_pretraining.py

# Oder direkt Pretraining ausführen
python scripts/training/run_pretraining.py
```

### Was wird getestet?

1. **Symbol-Extraktion:**
   - Werden Symbole aus Sammlungen korrekt extrahiert?
   - Werden einzelne Symbole korrekt verarbeitet?

2. **Symbol-Integration:**
   - Werden Symbole korrekt in die Bibliothek integriert?
   - Werden Duplikate korrekt erkannt?

3. **Learning Database:**
   - Werden Symbole korrekt in `learning_db.json` gespeichert?

---

## 📊 Test-Ergebnisse

### Output-Ordnerstruktur

```
outputs/pretraining_tests/
├── test_pretraining_YYYYMMDD_HHMMSS.log    # Test-Logs
└── test_results_YYYYMMDD_HHMMSS.json        # Test-Ergebnisse
```

### Test-Ergebnisse JSON

```json
{
  "timestamp": "2025-11-06T16:30:00",
  "success": true,
  "errors": [],
  "warnings": [],
  "metrics": {
    "files_found": 1,
    "initial_symbol_count": 0,
    "final_symbol_count": 150,
    "symbols_added": 150,
    "symbols_updated": 0,
    "duplicates_found": 5
  },
  "symbols_extracted": 1,
  "symbols_learned": 150,
  "collections_processed": 1,
  "individual_symbols_processed": 0
}
```

### Metriken

- **files_found:** Anzahl gefundener Bilddateien
- **initial_symbol_count:** Anzahl Symbole vor Pretraining
- **final_symbol_count:** Anzahl Symbole nach Pretraining
- **symbols_added:** Anzahl neu hinzugefügter Symbole
- **symbols_updated:** Anzahl aktualisierter Symbole
- **duplicates_found:** Anzahl gefundener Duplikate
- **symbols_extracted:** Anzahl extrahierter Symbole
- **symbols_learned:** Anzahl gelernte Symbole
- **collections_processed:** Anzahl verarbeiteter Sammlungen
- **individual_symbols_processed:** Anzahl verarbeiteter einzelner Symbole

---

## 🔍 Pretraining auswerten

### 1. Symbol-Extraktion prüfen

```python
# Beispiel: Test-Ergebnisse laden
import json
from pathlib import Path

results_file = Path("outputs/pretraining_tests/test_results_*.json")
with open(results_file, 'r') as f:
    results = json.load(f)

# Prüfe Extraktion
print(f"Symbols extracted: {results['symbols_extracted']}")
print(f"Symbols learned: {results['symbols_learned']}")
print(f"Collections processed: {results['collections_processed']}")
```

### 2. Learning Database prüfen

```python
# Beispiel: Learning Database laden
import json
from pathlib import Path

learning_db = Path("learning_db.json")
with open(learning_db, 'r') as f:
    db = json.load(f)

# Prüfe Symbole
symbols = db.get('symbols', [])
print(f"Total symbols in database: {len(symbols)}")

# Prüfe Symbol-Typen
symbol_types = {}
for symbol in symbols:
    symbol_type = symbol.get('type', 'unknown')
    symbol_types[symbol_type] = symbol_types.get(symbol_type, 0) + 1

print("Symbol types:")
for symbol_type, count in symbol_types.items():
    print(f"  {symbol_type}: {count}")
```

### 3. Fehler prüfen

```python
# Beispiel: Fehler prüfen
if results['errors']:
    print(f"Errors encountered: {len(results['errors'])}")
    for error in results['errors']:
        print(f"  - {error}")
```

---

## ⚠️ Häufige Probleme

### 1. Pretraining hängt

**Problem:** Das Skript hängt bei der Symbol-Extraktion.

**Ursachen:**
- LLM-API-Latenz (kann bei vielen Symbolen lange dauern)
- Große Sammlungen (z.B. PDF-Sammlung mit 100+ Symbolen)
- Netzwerkprobleme

**Lösung:**
- Prüfe Logs: `outputs/pretraining_tests/test_pretraining_*.log`
- Prüfe LLM-API-Status
- Reduziere Anzahl Symbole pro Durchlauf

### 2. Keine Symbole extrahiert

**Problem:** Keine Symbole werden extrahiert.

**Ursachen:**
- Falsche Bildformate
- Zu kleine/große Bilder
- Fehlende LLM-API-Credentials

**Lösung:**
- Prüfe Bildformate (PNG, JPG, JPEG)
- Prüfe Bildgröße (min. 50x50 Pixel)
- Prüfe GCP-Credentials

### 3. Duplikate nicht erkannt

**Problem:** Duplikate werden nicht erkannt.

**Ursachen:**
- Falsche Duplikat-Erkennung
- Ähnliche aber unterschiedliche Symbole

**Lösung:**
- Prüfe Duplikat-Erkennungs-Logik
- Manuelle Prüfung der Symbole

---

## 📈 Nächste Schritte

### Nach erfolgreichem Pretraining

1. **Viewshots generieren:**
   ```bash
   python scripts/utilities/extract_viewshots_from_uni_bilder.py
   ```

2. **Viewshots testen:**
   - Prüfe, ob Viewshots korrekt generiert wurden
   - Prüfe, ob Viewshots in Prompts verwendet werden

3. **Pipeline testen:**
   - Führe Tests mit Viewshots aus
   - Vergleiche Performance mit/ohne Viewshots

---

## ✅ Checkliste

### Vor Pretraining

- [ ] GCP-Credentials gesetzt
- [ ] Pretraining-Verzeichnis vorhanden
- [ ] Symbol-Bilder vorhanden
- [ ] Learning Database vorhanden

### Nach Pretraining

- [ ] Test erfolgreich abgeschlossen
- [ ] Symbole in Learning Database gespeichert
- [ ] Test-Ergebnisse gespeichert
- [ ] Fehler geprüft und behoben

---

## 🎯 Zusammenfassung

1. **Pretraining testen:** `python scripts/training/test_pretraining.py`
2. **Ergebnisse prüfen:** `outputs/pretraining_tests/test_results_*.json`
3. **Learning Database prüfen:** `learning_db.json`
4. **Viewshots generieren:** Nach erfolgreichem Pretraining
5. **Pipeline testen:** Mit Viewshots

---

**Status:** ✅ **Bereit für Pretraining-Test**

