# 📁 Scripts - Organisation

## 📂 Struktur

### 🧪 `validation/` - Test & Validation Skripte

**Haupt-Test-Skripte:**

- **`run_live_test.py`** ⭐ **HAUPT-TEST-SKRIPT**
  - Führt einen vollständigen Test mit Live-Log-Monitoring durch
  - Verwendet strukturierte Output-Ordner
  - Zeigt Logs live im Terminal an
  - **Verwendung:** `python scripts/validation/run_live_test.py`

- **`run_simple_test.py`**
  - Einfacher Test-Runner für schnelle Tests
  - Testet eine einzelne Konfiguration
  - **Verwendung:** `python scripts/validation/run_simple_test.py`

- **`run_strategy_validation.py`**
  - Strategy-Validation-Tests
  - Führt mehrere Strategien nacheinander aus
  - Berechnet KPIs für jede Strategie
  - **Verwendung:** `python scripts/validation/run_strategy_validation.py --test "Test 2"`

- **`run_strategy_validation_with_logs.py`**
  - Wrapper für `run_strategy_validation.py` mit Live-Log-Anzeige
  - **Verwendung:** `python scripts/validation/run_strategy_validation_with_logs.py`

- **`run_overnight_optimization.py`**
  - Overnight A/B Testing
  - Führt automatische A/B-Tests zwischen Strategien durch
  - Generiert umfassende Reports
  - **Verwendung:** `python scripts/validation/run_overnight_optimization.py`

**Monitoring-Skripte:**

- **`monitor_overnight.py`**: Überwacht Overnight-Prozess
- **`watchdog_overnight.py`**: Watchdog für Overnight
- **`auto_guardian.py`**: Auto-Guardian für Überwachung
- **`continuous_monitor.py`**: Kontinuierlicher Monitor
- **`test_startup_speed.py`**: Startup-Speed-Test
- **`diagnose_hang.py`**: Diagnose-Tool für Hangs

### 🎓 `training/` - Training & Pretraining

**Haupt-Training-Skripte:**

- **`build_vector_indices.py`** ⭐ **WICHTIG**
  - Erstellt Vektor-Indizes für schnellen Startup
  - **MUSS** ausgeführt werden nach Änderungen an `learning_db.json`
  - **Verwendung:** `python scripts/training/build_vector_indices.py`

- **`run_pretraining.py`**
  - Symbol-Pretraining
  - Verarbeitet alle Symbole aus `pretraining_symbols/`
  - **Verwendung:** `python scripts/training/run_pretraining.py`

- **`run_pretraining_stepwise.py`**
  - Stepwise Pretraining
  - Testet Uni-Legenden-Bilder zuerst, dann PDF-Collection
  - **Verwendung:** `python scripts/training/run_pretraining_stepwise.py`

**Optional (für Debugging):**

- **`test_pretraining.py`**: Testet Pretraining-Qualität
- **`check_extracted_symbols.py`**: Prüft extrahierte Symbole
- **`evaluate_extracted_symbols.py`**: Evaluiert extrahierte Symbole

### 🔧 `utilities/` - Utility-Skripte

**Learning-Database-Management:**

- **`backup_learning_db.py`**: Backup der Learning Database
- **`restore_learning_db.py`**: Wiederherstellung der Learning Database
- **`reset_learning_db.py`**: Zurücksetzen der Learning Database

**Cleanup:**

- **`cleanup_outputs.py`**: Aufräumen der Output-Ordner

**Viewshot-Extraktion:**

- **`extract_viewshots_from_pretraining_pdf.py`**: Viewshot-Extraktion aus PDF-Collection
- **`extract_viewshots_from_uni_bilder.py`**: Viewshot-Extraktion aus Uni-Bildern

### 🛠️ `utils/` - Script-Utilities

- **`live_log_monitor.py`**: Live-Log-Monitoring für Test-Skripte

### 📊 `legacy/` - Veraltete Skripte

**Alte/Deprecated Skripte (nur für Referenz):**

- Alle Skripte in diesem Ordner sind veraltet und werden nicht mehr verwendet
- Siehe Haupt-README.md für aktuelle Skripte

---

## 🚀 Haupt-Skripte (Root-Level)

- **`run_cli.py`**: CLI-Starter für Kommandozeile
- **`run_gui.py`**: GUI-Starter für grafische Benutzeroberfläche

---

## 📝 Verwendung

### Live-Test (Empfohlen)

```bash
# Führt einen vollständigen Test mit Live-Log-Monitoring durch
python scripts/validation/run_live_test.py
```

### Einfacher Test

```bash
# Führt einen einfachen Test durch
python scripts/validation/run_simple_test.py
```

### Strategy-Validation-Tests

```bash
# Einzelnen Test ausführen
python scripts/validation/run_strategy_validation.py --test "Test 2"

# Alle Tests ausführen
python scripts/validation/run_strategy_validation.py --test all

# Mit eigenem Bild
python scripts/validation/run_strategy_validation.py --test "Test 4" --image "data/input/Complex.png"
```

### Overnight-Optimization

```bash
# Startet Overnight A/B Testing
python scripts/validation/run_overnight_optimization.py
```

### Training

```bash
# Vector-Indizes erstellen (WICHTIG für schnellen Startup)
python scripts/training/build_vector_indices.py

# Pretraining ausführen
python scripts/training/run_pretraining.py

# Stepwise Pretraining
python scripts/training/run_pretraining_stepwise.py
```

### Utilities

```bash
# Backup Learning DB
python scripts/utilities/backup_learning_db.py

# Restore Learning DB
python scripts/utilities/restore_learning_db.py

# Reset Learning DB
python scripts/utilities/reset_learning_db.py

# Cleanup Outputs
python scripts/utilities/cleanup_outputs.py
```

---

## 🔄 Migration

Veraltete Skripte wurden entfernt oder in `legacy/` verschoben. Die aktuellen Skripte sind:

- ✅ `run_live_test.py` - Haupt-Test-Skript
- ✅ `run_strategy_validation.py` - Strategy-Validation-Tests
- ✅ `run_overnight_optimization.py` - Overnight A/B Testing
- ✅ `build_vector_indices.py` - Vector-Indizes erstellen

**Siehe Haupt-README.md für vollständige Dokumentation.**

---

## 📚 Weitere Informationen

- **[Haupt-README.md](../README.md)**: Vollständige Projekt-Dokumentation
- **[Pipeline-Dokumentation](../docs/PIPELINE_PROCESS_DETAILED.md)**: Detaillierte Pipeline-Beschreibung
- **[Output-Struktur](../docs/OUTPUT_STRUCTURE_STANDARD.md)**: Gold Standard für Output-Ordner
