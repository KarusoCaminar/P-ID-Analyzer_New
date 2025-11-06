# 📁 Scripts - Organisation

## 📂 Struktur

### 🧪 `validation/` - Strategy Validation Tests
**Haupt-Test-Skript für Pipeline-Isolation & Integration**

- **`run_strategy_validation.py`** ⭐ **HAUPT-TEST-SKRIPT**
  - Führt alle Strategy-Validation-Tests aus
  - Misst F1-Scores für verschiedene Pipeline-Konfigurationen
  - Validiert gegen Ground Truth-Daten
  - **Verwendung:** `python scripts/validation/run_strategy_validation.py --test "Test 2"`

### 🔧 `utilities/` - Utility-Skripte
**Hilfs-Skripte für Wartung und Verwaltung**

- **`backup_learning_db.py`** - Backup der Learning Database
- **`restore_learning_db.py`** - Wiederherstellung der Learning Database
- **`reset_learning_db.py`** - Zurücksetzen der Learning Database
- **`cleanup_outputs.py`** - Aufräumen der Output-Ordner
- **`cleanup_old_files.py`** - Aufräumen alter Dateien
- **`cleanup_repo.py`** - Repository-Cleanup
- **`extract_viewshots_from_uni_bilder.py`** - Viewshot-Extraktion

### 🎓 `training/` - Training & Pretraining
**Training und Pretraining-Skripte**

- **`run_pretraining.py`** - Symbol-Pretraining
- **`run_automated_testcamp.py`** - Automatisiertes Testcamp
- **`run_automated_test.py`** - Automatisierte Tests
- **`run_test_harness.py`** - Test-Harness
- **`run_test_with_validation.py`** - Tests mit Validierung

### 📊 `legacy/` - Alte/Deprecated Skripte
**Veraltete Skripte (nur für Referenz)**

- **`test_*.py`** - Alte Test-Skripte (ersetzt durch `run_strategy_validation.py`)
- **`test_imports.py`** - Import-Test (ersetzt durch `tests/test_imports.py`)
- **`quick_test.py`** - Quick-Test (ersetzt durch `run_strategy_validation.py`)
- **`smoke_test_gui.py`** - GUI-Smoke-Test
- **`visual_trace_debug.py`** - Visual-Trace-Debug

---

## 🚀 Haupt-Skripte (Root-Level)

### Strategy Validation
- **`run_strategy_validation.py`** ⭐ **WICHTIGSTES TEST-SKRIPT**
  - Führt alle Strategy-Validation-Tests aus
  - Siehe `validation/` Ordner

### System-Checks
- **`test_system_ready.py`** - System-Readiness-Check
  - Prüft ob alle Module importiert werden können
  - Prüft Konfiguration
  - Prüft GCP-Credentials

---

## 📝 Verwendung

### Strategy Validation Tests
```bash
# Einzelnen Test ausführen
python scripts/validation/run_strategy_validation.py --test "Test 2"

# Alle Tests ausführen
python scripts/validation/run_strategy_validation.py --test all

# Mit eigenem Bild
python scripts/validation/run_strategy_validation.py --test "Test 4" --image "data/input/Complex.png"
```

### System-Check
```bash
python scripts/test_system_ready.py
```

### Utilities
```bash
# Backup Learning DB
python scripts/utilities/backup_learning_db.py

# Cleanup Outputs
python scripts/utilities/cleanup_outputs.py
```

### Training
```bash
# Pretraining
python scripts/training/run_pretraining.py

# Automated Testcamp
python scripts/training/run_automated_testcamp.py
```

---

## 🔄 Migration

Alte Test-Skripte wurden in `legacy/` verschoben und werden durch `run_strategy_validation.py` ersetzt.

**Neue Test-Strategie:** Siehe `tests/STRATEGY_VALIDATION.md`

