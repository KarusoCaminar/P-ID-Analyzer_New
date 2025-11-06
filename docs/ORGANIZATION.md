# 📁 Dokumentations-Organisation

**Datum:** 2025-11-06  
**Status:** ✅ Organisiert

---

## 📂 Struktur

### 📚 Haupt-Ordner: `docs/`

```
docs/
├── README.md                    # Haupt-Dokumentation (Startpunkt)
├── IMPORTANT_FILES.md          # Code-Review Guide
├── CURRENT_FIXES_SUMMARY.md    # Aktuelle Fixes
├── PIPELINE_OPTIMIZATION_SUMMARY.md  # Pipeline-Optimierungen
├── PIPELINE_PROCESS_DETAILED.md      # Detaillierte Prozessbeschreibung
├── IMPLEMENTATION_SUMMARY.md   # Implementierungs-Zusammenfassung
│
├── analysis/                    # Analyse-Dokumente
│   ├── BEST_RUN_ANALYSIS.md
│   ├── CORE_SYSTEM_TEST.md
│   ├── ELEMENT_TYPE_LIST_AND_LEARNING_DB_ANALYSIS.md
│   ├── META_MODEL_EXPLANATION.md
│   └── META_MODEL_USAGE_ANALYSIS.md
│
├── guides/                      # Anleitungen & Guides
│   ├── TEST_STRATEGY_EXPLANATION.md  # ⭐ WICHTIG: Test-Erklärung
│   ├── QUICK_START.md
│   └── SETUP_ENV_STEPS.txt
│
├── status/                      # Status & Checks
│   ├── CODE_QUALITY_CHECK.md
│   ├── FINAL_TESTS_CHECKLIST.md
│   ├── VERIFICATION_STATUS.md
│   ├── CLEANUP_SUMMARY.md
│   └── OUTPUT_FOLDER_FIX.md
│
├── archiv/                      # Historische Dokumentation
│   └── ...
│
└── Pipeline Diagramme/          # Diagramme
    └── ...
```

---

## 📁 Scripts-Organisation

### 📂 Haupt-Ordner: `scripts/`

```
scripts/
├── README.md                    # Scripts-Organisation
├── test_system_ready.py        # System-Check (Root-Level)
│
├── validation/                   # ⭐ Strategy Validation Tests
│   └── run_strategy_validation.py  # HAUPT-TEST-SKRIPT
│
├── utilities/                    # Utility-Skripte
│   ├── backup_learning_db.py
│   ├── restore_learning_db.py
│   ├── reset_learning_db.py
│   ├── cleanup_outputs.py
│   ├── cleanup_old_files.py
│   ├── cleanup_repo.py
│   └── extract_viewshots_from_uni_bilder.py
│
├── training/                     # Training & Pretraining
│   ├── run_pretraining.py
│   ├── run_automated_testcamp.py
│   ├── run_automated_test.py
│   ├── run_test_harness.py
│   └── run_test_with_validation.py
│
└── legacy/                      # Alte/Deprecated Skripte
    ├── test_*.py                # Alte Test-Skripte
    ├── quick_test.py
    ├── smoke_test_gui.py
    └── visual_trace_debug.py
```

---

## 🔍 Schnellzugriff

### Wichtigste Dokumente

1. **[README.md](README.md)** - Haupt-Dokumentation (Startpunkt)
2. **[TEST_STRATEGY_EXPLANATION.md](guides/TEST_STRATEGY_EXPLANATION.md)** - ⭐ Test-Erklärung & Auswertung
3. **[IMPORTANT_FILES.md](IMPORTANT_FILES.md)** - Code-Review Guide
4. **[CURRENT_FIXES_SUMMARY.md](CURRENT_FIXES_SUMMARY.md)** - Aktuelle Fixes

### Wichtigste Skripte

1. **[run_strategy_validation.py](../scripts/validation/run_strategy_validation.py)** - ⭐ HAUPT-TEST-SKRIPT
2. **[test_system_ready.py](../scripts/test_system_ready.py)** - System-Check

---

**Status:** ✅ **Organisiert und bereit**

