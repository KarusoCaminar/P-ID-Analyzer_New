# 🧪 Test-Konfiguration: Bilder pro Test

**Datum:** 2025-11-06  
**Status:** ✅ Konfiguriert

---

## 📋 Test-Bilder Konfiguration

### Test 1: Baseline Phase 1 (Legenden-Erkennung)
- **Bild:** `training_data/complex_pids/page_1_original.png` ⭐ **MIT Legende**
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Grund:** Test 1 testet Legenden-Erkennung → benötigt Bild MIT Legende

### Test 2: Baseline Simple P&ID (Monolith-All)
- **Bild:** `training_data/simple_pids/Einfaches P&I.png` ⭐ **Simple P&ID**
- **Ground Truth:** `training_data/simple_pids/Einfaches P&I_truth.json`
- **Grund:** Test 2 testet Monolith-Only → einfaches Bild für Baseline

### Test 3: Baseline Swarm-Only
- **Bild:** `training_data/simple_pids/Einfaches P&I.png` ⭐ **Simple P&ID**
- **Ground Truth:** `training_data/simple_pids/Einfaches P&I_truth.json`
- **Grund:** Test 3 testet Swarm-Only → einfaches Bild für Baseline

### Test 4: Baseline Complex P&ID (Spezialisten-Kette)
- **Bild:** `training_data/complex_pids/page_1_original.png` ⭐ **Komplexes Bild**
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Grund:** Test 4 testet vollständige Pipeline → komplexes Bild mit Legende

### Test 5a: Test 4 + Predictive (2d)
- **Bild:** `training_data/complex_pids/page_1_original.png` ⭐ **Komplexes Bild**
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Grund:** Basiert auf Test 4 → gleiches Bild

### Test 5b: Test 4 + Polyline (2e)
- **Bild:** `training_data/complex_pids/page_1_original.png` ⭐ **Komplexes Bild**
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Grund:** Basiert auf Test 4 → gleiches Bild

### Test 5c: Test 4 + Self-Correction (3)
- **Bild:** `training_data/complex_pids/page_1_original.png` ⭐ **Komplexes Bild**
- **Ground Truth:** `training_data/complex_pids/page_1_original_truth_cgm.json`
- **Grund:** Basiert auf Test 4 → gleiches Bild

---

## 🔧 Automatische .env-Ladung

Alle Skripte laden jetzt automatisch die `.env` Datei:

- ✅ `run_cli.py` - CLI-Skript
- ✅ `run_gui.py` - GUI-Skript
- ✅ `scripts/validation/run_strategy_validation.py` - Test-Skript
- ✅ `src/utils/env_loader.py` - Zentraler Loader

**Verwendung:**
```python
from src.utils.env_loader import load_env_automatically
load_env_automatically()
```

---

## 📊 Test-Reihenfolge

Tests werden in dieser Reihenfolge ausgeführt:

1. **Test 1** - Legenden-Erkennung (Komplexes Bild)
2. **Test 2** - Simple P&ID Monolith (Simple Bild)
3. **Test 3** - Simple P&ID Swarm (Simple Bild)
4. **Test 4** - Complex P&ID Pipeline (Komplexes Bild)
5. **Test 5a** - Test 4 + Predictive (Komplexes Bild)
6. **Test 5b** - Test 4 + Polyline (Komplexes Bild)
7. **Test 5c** - Test 4 + Self-Correction (Komplexes Bild)

---

**Status:** ✅ **Konfiguriert und bereit für Tests**

