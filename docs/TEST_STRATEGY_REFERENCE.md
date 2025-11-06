# Test-Strategie: Referenz

**Datum:** 2025-11-06  
**Status:** ✅ Korrigierte Test-Konfigurationen

---

## Test-Übersicht

| Test | Ziel | Bild | Feature Flags |
|------|------|------|---------------|
| **Test 1** | Phase 1 (Legenden-Erkennung) | Complex P&ID (mit Legende) | Nur Phase 1 aktiv |
| **Test 2** | Baseline Simple P&ID (Monolith-All) | Simple P&ID | Monolith aktiv |
| **Test 3** | Baseline Swarm-Only | Simple P&ID | Swarm aktiv |
| **Test 4** | Baseline Complex P&ID (Spezialisten-Kette) | Complex P&ID | Swarm + Monolith + Fusion |
| **Test 5a** | Test 4 + Predictive (2d) | Complex P&ID | Test 4 + Predictive |
| **Test 5b** | Test 4 + Polyline (2e) | Complex P&ID | Test 4 + Polyline |
| **Test 5c** | Test 4 + Self-Correction (3) | Complex P&ID | Test 4 + Self-Correction |

---

## Test 1: Baseline Phase 1 (Legenden-Erkennung)

**Ziel:** Testet nur Phase 1 (Pre-Analysis) - Legenden- und Metadaten-Extraktion

**Bild:** `page_1_original.png` (Complex P&ID mit Legende)

**Feature Flags:**
```python
{
    "use_swarm_analysis": False,
    "use_monolith_analysis": False,
    "use_fusion": False,
    "use_predictive_completion": False,
    "use_polyline_refinement": False,
    "use_self_correction_loop": False
}
```

**Erwartetes Ergebnis:**
- ✅ Legende extrahiert
- ✅ Metadaten extrahiert
- ❌ Keine Elemente/Verbindungen (Phase 2 übersprungen)
- F1-Score: 0.0000 (erwartet)

---

## Test 2: Baseline Simple P&ID (Monolith-All)

**Ziel:** Testet Monolith-Analyzer für einfache P&IDs

**Bild:** `Einfaches P&I.png` (Simple P&ID)

**Feature Flags:**
```python
{
    "use_swarm_analysis": False,
    "use_monolith_analysis": True,  # ✅ AKTIVIERT
    "use_fusion": False,
    "use_predictive_completion": False,
    "use_polyline_refinement": False,
    "use_self_correction_loop": False
}
```

**Erwartetes Ergebnis:**
- ✅ Monolith findet Elemente + Verbindungen
- ✅ F1-Score > 0.0 (Elemente sollten matchen)
- ✅ Konsistente Typ-Erkennung

---

## Test 3: Baseline Swarm-Only

**Ziel:** Testet Swarm-Analyzer isoliert

**Bild:** `Einfaches P&I.png` (Simple P&ID)

**Feature Flags:**
```python
{
    "use_swarm_analysis": True,  # ✅ AKTIVIERT
    "use_monolith_analysis": False,
    "use_fusion": False,
    "use_predictive_completion": False,
    "use_polyline_refinement": False,
    "use_self_correction_loop": False
}
```

**Erwartetes Ergebnis:**
- ✅ Swarm findet Elemente (keine Verbindungen)
- ✅ F1-Score > 0.0 (Elemente sollten matchen)
- ✅ Tiled Analysis funktioniert

---

## Test 4: Baseline Complex P&ID (Spezialisten-Kette)

**Ziel:** Testet vollständige Pipeline für komplexe P&IDs (Swarm + Monolith + Fusion)

**Bild:** `page_1_original.png` (Complex P&ID)

**Feature Flags:**
```python
{
    "use_swarm_analysis": True,  # ✅ AKTIVIERT
    "use_monolith_analysis": True,  # ✅ AKTIVIERT
    "use_fusion": True,  # ✅ AKTIVIERT
    "use_predictive_completion": False,
    "use_polyline_refinement": False,
    "use_self_correction_loop": False
}
```

**Erwartetes Ergebnis:**
- ✅ Swarm findet Elemente
- ✅ Monolith findet Elemente + Verbindungen
- ✅ Fusion kombiniert Ergebnisse
- ✅ F1-Score > 0.0

---

## Test 5a: Test 4 + Predictive (2d)

**Ziel:** Testet Test 4 + Predictive Completion (Phase 2d)

**Bild:** `page_1_original.png` (Complex P&ID)

**Feature Flags:**
```python
{
    "use_swarm_analysis": True,  # ✅ AKTIVIERT
    "use_monolith_analysis": True,  # ✅ AKTIVIERT
    "use_fusion": True,  # ✅ AKTIVIERT
    "use_predictive_completion": True,  # ✅ AKTIVIERT (Phase 2d)
    "use_polyline_refinement": False,
    "use_self_correction_loop": False
}
```

**Erwartetes Ergebnis:**
- ✅ Test 4 Ergebnisse
- ✅ Predictive Completion fügt fehlende Verbindungen hinzu
- ✅ Connection F1 sollte besser sein als Test 4

---

## Test 5b: Test 4 + Polyline (2e)

**Ziel:** Testet Test 4 + Polyline Refinement (Phase 2e)

**Bild:** `page_1_original.png` (Complex P&ID)

**Feature Flags:**
```python
{
    "use_swarm_analysis": True,  # ✅ AKTIVIERT
    "use_monolith_analysis": True,  # ✅ AKTIVIERT
    "use_fusion": True,  # ✅ AKTIVIERT
    "use_predictive_completion": False,
    "use_polyline_refinement": True,  # ✅ AKTIVIERT (Phase 2e)
    "use_self_correction_loop": False
}
```

**Erwartetes Ergebnis:**
- ✅ Test 4 Ergebnisse
- ✅ Polyline Refinement verbessert Verbindungen
- ✅ Connection F1 sollte besser sein als Test 4

---

## Test 5c: Test 4 + Self-Correction (3)

**Ziel:** Testet Test 4 + Self-Correction Loop (Phase 3)

**Bild:** `page_1_original.png` (Complex P&ID)

**Feature Flags:**
```python
{
    "use_swarm_analysis": True,  # ✅ AKTIVIERT
    "use_monolith_analysis": True,  # ✅ AKTIVIERT
    "use_fusion": True,  # ✅ AKTIVIERT
    "use_predictive_completion": False,
    "use_polyline_refinement": True,  # ✅ AKTIVIERT (für Test 4)
    "use_self_correction_loop": True,  # ✅ AKTIVIERT (Phase 3)
    "self_correction_min_quality_score": 90.0  # WICHTIG: Damit Phase 3 läuft
}
```

**Erwartetes Ergebnis:**
- ✅ Test 4 Ergebnisse
- ✅ Self-Correction Loop korrigiert Fehler
- ✅ F1-Score sollte besser sein als Test 4
- ✅ Weniger Halluzinationen

---

## Korrekte Test-Konfigurationen

### Test 1: Phase 1 Only
- **Alle Phasen deaktiviert außer Phase 1**
- **Erwartet:** Legende + Metadaten, keine Elemente

### Test 2: Monolith-All
- **Nur Monolith aktiv**
- **Erwartet:** Elemente + Verbindungen von Monolith

### Test 3: Swarm-Only
- **Nur Swarm aktiv**
- **Erwartet:** Elemente von Swarm (keine Verbindungen)

### Test 4: Complex P&ID (Baseline)
- **Swarm + Monolith + Fusion aktiv**
- **Erwartet:** Kombinierte Ergebnisse

### Test 5a: Test 4 + Predictive
- **Test 4 + `use_predictive_completion: True`**
- **Erwartet:** Test 4 + Predictive Completion

### Test 5b: Test 4 + Polyline
- **Test 4 + `use_polyline_refinement: True`**
- **Erwartet:** Test 4 + Polyline Refinement

### Test 5c: Test 4 + Self-Correction
- **Test 4 + `use_self_correction_loop: True`**
- **WICHTIG:** `self_correction_min_quality_score: 90.0` (damit Phase 3 läuft)
- **Erwartet:** Test 4 + Self-Correction Loop

---

## Bekannte Probleme

### Test 5a: Predictive Completion
- **Problem:** `use_predictive_completion: False` in test_metadata.md
- **Fix:** Sollte `True` sein (wird in `get_test_overrides` korrekt gesetzt)

### Test 5b: Polyline Refinement
- **Problem:** Phase 2d (Predictive) crashed → Test 5b erreicht Phase 2e nie
- **Fix:** Phase 2d Fehler beheben oder Test 5b sollte Predictive deaktivieren

### Test 5c: Self-Correction
- **Problem:** `use_self_correction_loop: False` in test_metadata.md
- **Problem:** Connection F1 sehr schlecht (0.043) → 56 halluzinierte Verbindungen
- **Fix:** Sollte `True` sein + `self_correction_min_quality_score: 90.0`

---

## Nächste Schritte

1. ✅ **Test-Konfigurationen korrigieren** (get_test_overrides prüfen)
2. ✅ **Test 5a:** `use_predictive_completion: True` sicherstellen
3. ✅ **Test 5b:** Phase 2d Fehler beheben oder Predictive deaktivieren
4. ✅ **Test 5c:** `use_self_correction_loop: True` + `self_correction_min_quality_score: 90.0`

