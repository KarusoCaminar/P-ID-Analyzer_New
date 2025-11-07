# 🔧 Parameter Tuning - Problem & Fix

**Datum:** 2025-11-07  
**Status:** ✅ Behoben

---

## 🐛 Problem-Analyse

### Was passiert ist:

1. **Circuit Breaker aktiviert** (nach 5 API-Fehlern):
   ```
   Circuit breaker is open. Skipping API call to minimize failures.
   ```

2. **Response Validation Fehler**:
   ```
   LLM response is not a dictionary (type: list) - validation failed
   RESPONSE_VALIDATION_FAILED: Response structure invalid
   ```

3. **Self-Correction Endlosschleife**:
   - `simple_whole_image` Strategie hat `use_self_correction_loop: true`
   - 5 Iterationen mit Swarm-Analyse pro Test
   - Jede Iteration = viele API-Calls
   - Bei Fehlern → Circuit Breaker öffnet → Alle Calls blockiert
   - Pipeline versucht es trotzdem weiter → Endlosschleife

4. **Ergebnis**:
   - Test 1 dauerte 56 Minuten (statt ~5-10 Minuten)
   - Circuit Breaker blockiert alle weiteren Tests
   - 149 Tests würden noch viele Stunden dauern

---

## ✅ Fixes Implementiert

### 1. **Self-Correction für Parameter-Tuning deaktiviert**

**Grund**: Self-Correction ist für Parameter-Tuning nicht nötig. Wir testen nur die `adaptive_threshold` Parameter, nicht die gesamte Pipeline-Qualität.

**Änderung**:
```python
params_override = {
    **strategy_config,
    'use_self_correction_loop': False,  # CRITICAL: Disable self-correction
    'use_polyline_refinement': False,   # Disable polyline refinement
    'use_predictive_completion': False, # Disable predictive completion
    'use_visual_feedback': False        # Disable visual feedback
}
```

**Ersparnis**: ~50 Minuten pro Test → ~5-10 Minuten pro Test

### 2. **Parameter-Range reduziert**

**Grund**: Schnelleres Finden des optimalen Bereichs.

**Änderung**:
- `ADAPTIVE_THRESHOLD_FACTORS`: [0.01, 0.02, 0.03, 0.05] (4 Werte statt 6)
- `ADAPTIVE_THRESHOLD_MINS`: [20, 25, 30] (3 Werte statt 5)
- `ADAPTIVE_THRESHOLD_MAXS`: [125, 150, 200] (3 Werte statt 5)

**Ergebnis**: 36 Kombinationen statt 150 (75% weniger Tests)

**Geschätzte Zeit**: 36 × 5-10 Minuten = 3-6 Stunden (statt 12-25 Stunden)

---

## 📊 Erwartete Verbesserungen

### Vorher:
- **150 Tests** × **56 Minuten** = **~140 Stunden** (mit Circuit Breaker Problemen)
- Circuit Breaker blockiert nach wenigen Tests
- Endlosschleifen bei Self-Correction

### Nachher:
- **36 Tests** × **5-10 Minuten** = **3-6 Stunden**
- Keine Circuit Breaker Probleme (weniger API-Calls)
- Keine Endlosschleifen (Self-Correction deaktiviert)

---

## 🎯 Nächste Schritte

1. ✅ Fixes implementiert
2. ⏳ Neuen Parameter-Tuning-Lauf starten
3. ⏳ Ergebnisse analysieren (Top 5 Parameter-Kombinationen)
4. ⏳ Beste Parameter in `config.yaml` eintragen
5. ⏳ Validierung auf komplexem Bild

---

## 💡 Tipp für zukünftige Parameter-Tuning-Läufe

- **Immer Self-Correction deaktivieren** für schnelle Parameter-Tests
- **Kleine Parameter-Ranges** testen, dann verfeinern
- **Circuit Breaker Threshold erhöhen** für Parameter-Tuning (optional)
- **Nur relevante Phasen aktivieren** (für Threshold-Tuning: nur CV Line Extraction)

