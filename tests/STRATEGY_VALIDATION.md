# 🔬 Teststrategie: "Pipeline Isolation & Integration"

## Übersicht

Das Ziel dieser Teststrategie ist es, die Performance jeder Komponente (Phase) isoliert zu messen, bevor wir sie kombinieren. Jeder Testlauf sollte (falls möglich) gegen die "Ground Truth"-Daten von "Einfaches P&I" validiert werden, um `element_f1` und `connection_f1` zu erhalten.

## Test-Harness

**Skript:** `scripts/run_strategy_validation.py`

**Verwendung:**
```bash
# Einzelnen Test ausführen
python scripts/run_strategy_validation.py --test "Test 2"

# Alle Tests ausführen
python scripts/run_strategy_validation.py --test all

# Mit eigenem Bild und Ground Truth
python scripts/run_strategy_validation.py --test "Test 4" --image "data/input/Complex P&I.png" --ground-truth "data/ground_truth/Complex P&I.json"
```

## 🧪 Phase 1: Baseline-Tests (Kern-System)

### Test 1: Baseline Phase 1 (Legenden-Erkennung)

**Ziel:** Stabilität und Korrektheit der Legenden-Erkennung prüfen.

**Aktion:** Pipeline so konfigurieren, dass nur Phase 1 (Pre-Analysis) läuft.

**Deaktivieren:** Alle Phasen ab Phase 2.

**Datensammlung:**
- Prüfe die `legend_info.json` (für "Einfaches P&I" sollte sie leer sein, da keine Legende vorhanden ist).
- Prüfe mit einem Diagramm, das eine Legende hat. Wird die `symbol_map` korrekt befüllt?

**Erfolgskriterium:** Phase 1 läuft stabil und extrahiert Legenden-Daten korrekt, ohne bei fehlenden Legenden abzustürzen.

**Konfiguration:**
```python
{
    "use_swarm_analysis": False,
    "use_monolith_analysis": False,
    "use_fusion": False,
    "use_predictive_completion": False,
    "use_polyline_refinement": False,
    "use_self_correction_loop": False,
    "use_post_processing": True  # Für KPIs
}
```

---

### Test 2: Baseline "Simple P&ID" (Monolith "Alles-Finder")

**Ziel:** Die Performance des "guten Laufs" (Monolith findet Elemente + Verbindungen) reproduzieren. Dies ist die Strategie für einfache Diagramme.

**Aktion:** Pipeline-Logik so einstellen, dass sie dieser Kette folgt:
- Phase 1 → Phase 2c (Monolith) → Phase 4 (Post-Processing)

**Wichtige Konfiguration:**
- Monolith (Phase 2c): Findet Elemente UND Verbindungen (wie im alten ...205421-Lauf).

**Deaktivieren:** Swarm (2a), Guard Rails (2b), Fusion (2c-Fusion), Predictive (2d), Polyline (2e), Self-Correction (3).

**Datensammlung:**
- Prüfe `results.json`: Wie hoch sind `element_f1` und `connection_f1`?
- Ist das Ergebnis sauber und frei von Halluzinationen (z.B. keine FT 11 -> FT 10-Verbindung)?

**Erfolgskriterium:** F1-Scores sind hoch. Das Ergebnis ist dem "guten Lauf" (...205421) ebenbürtig.

**Konfiguration:**
```python
{
    "use_swarm_analysis": False,
    "use_monolith_analysis": True,  # Monolith findet alles
    "use_fusion": False,
    "use_predictive_completion": False,
    "use_polyline_refinement": False,
    "use_self_correction_loop": False,
    "use_post_processing": True
}
```

---

### Test 3: Baseline Phase 2a (Swarm "Elemente-Finder")

**Ziel:** Die reine Performance des Swarm (Flash-Modell) bei der Element-Erkennung messen.

**Aktion:** Pipeline-Logik auf diese Kette setzen:
- Phase 1 → Phase 2a (Swarm) → Phase 4 (Post-Processing)

**Wichtige Konfiguration:**
- Deaktivieren: Guard Rails (2b), Monolith (2c), Fusion (2c-Fusion), 2d, 2e, 3.

**Datensammlung:**
- Prüfe `results.json`: Wie hoch ist `element_f1`?
- Ist das `connections`-Array wie erwartet leer (oder fast leer)?

**Erfolgskriterium:** `element_f1` ist hoch. Der Swarm liefert schnell und präzise nur Elemente.

**Konfiguration:**
```python
{
    "use_swarm_analysis": True,  # Swarm findet Elemente
    "use_monolith_analysis": False,
    "use_fusion": False,
    "use_predictive_completion": False,
    "use_polyline_refinement": False,
    "use_self_correction_loop": False,
    "use_post_processing": True
}
```

---

### Test 4: Baseline "Complex P&ID" (Spezialisten-Kette)

**Ziel:** Die Performance der neuen Kern-Architektur (Swarm → GR → Monolith) ohne die fehlerhaften "Helfer"-Phasen messen. Dies ist die Strategie für komplexe Diagramme.

**Aktion:** Pipeline-Logik auf die designierte Kette setzen:
- Phase 1 → 2a (Swarm) → 2b (Guard Rails) → 2c (Monolith "Connect-Only") → 2c (Fusion) → Phase 4

**Wichtige Konfiguration:**
- Monolith (Phase 2c): Muss den Prompt für "Finde nur Verbindungen basierend auf dieser JSON-Liste" verwenden.
- Deaktivieren: Predictive (2d), Polyline (2e), Self-Correction (3).

**Datensammlung (Sehr wichtig):**
- Prüfe `pipeline.log`: Hat der Monolith (2c) die Element-Liste von 2b korrekt erhalten?
- Prüfe `llm_calls.log`: Was hat der Monolith (2c) tatsächlich geantwortet? (Im ...092155-Lauf waren es nur 2 Verbindungen).
- Prüfe `results.json`: Wie hoch ist `connection_f1`? Enthält es Halluzinationen (z.B. FT 11 -> FT 10)? Es sollte nicht, da 2d/2e deaktiviert sind.

**Erfolgskriterium:** Das Ergebnis ist sauber (keine Halluzinationen). Es darf unvollständig sein (niedriger `connection_f1`), aber es darf keinen Müll enthalten.

**Konfiguration:**
```python
{
    "use_swarm_analysis": True,  # Swarm findet Elemente
    "use_monolith_analysis": True,  # Monolith findet Verbindungen
    "use_fusion": True,  # Fusion kombiniert Ergebnisse
    "use_predictive_completion": False,
    "use_polyline_refinement": False,
    "use_self_correction_loop": False,
    "use_post_processing": True
}
```

---

## 🛠️ Phase 2: Debugging der "Helfer"-Phasen (Basierend auf Test 4)

Erst wenn Test 4 eine saubere, aber unvollständige Basis liefert, können die "Helfer"-Phasen sinnvoll getestet werden, um zu sehen, ob sie helfen oder schaden.

### Test 5a: Isoliere Phase 2d (Predictive Completion)

**Aktion:** Führe die Kette aus Test 4 aus, aber schalte nur Phase 2d (Predictive) hinzu.

**Datensammlung:** Prüfe `results.json`. Hat sich der F1-Score verbessert (weil fehlende Verbindungen ergänzt wurden) oder verschlechtert (weil Halluzinationen wie FT 11 -> FT 10 hinzugefügt wurden)?

**Ziel:** Kausalen Beweis finden, ob Phase 2d Rauschen hinzufügt.

**Konfiguration:**
```python
{
    "use_swarm_analysis": True,
    "use_monolith_analysis": True,
    "use_fusion": True,
    "use_predictive_completion": True,  # NUR diese Phase zusätzlich
    "use_polyline_refinement": False,
    "use_self_correction_loop": False,
    "use_post_processing": True
}
```

---

### Test 5b: Isoliere Phase 2e (Polyline Refinement)

**Aktion:** Führe die Kette aus Test 4 aus, aber schalte nur Phase 2e (Polyline) hinzu.

**Datensammlung:** Prüfe `results.json`. Hat diese CV-basierte Phase Verbindungen hinzugefügt oder entfernt? Hat sie den F1-Score verändert?

**Ziel:** Kausalen Beweis finden, ob Phase 2e Rauschen hinzufügt.

**Konfiguration:**
```python
{
    "use_swarm_analysis": True,
    "use_monolith_analysis": True,
    "use_fusion": True,
    "use_predictive_completion": False,
    "use_polyline_refinement": True,  # NUR diese Phase zusätzlich
    "use_self_correction_loop": False,
    "use_post_processing": True
}
```

---

### Test 5c: Isoliere Phase 3 (Self-Correction)

**Aktion:** Führe die Kette aus Test 4 aus, aber schalte nur Phase 3 (Self-Correction) hinzu.

**Datensammlung:**
- Prüfe `pipeline.log`: Läuft der Loop überhaupt? (Im ...092155-Lauf stoppte er sofort wegen Quality Score (68.17) >= Min Score (60.0)).
- **Fix:** Der Min Score (in der `config.yaml`) für den Early stop muss auf einen viel höheren Wert (z.B. 90.0) gesetzt werden, sonst wird er nie laufen.

**Ziel:** Die Konfiguration von Phase 3 reparieren, damit sie bei echten Problemen überhaupt anspringt.

**Konfiguration:**
```python
{
    "use_swarm_analysis": True,
    "use_monolith_analysis": True,
    "use_fusion": True,
    "use_predictive_completion": False,
    "use_polyline_refinement": False,
    "use_self_correction_loop": True,  # NUR diese Phase zusätzlich
    "self_correction_min_quality_score": 90.0,  # WICHTIGER FIX: Min Score erhöhen
    "use_post_processing": True
}
```

---

## 📊 Ergebnis dieser Teststrategie

Nach diesen 5 Test-Kategorien haben Sie eine exakte Datenlage:

1. **Test 2** zeigt die (wahrscheinlich hohe) Baseline für einfache P&IDs.
2. **Test 4** zeigt die (wahrscheinlich unvollständige, aber saubere) Baseline für komplexe P&IDs.
3. **Test 5a/5b** beweist, welche der "Helfer"-Phasen die Halluzinationen (FT 11 -> FT 10) erzeugt hat, die den F1-Score im letzten Lauf zerstört haben.
4. **Test 5c** zeigt, wie Phase 3 konfiguriert werden muss, damit sie funktioniert.

Mit diesen Daten können Sie dann die Phase 0 (Complexity Analysis) intelligent einstellen, um je nach Diagramm zwischen Strategie (Test 2) und (Test 4 + reparierte Helfer) zu wechseln.

---

## 📁 Ausgabe-Struktur

Alle Test-Ergebnisse werden in `outputs/strategy_validation/` gespeichert:

```
outputs/strategy_validation/
├── Test_1_Baseline_Phase_1/
│   ├── results.json
│   ├── pipeline.log
│   └── ...
├── Test_2_Baseline_Simple_PID/
│   └── ...
├── Test_4_Baseline_Complex_PID/
│   └── ...
├── summary_20250101_120000.json  # Finale Zusammenfassung
└── ...
```

---

## 🔍 Analyse der Ergebnisse

### Vergleichstabelle

Nach Ausführung aller Tests können Sie eine Vergleichstabelle erstellen:

| Test | Element F1 | Connection F1 | Element Precision | Connection Precision | Bemerkungen |
|------|------------|---------------|-------------------|----------------------|-------------|
| Test 2 (Monolith-All) | X.XX | X.XX | X.XX | X.XX | Sauber, hohe Scores |
| Test 4 (Spezialisten-Kette) | X.XX | X.XX | X.XX | X.XX | Sauber, aber unvollständig |
| Test 5a (+ Predictive) | X.XX | X.XX | X.XX | X.XX | Halluzinationen? |
| Test 5b (+ Polyline) | X.XX | X.XX | X.XX | X.XX | Verbesserung? |
| Test 5c (+ Self-Correction) | X.XX | X.XX | X.XX | X.XX | Läuft überhaupt? |

### Entscheidungskriterien

- **Wenn Test 5a F1 verschlechtert:** Predictive Completion deaktivieren oder Parameter anpassen.
- **Wenn Test 5b F1 verbessert:** Polyline Refinement behalten.
- **Wenn Test 5c nicht läuft:** `self_correction_min_quality_score` anpassen.

---

## ⚠️ Wichtige Hinweise

1. **Teuer:** Diese Tests führen echte LLM-Aufrufe durch und können Minuten dauern.
2. **Ground Truth erforderlich:** Für aussagekräftige F1-Scores benötigen Sie Ground Truth-Daten.
3. **Isolation:** Jeder Test isoliert eine Komponente, um kausale Zusammenhänge zu identifizieren.
4. **Reproduzierbarkeit:** Alle Konfigurationen werden in `params_override` gespeichert und sind nachvollziehbar.

---

## 🚀 Nächste Schritte

Nach erfolgreicher Durchführung aller Tests:

1. **Analyse:** Identifizieren Sie die problematischen Phasen (5a/5b/5c).
2. **Reparatur:** Passen Sie die Parameter der problematischen Phasen an.
3. **Validierung:** Führen Sie die Tests erneut aus, um Verbesserungen zu bestätigen.
4. **Integration:** Integrieren Sie die optimierten Konfigurationen in die Haupt-Pipeline.

