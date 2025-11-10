# Test Analysis Summary - 2025-11-08

## Übersicht

**Neuester Test:** simple_whole_image (2025-11-08 02:03:07)
**Dauer:** 32.08 Minuten
**Ergebnis:** **VERSCHLECHTERUNG** gegenüber vorherigem Test

## Kritische Probleme

### 1. Phase 3 Verschlechtert Ergebnisse

- **Phase 3 lief:** 5 Iterationen durchgeführt
- **Best Score in Phase 3:** 53.94
- **Final Score:** 30.12 (SCHLECHTER als vor Phase 3: 45.95)
- **Problem:** Phase 3 hat die Ergebnisse verschlechtert, nicht verbessert

### 2. Massive Rate Limit Probleme

- **429 Errors:** 288x
- **Circuit Breaker Opens:** 179x
- **Timeouts:** 212x
- **Response Rate:** nur 55.6% (viele Requests scheitern)
- **Ursache:** Phase 3 startete Swarm-Analyse (vor dem Fix), obwohl `use_swarm_analysis: false`

### 3. ID-Mismatches (Connection F1 = 0.0)

**Analysis Connections:**
- CHP1 -> PU321
- PU455 -> Ventilation
- PU306 -> SV306
- V-RL -> SV306

**Ground Truth Connections:**
- CHP1 -> MV3121A
- MV3121A -> PU3121
- MV3121B -> PU3101
- CHP2 -> MV3131A

**Problem:** Komplett unterschiedliche IDs! LLM generiert generische IDs statt echter IDs aus Bild.

### 4. Zu Wenige Elemente Gefunden

- **Gefunden:** 9 Elemente
- **Ground Truth:** 24 Elemente
- **Fehlend:** 19 Elemente (79% fehlen!)
- **Ursache:** Monolith allein findet zu wenige Elemente auf komplexen Bildern

### 5. Performance Problem

- **Duration:** 32.08 Minuten (vs 2.48 min bei hybrid_fusion)
- **Ursache:** Viele Rate Limits -> viele Retries -> lange Wartezeiten

## Vergleich: Alle Tests

| Test | Strategy | Quality Score | Element F1 | Connection F1 | Duration |
|------|----------|---------------|------------|---------------|----------|
| #1 (neueste) | simple_whole_image | 30.12 | 0.303 | 0.0 | 32.08 min |
| #2 | hybrid_fusion | 45.95 | 0.649 | 0.0 | 2.48 min |
| #3 | hybrid_fusion | 45.95 | 0.649 | 0.0 | 3.28 min |
| #4 | hybrid_fusion | 46.3 | 0.658 | 0.0 | 6.88 min |
| #5 (beste) | hybrid_fusion | 52.87 | 0.727 | 0.043 | 7.86 min |

**Bester Test:** #5 (hybrid_fusion, 52.87 Quality Score, 0.043 Connection F1)

## Fazit

### Probleme Identifiziert

1. **Phase 3 verschlechtert Ergebnisse** wenn zu viele Rate Limits auftreten
2. **simple_whole_image ist NICHT besser** als hybrid_fusion für komplexe Bilder
3. **ID-Mismatches bleiben das Hauptproblem** (Connection F1 = 0.0)
4. **Monolith allein findet zu wenige Elemente** auf komplexen Bildern

### Empfehlungen

1. **Verwende hybrid_fusion** (nicht simple_whole_image) für komplexe Bilder
2. **Phase 3 Fix:** Respektiere `use_swarm_analysis` Flag (bereits implementiert)
3. **ID-Mismatches Fix:** LLM muss echte IDs aus Bild extrahieren, nicht generische IDs generieren
4. **Prompts verbessern:** Explizit anweisen, echte IDs aus Bild-Text zu extrahieren

### Nächste Schritte

1. ✅ Fix: Phase 3 respektiert jetzt `use_swarm_analysis` Flag
2. ⏳ Fix: ID-Mismatches (LLM generiert echte IDs aus Bild)
3. ⏳ Test: hybrid_fusion mit Phase 3 (ohne Swarm in Phase 3)
4. ⏳ Optimierung: Prompts für ID-Extraktion verbessern

## Metriken Detail

### Neuester Test (simple_whole_image)

- **Quality Score:** 30.12
- **Element F1:** 0.303
- **Element Precision:** 0.556
- **Element Recall:** 0.208 (nur 20.8% der Elemente gefunden!)
- **Connection F1:** 0.0
- **Connection Precision:** 0.0
- **Connection Recall:** 0.0
- **Elements Found:** 9
- **Ground Truth Elements:** 24
- **Missed Elements:** 19
- **Hallucinated Elements:** 4
- **Connections Found:** 11
- **Ground Truth Connections:** 35
- **Missed Connections:** 35
- **Hallucinated Connections:** 10

### Phase 3 Status

- **Phase 3 lief:** ✅ Ja (5 Iterationen)
- **Best Score:** 53.94 (in Phase 3)
- **Final Score:** 30.12 (nach Phase 3 - VERSCHLECHTERUNG!)
- **Problem:** Rate Limits haben Phase 3 beeinträchtigt

### Error Statistics

- **429 Errors:** 288x
- **Circuit Breaker Opens:** 179x
- **Timeouts:** 212x
- **API Requests:** 279
- **API Responses:** 155
- **Response Rate:** 55.6%

