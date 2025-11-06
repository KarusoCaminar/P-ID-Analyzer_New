# Ideen & Learnings aus dem Entwicklungsprozess

**Sammel-Dokumentation aller Ideen, Erkenntnisse und Best Practices aus dem Chat-Verlauf.**

---

## 🎯 Kernziele

1. **100% korrekte Symbol-Erkennung** - Jedes Symbol muss korrekt erkannt werden
2. **100% korrekte Verbindungs-Erkennung** - Alle Pfade/Verbindungen müssen korrekt sein
3. **Iterative BBox-Verfeinerung** - Bounding Boxes müssen über Iterationen hinweg präziser werden
4. **Geschwindigkeit** - Schnelle Verarbeitung auch komplexer Uni-Bilder
5. **Minimale API-Nutzung** - Effiziente Nutzung der LLM-API

---

## 💡 Implementierte Ideen

### 1. Two-Pass Pipeline
**Idee:** Grober Durchlauf mit großen Tiles, dann Verfeinerung mit kleinen Tiles in unsicheren Bereichen.

**Status:** ✅ Implementiert in Swarm Analyzer

### 2. Confidence Map
**Idee:** Visuelle Darstellung der Unsicherheit, um gezielt nachzuarbeiten.

**Status:** ✅ Implementiert (Confidence Maps werden generiert)

### 3. Targeted Re-Analysis
**Idee:** Nur unsichere Zonen neu analysieren, nicht das ganze Bild.

**Status:** ✅ Implementiert in Self-Correction Loop

### 4. Viewshot Examples
**Idee:** Visuelle Beispiele von Symbolen aus echten Uni-Bildern in Prompts einbinden.

**Status:** ✅ Implementiert (Viewshot-Verzeichnis vorhanden)

### 5. Chain-of-Thought Reasoning
**Idee:** LLM soll Schritt-für-Schritt denken, nicht direkt antworten.

**Status:** ✅ Implementiert in Post-Processing (Phase 4.5)

### 6. Cascade BBox Regression
**Idee:** Iterative Verfeinerung von Bounding Boxes mit höheren IoU-Zielen.

**Status:** ✅ Implementiert (IoU-Bug behoben)

### 7. Skeleton-based Line Extraction
**Idee:** Computer Vision für präzise Linien-Extraktion, trennt Symbol-Linien von Pipeline-Linien.

**Status:** ✅ Implementiert (optional aktivierbar)

### 8. Legend-Symbol Matching
**Idee:** Symbole aus Legende mit erkannten Symbolen abgleichen.

**Status:** ✅ Implementiert (Legend Consistency Critic)

### 9. Multi-Model Critic
**Idee:** Mehrere LLM-Modelle für umfassende Validierung.

**Status:** ✅ Implementiert

### 10. Fusion Engine Strategy
**Idee:** Intelligente Zusammenführung von Swarm und Monolith mit IoU-Matching.

**Status:** ✅ Implementiert (IoU-Threshold: 0.3)

---

## 🔍 Erkenntnisse & Best Practices

### Error Amplification vermeiden

**Problem:** Pipeline verschlechtert gute Ergebnisse durch zu aggressive Korrekturen.

**Lösungen:**
1. **Hard Stop bei Degradation** - Stoppt Loop wenn Score 2x in Folge sinkt
2. **Early Stop bei gutem Score** - Stoppt bei Score >= 80% (konfigurierbar)
3. **Simple P&ID Mode** - Reduzierte Iterationen für einfache P&IDs
4. **Kritiker-Bypass** - Phase 3 kann deaktiviert werden

### Metacritic Kalibrierung

**Problem:** Metacritic flaggt korrekte Elemente fälschlicherweise als Halluzinationen.

**Lösung:**
- **IoU-basierte Evaluation** statt fixer Koordinaten-Unterschiede
- IoU < 0.3 = mögliche Halluzination
- IoU >= 0.3 = gleiches Element (auch wenn BBox-Größe unterschiedlich)
- BBox-Präzisions-Unterschiede sind KEINE Halluzinationen

### Active Learning

**Problem:** AL lernt aus degradierten Scores und verstärkt Fehler.

**Lösung:**
- `use_active_learning` Parameter (deaktiviert während Tests)
- Backup/Restore System für Learning-DB
- Reset-Script nach Code-Fixes

### Model-Strategien

**Erkenntnis:** Verschiedene Phasen benötigen verschiedene Modelle.

**Strategien:**
1. **Swarm:** Flash (schnell, viele Calls)
2. **Monolith:** Pro (komplex, weniger Calls)
3. **Detail:** Pro (hohe Qualität)
4. **Simple P&IDs:** Flash für alle (Geschwindigkeit)

### Fusion-Logik

**Erkenntnis:** IoU-Threshold ist kritisch für Fusion-Qualität.

**Aktuell:** IoU 0.3 (gut für Balance zwischen Precision und Recall)

**Test-Strategie:** Parameter-Sweep (IoU 0.1, 0.5, highest confidence wins)

---

## 🚫 Verworfen oder nicht implementiert

### 1. Whole-Image Re-Analysis Fallback
**Status:** ❌ Deaktiviert für Simple P&IDs (verursacht Overkill)

### 2. Extrem aggressive BBox Refinement
**Status:** ⚠️ Deaktiviert in Phase 4.7 (kann gute Daten verschlechtern)

### 3. ChatGPT/Anthropic Integration
**Status:** ❌ Nur deutsche Vertex-Modelle (2.5 Flash/Pro)

---

## 📚 Konzepte & Architektur-Ideen

### Pipeline-Phasen

1. **Phase 1: Pre-Analysis** - Metadata, Legend, Exclusion Zones
2. **Phase 2: Parallel Core Analysis**
   - 2a: Swarm Analysis (Tile-basiert)
   - 2b: Monolith Analysis (Quadrant-basiert)
   - 2c: Fusion Engine
   - 2d: Predictive Completion
   - 2e: Polyline Refinement
3. **Phase 3: Self-Correction Loop**
   - Kritiker-Identifikation
   - Targeted Re-Analysis
   - Iterative Verbesserung
4. **Phase 4: Post-Processing**
   - 4.1: Type Validation
   - 4.2: Confidence Filtering
   - 4.3: Graph Validation
   - 4.4: Connection Completion
   - 4.5: CoT Reasoning
   - 4.6: ID Correction
   - 4.7: CV BBox Refinement (optional)

### Kritiker-System

1. **Metacritic** - Cross-Validation Monolith vs. Swarm
2. **Topology Critic** - Graph-Konsistenz
3. **Legend Consistency Critic** - Legenden-Symbol-Konsistenz
4. **Multi-Model Critic** - Umfassende Validierung

### Learning-System

1. **Symbol Library** - Persistente Symbol-Datenbank
2. **Active Learning** - Lernen aus erfolgreichen Analysen
3. **Pretraining** - Vor-Training mit Symbol-Sammlung
4. **Confidence Calibration** - Anpassung basierend auf historischer Performance

---

## 🎓 Lessons Learned

### 1. Teste ohne Phase 3/4 zuerst
- Ermittle Basis-Qualität von Phase 2
- Identifiziere wo Fehler entstehen
- Verhindere Error Amplification

### 2. IoU statt fixer Koordinaten
- Skalierbar für verschiedene Bildgrößen
- Berücksichtigt BBox-Präzisions-Unterschiede
- Reduziert False Positives

### 3. Simple P&ID Mode
- Reduzierte Iterationen für einfache Diagramme
- Early Stop bei gutem Score
- Verhindert Overkill

### 4. Model-Strategien
- Verschiedene Modelle für verschiedene Phasen
- Flash für schnelle Tasks
- Pro für komplexe Tasks

### 5. Active Learning während Tests deaktivieren
- Verhindert Lernen aus schlechten Korrekturen
- Ermöglicht saubere Tests
- Reset nach Code-Fixes

---

## 🔮 Zukünftige Ideen

### 1. Adaptive Tile Strategy
**Idee:** Dynamische Anpassung der Tile-Größe basierend auf Symbol-Dichte.

**Status:** ⏳ Noch nicht implementiert

### 2. Multi-Scale Analysis
**Idee:** Analyse auf verschiedenen Skalierungs-Ebenen (groß → klein).

**Status:** ⏳ Noch nicht implementiert

### 3. Context-Aware Type Inference
**Idee:** Type-Inferenz basierend auf Kontext (Labels, Position, Nachbarn).

**Status:** ⚠️ Teilweise implementiert

### 4. Error Explanation by LLM
**Idee:** LLM erklärt Fehlerursachen und schlägt Fixes vor.

**Status:** ⏳ Noch nicht implementiert

### 5. Automated Test Suite
**Idee:** Automatische Test-Suite mit verschiedenen P&ID-Typen.

**Status:** ⚠️ Teilweise implementiert (Phase 1 Tests)

---

**Letzte Aktualisierung:** 2025-11-04 20:30

