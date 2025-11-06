# Pipeline-Prozess: Detaillierte Erklärung

**Datum:** 2025-11-06  
**Status:** ✅ Aktuell

## 🎯 Übersicht

Diese Dokumentation erklärt exakt den Pipeline-Prozess und wie nach jeder Schleife die Informationen gesichert, weitergegeben oder geprüft werden.

## 📊 Pipeline-Sequenz (Detailliert)

### **Phase 0: Complexity Analysis (CV-basiert)**

**Zweck:** Schnelle Komplexitätserkennung für Strategie-Auswahl

**Prozess:**
1. **Input:** Bild-Pfad
2. **Analyse:** CV-basierte Multi-Metrik-Analyse (Edge Density, Object Density, Junctions)
3. **Output:** Strategie-Name (`simple_pid_strategy` oder `optimal_swarm_monolith`)
4. **Speicherung:** Strategie wird in `self.model_strategy` gespeichert
5. **Weitergabe:** Strategie wird für alle nachfolgenden Phasen verwendet

**Informationen:**
- **Gespeichert:** `self.model_strategy` (Strategie-Konfiguration)
- **Weitergabe:** An alle nachfolgenden Phasen
- **Geprüft:** Strategie-Validierung

---

### **Phase 1: Pre-Analysis (Legend Extraction)**

**Zweck:** Legenden-Erkennung und -Extraktion

**Prozess:**
1. **Input:** Bild-Pfad
2. **CV-First:** `find_legend_rectangle_cv()` findet schwarzes Rechteck
3. **Crop:** Bild wird auf Legenden-Bereich zugeschnitten
4. **LLM-Call:** Legenden-Extraktion mit spezialisiertem Prompt
5. **Output:** Legend-Context (symbol_map, line_map)
6. **Speicherung:** `self._global_knowledge_repo['symbol_map']`, `self._global_knowledge_repo['line_map']`
7. **Weitergabe:** Legend-Context wird an Swarm und Monolith übergeben

**Informationen:**
- **Gespeichert:** `self._global_knowledge_repo` (Legend-Context)
- **Weitergabe:** An Swarm-Analyzer und Monolith-Analyzer
- **Geprüft:** Legend-Validierung

---

### **Phase 2: Sequential Core Analysis**

#### **Phase 2a: Swarm Analysis (Element-Erkennung)**

**Zweck:** Element-Erkennung (Spezialist, ignoriert Verbindungen)

**Prozess:**
1. **Input:** Bild-Pfad, Legend-Context
2. **Analyse:** Swarm-Analyzer findet alle Elemente (Symbols, Text Labels)
3. **Output:** `{"elements": [...], "connections": []}`
4. **Speicherung:** `swarm_result` (temporär)
5. **Weitergabe:** Element-Liste wird an Guard Rails übergeben

**Informationen:**
- **Gespeichert:** `swarm_result` (Element-Liste)
- **Weitergabe:** An Guard Rails (Phase 2b)
- **Geprüft:** Element-Validierung (Koordinaten, Typen)

#### **Phase 2b: Guard Rails (Inference Rules)**

**Zweck:** Bereinigung und Anreicherung der Element-Liste

**Prozess:**
1. **Input:** Swarm-Element-Liste
2. **Bereinigung:** 
   - SamplePoint-S: `id == 'S'` → `type = 'Sample Point'`
   - ISA-Supply: `'isa' in id/label` → `type = 'Source'`
   - Confidence-Boost für alle Elemente
3. **Output:** Bereinigte Element-Liste
4. **Speicherung:** `swarm_graph["elements"]` (überschreibt Swarm-Ergebnis)
5. **Weitergabe:** Bereinigte Element-Liste wird an Monolith übergeben

**Informationen:**
- **Gespeichert:** `swarm_graph["elements"]` (bereinigte Element-Liste)
- **Weitergabe:** An Monolith-Analyzer (Phase 2c)
- **Geprüft:** Guard-Rails-Validierung

#### **Phase 2c: Monolith Analysis (Verbindungs-Erkennung)**

**Zweck:** Verbindungs-Erkennung (Spezialist, nutzt Element-Liste als Input)

**Prozess:**
1. **Input:** Bild-Pfad, Bereinigte Element-Liste (als JSON), Legend-Context
2. **Element-Liste-Vorbereitung:**
   - BBox-Serialisierung (Pydantic-Modelle → Dict)
   - Confidence hinzufügen
   - JSON-Serialisierung
3. **Analyse:** Monolith-Analyzer findet alle Verbindungen zwischen Elementen
4. **Output:** `{"elements": [], "connections": [...]}`
5. **Speicherung:** `monolith_result` (temporär)
6. **Weitergabe:** Verbindungs-Liste wird an Fusion übergeben

**Informationen:**
- **Gespeichert:** `monolith_result` (Verbindungs-Liste)
- **Weitergabe:** An Fusion (Phase 2c)
- **Geprüft:** Verbindungs-Validierung (IDs, BBox-Matching)

#### **Phase 2c: Fusion (Montage)**

**Zweck:** Kombination von Swarm-Elementen und Monolith-Verbindungen

**Prozess:**
1. **Input:** Swarm-Element-Liste, Monolith-Verbindungs-Liste
2. **Montage:**
   - Elemente: Von Swarm (bereits bereinigt)
   - Verbindungen: Von Monolith (Spezialist)
3. **Output:** `{"elements": [...], "connections": [...]}`
4. **Speicherung:** `self._analysis_results` (Haupt-Ergebnis)
5. **Weitergabe:** An Phase 2d, 2e, 3, 4

**Informationen:**
- **Gespeichert:** `self._analysis_results` (Haupt-Ergebnis)
- **Weitergabe:** An alle nachfolgenden Phasen
- **Geprüft:** Graph-Struktur-Validierung

---

### **Phase 2d: Predictive Completion**

**Zweck:** Vorhersagende Vervollständigung fehlender Verbindungen

**Prozess:**
1. **Input:** `self._analysis_results`
2. **Analyse:** Vorhersagende Vervollständigung basierend auf Topologie
3. **Output:** Erweiterte Verbindungs-Liste
4. **Speicherung:** `self._analysis_results` (überschreibt)
5. **Weitergabe:** An Phase 2e, 3, 4

**Informationen:**
- **Gespeichert:** `self._analysis_results` (erweitert)
- **Weitergabe:** An Phase 2e, 3, 4
- **Geprüft:** Topologie-Validierung

---

### **Phase 2e: Polyline Refinement**

**Zweck:** CV-basierte Polylinien-Verfeinerung

**Prozess:**
1. **Input:** `self._analysis_results`, Bild-Pfad
2. **CV-Analyse:**
   - Text-Removal (OCR + CV)
   - Skeletonization
   - Adaptive Thresholds
   - Gap-Bridging
3. **Output:** Verfeinerte Polylinien
4. **Speicherung:** `self._analysis_results["connections"]` (Polylinien aktualisiert)
5. **Weitergabe:** An Phase 3, 4

**Informationen:**
- **Gespeichert:** `self._analysis_results["connections"]` (Polylinien)
- **Weitergabe:** An Phase 3, 4
- **Geprüft:** Polyline-Validierung

---

### **Phase 3: Self-Correction Loop**

**Zweck:** Iterative Selbstkorrektur basierend auf Fehleranalyse

**Prozess:**
1. **Input:** `self._analysis_results`, Bild-Pfad, Truth-Data (optional)
2. **Iteration 1-N:**
   - **Fehleranalyse:** Berechnung von KPIs, Fehler-Identifikation
   - **Uncertainty-Zones:** Identifikation unsicherer Bereiche
   - **Re-Analyse:** Gezielte Re-Analyse unsicherer Bereiche
   - **Merge:** Zusammenführung neuer Ergebnisse mit bestehenden
   - **Score-Berechnung:** Quality-Score-Berechnung
   - **Early-Stop:** Bei einfachen P&IDs (Score > 85.0) oder max Iterationen
3. **Output:** Bestes Ergebnis (`best_result`)
4. **Speicherung:** 
   - `best_result` (temporär)
   - `self._analysis_results` (final)
   - Score-History in `best_result["final_ai_data"]["score_history"]`
5. **Weitergabe:** An Phase 4

**Informationen:**
- **Gespeichert:** 
  - `best_result` (beste Iteration)
  - `self._analysis_results` (final)
  - Score-History
- **Weitergabe:** An Phase 4
- **Geprüft:** 
  - KPI-Validierung
  - Fehler-Validierung
  - Score-Validierung

**Nach jeder Iteration:**
- **Gespeichert:** Score, Elemente, Verbindungen
- **Weitergabe:** An nächste Iteration oder Phase 4
- **Geprüft:** Score-Verbesserung, Fehler-Reduktion

---

### **Phase 4: Post-Processing**

**Zweck:** Finale Nachbearbeitung und Validierung

**Prozess:**
1. **Input:** `best_result`, Bild-Pfad, Truth-Data (optional)
2. **Post-Processing:**
   - Chain-of-Thought Reasoning (Splits/Merges, Missing Elements)
   - Confidence-Filterung
   - Graph-Validierung
   - KPI-Berechnung
3. **Output:** Finales `AnalysisResult`
4. **Speicherung:**
   - `final_result` (AnalysisResult)
   - JSON-Dateien (results.json, kpis.json, cgm_data.json)
   - Visualisierungen (debug_map.png, confidence_map.png, etc.)
   - HTML-Report (report.html)
5. **Weitergabe:** An Benutzer (Return-Wert)

**Informationen:**
- **Gespeichert:**
  - `final_result` (AnalysisResult)
  - JSON-Dateien
  - Visualisierungen
  - HTML-Report
- **Weitergabe:** An Benutzer
- **Geprüft:**
  - Finale KPI-Validierung
  - Graph-Validierung
  - Visualisierungs-Validierung

---

## 🔄 Informationsfluss (Nach jeder Schleife)

### **1. Gespeicherte Informationen:**

#### **Nach Phase 0:**
- `self.model_strategy` - Strategie-Konfiguration

#### **Nach Phase 1:**
- `self._global_knowledge_repo['symbol_map']` - Symbol-Map
- `self._global_knowledge_repo['line_map']` - Line-Map

#### **Nach Phase 2a (Swarm):**
- `swarm_result` - Element-Liste (temporär)

#### **Nach Phase 2b (Guard Rails):**
- `swarm_graph["elements"]` - Bereinigte Element-Liste

#### **Nach Phase 2c (Monolith):**
- `monolith_result` - Verbindungs-Liste (temporär)

#### **Nach Phase 2c (Fusion):**
- `self._analysis_results` - Haupt-Ergebnis (Elemente + Verbindungen)

#### **Nach Phase 2d:**
- `self._analysis_results` - Erweiterte Verbindungs-Liste

#### **Nach Phase 2e:**
- `self._analysis_results["connections"]` - Verfeinerte Polylinien

#### **Nach Phase 3 (jede Iteration):**
- `best_result` - Bestes Ergebnis (temporär)
- `self._analysis_results` - Finales Ergebnis
- `score_history` - Score-Historie

#### **Nach Phase 4:**
- `final_result` - Finales AnalysisResult
- JSON-Dateien (results.json, kpis.json, cgm_data.json)
- Visualisierungen (debug_map.png, confidence_map.png, kpi_dashboard.png, score_curve.png)
- HTML-Report (report.html)

### **2. Weitergegebene Informationen:**

#### **Phase 0 → Phase 1-4:**
- Strategie-Konfiguration (`self.model_strategy`)

#### **Phase 1 → Phase 2:**
- Legend-Context (`symbol_map`, `line_map`)

#### **Phase 2a → Phase 2b:**
- Element-Liste (`swarm_result`)

#### **Phase 2b → Phase 2c:**
- Bereinigte Element-Liste (`swarm_graph["elements"]`)

#### **Phase 2c → Phase 2c (Fusion):**
- Verbindungs-Liste (`monolith_result`)

#### **Phase 2c (Fusion) → Phase 2d-4:**
- Haupt-Ergebnis (`self._analysis_results`)

#### **Phase 3 → Phase 4:**
- Bestes Ergebnis (`best_result`)

#### **Phase 4 → Benutzer:**
- Finales AnalysisResult (`final_result`)

### **3. Geprüfte Informationen:**

#### **Nach jeder Phase:**
- **Validierung:** Ergebnisse werden auf Korrektheit geprüft
- **Fehlerbehandlung:** Fehler werden abgefangen und geloggt
- **Fallback:** Fallback-Mechanismen bei Fehlern

#### **Nach Phase 2:**
- **Graph-Validierung:** Graph-Struktur wird geprüft
- **Koordinaten-Validierung:** Koordinaten werden korrigiert

#### **Nach Phase 3 (jede Iteration):**
- **KPI-Validierung:** KPIs werden berechnet und geprüft
- **Fehler-Validierung:** Fehler werden identifiziert
- **Score-Validierung:** Score wird berechnet und geprüft

#### **Nach Phase 4:**
- **Finale Validierung:** Finale KPIs werden berechnet
- **Visualisierungs-Validierung:** Visualisierungen werden generiert
- **Report-Validierung:** HTML-Report wird generiert

---

## 📂 Debug-Informationen (Am Ende verfügbar)

### **1. Visualisierungen:**
- **debug_map.png** - Debug-Map mit allen Elementen und Verbindungen
- **confidence_map.png** - Confidence-Map mit Confidence-Werten
- **kpi_dashboard.png** - KPI-Dashboard mit allen Metriken
- **score_curve.png** - Score-Kurve (wenn Phase 3 aktiviert)

### **2. JSON-Dateien:**
- **results.json** - Finale Ergebnisse (Elemente, Verbindungen)
- **kpis.json** - KPIs (F1-Score, Precision, Recall, etc.)
- **cgm_data.json** - CGM-Daten (für weitere Verarbeitung)
- **legend_info.json** - Legend-Informationen

### **3. HTML-Report:**
- **report.html** - Professioneller HTML-Report mit allen Visualisierungen und KPIs

### **4. Log-Dateien:**
- **pipeline_YYYYMMDD_HHMMSS.log** - Pipeline-Log (im Output-Ordner)
- **llm_calls_YYYYMMDD_HHMMSS.log** - LLM-Calls-Log (im outputs/logs Ordner)

---

## ✅ Status

**Pipeline-Prozess:**
- ✅ Detailliert dokumentiert
- ✅ Informationsfluss erklärt
- ✅ Debug-Informationen dokumentiert

---

**Status:** ✅ **Pipeline-Prozess vollständig dokumentiert**

