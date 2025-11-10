# STRATEGY TEST PLAN - Comprehensive Strategy Testing

## 🎯 **TEST ZIEL:**

Testen aller Strategien auf beiden Bildern:
1. **Simple PID** (Einfaches P&I.png)
2. **Uni-1** (Verfahrensfließbild_Uni.png)

Mit Fokus auf:
- **Fusion Strategy** (`hybrid_fusion`) - Redundanz-Test
- **Internal KPIs** (ohne Ground Truth)
- **Finale Daten** um zu sagen ob das System funktioniert

## 📊 **STRATEGIEN GETESTET:**

### 1. **simple_whole_image**
- **Beschreibung:** Monolith-Analyse des gesamten Bildes
- **Verwendung:** Einfache P&IDs
- **Features:**
  - Swarm deaktiviert (Tiles sind bei einfachen Bildern schlechter)
  - Monolith aktiviert (ganzes Bild)
  - Self-Correction aktiviert
  - Normalization aktiviert

### 2. **hybrid_fusion** (FOKUS)
- **Beschreibung:** Swarm + Monolith + Fusion
- **Verwendung:** Komplexe P&IDs, maximale Qualität
- **Features:**
  - Swarm aktiviert (Element-Erkennung)
  - Monolith aktiviert (Verbindungs-Erkennung)
  - Fusion aktiviert (kombiniert beide)
  - Self-Correction aktiviert
  - Normalization aktiviert
- **Redundanz-Test:** Beide Analyzer laufen parallel und werden fusioniert

## 📈 **INTERNE KPIS (OHNE GROUND TRUTH):**

Die Tests verwenden **interne KPIs** (keine Ground Truth):
- **Quality Score:** 0-100 (basierend auf Struktur + Confidence)
- **Total Elements:** Anzahl gefundener Elemente
- **Total Connections:** Anzahl gefundener Verbindungen
- **Graph Density:** Graph-Dichte (0.0-1.0)
- **Connected Elements:** Anzahl verbundener Elemente
- **Isolated Elements:** Anzahl isolierter Elemente
- **Num Cycles:** Anzahl Zyklen im Graph
- **Max Centrality:** Maximale Zentralität
- **Avg Element Confidence:** Durchschnittliche Element-Confidence
- **Avg Connection Confidence:** Durchschnittliche Verbindungs-Confidence

## 🔍 **WAS WIRD GETESTET:**

### **Test 1: Simple PID - simple_whole_image**
- **Erwartung:** Monolith sollte Elemente und Verbindungen finden
- **KPIs:** Quality Score sollte > 50 sein
- **Elemente:** Mindestens 5-10 Elemente
- **Verbindungen:** Mindestens 3-5 Verbindungen

### **Test 2: Simple PID - hybrid_fusion**
- **Erwartung:** Swarm + Monolith sollten kombiniert werden
- **KPIs:** Quality Score sollte > 60 sein (Fusion verbessert Qualität)
- **Elemente:** Mehr Elemente als bei simple_whole_image
- **Verbindungen:** Mehr Verbindungen als bei simple_whole_image
- **Redundanz:** Beide Analyzer sollten ähnliche Ergebnisse liefern

### **Test 3: Uni-1 - simple_whole_image**
- **Erwartung:** Monolith sollte komplexes Bild analysieren
- **KPIs:** Quality Score sollte > 40 sein (komplexeres Bild)
- **Elemente:** Mindestens 10-20 Elemente
- **Verbindungen:** Mindestens 5-10 Verbindungen

### **Test 4: Uni-1 - hybrid_fusion** (FOKUS)
- **Erwartung:** Fusion sollte beste Ergebnisse liefern
- **KPIs:** Quality Score sollte > 50 sein (Fusion verbessert Qualität)
- **Elemente:** Mehr Elemente als bei simple_whole_image
- **Verbindungen:** Mehr Verbindungen als bei simple_whole_image
- **Redundanz:** Beide Analyzer sollten ähnliche Ergebnisse liefern
- **Fusion Quality Check:** Sollte funktionieren (nur bessere Ergebnisse werden akzeptiert)

## ✅ **ERFOLGSKRITERIEN:**

### **Für Simple PID:**
1. ✅ Beide Strategien finden Elemente (> 5)
2. ✅ Beide Strategien finden Verbindungen (> 3)
3. ✅ Quality Score > 50 für beide Strategien
4. ✅ Fusion findet mehr Elemente/Verbindungen als simple_whole_image

### **Für Uni-1:**
1. ✅ Beide Strategien finden Elemente (> 10)
2. ✅ Beide Strategien finden Verbindungen (> 5)
3. ✅ Quality Score > 40 für beide Strategien
4. ✅ Fusion findet mehr Elemente/Verbindungen als simple_whole_image
5. ✅ **Fusion Quality Check funktioniert** (nur bessere Ergebnisse werden akzeptiert)

## 📁 **OUTPUT STRUKTUR:**

```
outputs/strategy_tests/
├── simple_whole_image_Simple PID_YYYYMMDD_HHMMSS/
│   ├── data/
│   │   └── test_result.json
│   ├── visualizations/
│   │   ├── debug_map.png
│   │   ├── confidence_map.png
│   │   └── score_curve.png
│   └── logs/
│       └── test.log
├── hybrid_fusion_Simple PID_YYYYMMDD_HHMMSS/
│   └── ...
├── simple_whole_image_Uni-1_YYYYMMDD_HHMMSS/
│   └── ...
├── hybrid_fusion_Uni-1_YYYYMMDD_HHMMSS/
│   └── ...
└── test_summary_YYYYMMDD_HHMMSS.json
```

## 🔧 **FIXES GETESTET:**

1. ✅ **Interne KPIs ohne Ground Truth** - Quality Score wird berechnet
2. ✅ **Monolith Response Validation** - Robusteres Parsing
3. ✅ **B-Boxes in Visualisierungen** - Werden eingezeichnet
4. ✅ **Fusion Quality Check** - Nur bessere Ergebnisse werden akzeptiert

## 📊 **ERGEBNISSE:**

Die Tests generieren:
1. **Test Results:** JSON-Dateien mit Ergebnissen pro Test
2. **Summary Report:** Zusammenfassung aller Tests
3. **Visualizations:** Debug-Maps, Confidence-Maps, Score-Curves
4. **Logs:** Detaillierte Logs für jeden Test

## 🎯 **FINALES URTEIL:**

Nach Abschluss der Tests können wir sagen:
1. ✅ **Funktioniert das System?** (Quality Score > 50)
2. ✅ **Funktioniert Fusion?** (Mehr Elemente/Verbindungen als simple_whole_image)
3. ✅ **Funktioniert Redundanz?** (Beide Analyzer liefern ähnliche Ergebnisse)
4. ✅ **Funktionieren interne KPIs?** (Quality Score wird berechnet ohne Ground Truth)

## 📝 **NÄCHSTE SCHRITTE:**

1. ⏳ **Tests laufen** (4 Tests: 2 Strategien × 2 Bilder)
2. ⏳ **Ergebnisse analysieren** (Quality Score, Elemente, Verbindungen)
3. ⏳ **Fusion validieren** (Redundanz-Test)
4. ⏳ **Finales Urteil** (Funktioniert das System?)

