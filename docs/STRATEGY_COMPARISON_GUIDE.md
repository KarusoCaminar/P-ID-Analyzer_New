# 🔬 Fusion Strategy Comparison Guide

**Datum:** 2025-11-07  
**Zweck:** Systematischer Vergleich verschiedener Fusion-Strategien

---

## 📊 Strategien zum Testen

### **Strategie 1: Current (Vollständige Redundanz mit ID-Korrektur)**
- **Status:** ✅ Implementiert
- **Beschreibung:** Vollständige Redundanz - beide Analyzer finden Elemente + Verbindungen
- **Swarm:** Elemente + lokale Verbindungen (innerhalb Tiles)
- **Monolith:** Elemente + globale Verbindungen (ganzes Bild, kann IDs korrigieren)
- **Fusion:** Intelligent kombinieren (Dual Detection, Authority, ID-Korrektur)
- **Vorteile:**
  - Redundanz → bessere Qualität
  - ID-Korrektur → richtige Verbindungen
  - Dual Detection → höhere Confidence
  - Authority → Halluzinations-Filter
- **Nachteile:**
  - Langsamer (mehr API-Calls)
  - Mehr Kosten

### **Strategie 2: Separation (Trennung)**
- **Status:** ⚠️ Benötigt Prompt-Änderungen
- **Beschreibung:** Klassische Trennung - jeder Analyzer macht was er am besten kann
- **Swarm:** NUR Elemente
- **Monolith:** NUR Verbindungen
- **Fusion:** Einfaches Mergen
- **Vorteile:**
  - Schnell (weniger API-Calls)
  - Spezialisiert
  - Weniger Kosten
- **Nachteile:**
  - Keine Redundanz
  - ID-Mismatches können nicht gefunden werden
  - Fusion kann nicht intelligent matchen

### **Strategie 3: Local/Global (Lokale/Globale Verbindungen)**
- **Status:** ⚠️ Benötigt Prompt-Änderungen
- **Beschreibung:** Hybrid-Ansatz - lokale vs. globale Verbindungen
- **Swarm:** Elemente + lokale Verbindungen (innerhalb Tiles)
- **Monolith:** Globale Verbindungen (ganzes Bild)
- **Fusion:** Lokale vs. globale Verbindungen matchen
- **Vorteile:**
  - Redundanz für Verbindungen
  - Lokale Verbindungen werden nicht verpasst
  - Globale Verbindungen werden erkannt
- **Nachteile:**
  - Komplexer (mehr Logik)
  - Mögliche Duplikate

### **Strategie 4: ID Override (ID-Überschreibung)**
- **Status:** ✅ Teilweise implementiert (Monolith kann IDs korrigieren)
- **Beschreibung:** Monolith kann falsche IDs von Swarm korrigieren
- **Swarm:** Elemente (möglicherweise falsche IDs)
- **Monolith:** Verbindungen + ID-Korrektur (ganzes Bild)
- **Fusion:** Verwendet korrigierte IDs
- **Vorteile:**
  - ID-Korrektur ohne vollständige Redundanz
  - Schneller als vollständige Redundanz
  - Richtige Verbindungen
- **Nachteile:**
  - Keine Redundanz für Elemente
  - Abhängig von Monolith-Qualität

### **Strategie 5: Full Redundancy (Vollständige Redundanz ohne ID-Korrektur)**
- **Status:** ✅ Implementiert (ähnlich wie Current)
- **Beschreibung:** Vollständige Redundanz ohne explizite ID-Korrektur
- **Swarm:** Elemente + Verbindungen
- **Monolith:** Elemente + Verbindungen
- **Fusion:** Intelligent kombinieren (Dual Detection, Authority)
- **Vorteile:**
  - Redundanz → bessere Qualität
  - Dual Detection → höhere Confidence
  - Authority → Halluzinations-Filter
- **Nachteile:**
  - Langsamer (mehr API-Calls)
  - Mehr Kosten
  - Keine explizite ID-Korrektur

---

## 🧪 Test-Durchführung

### **1. Einzelner Test (Aktuelle Strategie)**
```bash
python scripts/validation/run_live_test.py --image complex --strategy hybrid_fusion
```

### **2. Strategie-Vergleich (Alle Strategien)**
```bash
# Teste alle Strategien nacheinander
python scripts/validation/compare_fusion_strategies.py --strategies current full_redundancy separation local_global id_override --image complex
```

### **3. Einzelne Strategie testen**
```bash
# Teste nur eine Strategie
python scripts/validation/compare_fusion_strategies.py --strategies current --image complex
```

---

## 📈 Vergleichs-Metriken

### **Qualitäts-Metriken:**
- **Quality Score:** Gesamt-Qualität (0-100)
- **Element F1:** Element-Erkennung (Precision, Recall, F1)
- **Connection F1:** Verbindungs-Erkennung (Precision, Recall, F1)

### **Performance-Metriken:**
- **Duration:** Ausführungszeit (Minuten)
- **API Calls:** Anzahl der API-Aufrufe
- **Cost:** Geschätzte Kosten

### **Robustheit-Metriken:**
- **Dual Detection Rate:** Anteil der Verbindungen, die beide Analyzer finden
- **ID Correction Rate:** Anteil der korrigierten IDs
- **Hallucination Rate:** Anteil der halluzinierten Elemente/Verbindungen

---

## 📊 Erwartete Ergebnisse

### **Strategie 1: Current (Vollständige Redundanz mit ID-Korrektur)**
- **Quality Score:** 85-95 (hoch)
- **Connection F1:** 0.80-0.90 (hoch)
- **Duration:** 15-25 Minuten (langsam)
- **Dual Detection Rate:** 60-80% (hoch)

### **Strategie 2: Separation (Trennung)**
- **Quality Score:** 70-85 (mittel)
- **Connection F1:** 0.60-0.80 (mittel)
- **Duration:** 10-15 Minuten (schnell)
- **Dual Detection Rate:** 0% (keine Redundanz)

### **Strategie 3: Local/Global (Lokale/Globale Verbindungen)**
- **Quality Score:** 80-90 (hoch)
- **Connection F1:** 0.75-0.85 (hoch)
- **Duration:** 12-18 Minuten (mittel)
- **Dual Detection Rate:** 40-60% (mittel)

### **Strategie 4: ID Override (ID-Überschreibung)**
- **Quality Score:** 75-90 (hoch)
- **Connection F1:** 0.70-0.85 (hoch)
- **Duration:** 12-20 Minuten (mittel)
- **ID Correction Rate:** 10-30% (mittel)

### **Strategie 5: Full Redundancy (Vollständige Redundanz)**
- **Quality Score:** 80-95 (hoch)
- **Connection F1:** 0.75-0.90 (hoch)
- **Duration:** 15-25 Minuten (langsam)
- **Dual Detection Rate:** 60-80% (hoch)

---

## 🎯 Entscheidungs-Kriterien

### **Für maximale Qualität:**
- **Empfehlung:** Current (Vollständige Redundanz mit ID-Korrektur)
- **Grund:** Höchste Quality Score, beste Connection F1, ID-Korrektur

### **Für Geschwindigkeit:**
- **Empfehlung:** Separation (Trennung)
- **Grund:** Schnellste Ausführung, weniger API-Calls

### **Für Balance:**
- **Empfehlung:** Local/Global (Lokale/Globale Verbindungen)
- **Grund:** Guter Kompromiss zwischen Qualität und Geschwindigkeit

### **Für ID-Korrektur ohne vollständige Redundanz:**
- **Empfehlung:** ID Override (ID-Überschreibung)
- **Grund:** ID-Korrektur ohne vollständige Redundanz

---

## 📝 Nächste Schritte

1. **Test 1:** Current (Vollständige Redundanz mit ID-Korrektur) - ✅ Gestartet
2. **Test 2:** Separation (Trennung) - ⚠️ Benötigt Prompt-Änderungen
3. **Test 3:** Local/Global (Lokale/Globale Verbindungen) - ⚠️ Benötigt Prompt-Änderungen
4. **Test 4:** ID Override (ID-Überschreibung) - ✅ Teilweise implementiert
5. **Test 5:** Full Redundancy (Vollständige Redundanz) - ✅ Implementiert

### **Vergleichs-Report:**
Nach allen Tests wird ein Vergleichs-Report generiert:
- `outputs/strategy_comparison/{timestamp}/data/comparison_report.json`
- Enthält alle Metriken für alle Strategien
- Empfehlung basierend auf Kriterien

---

## 🔍 Monitoring

### **Live-Monitoring:**
```bash
# Monitor live test
python scripts/validation/monitor_live_test.py

# Monitor strategy comparison
tail -f outputs/strategy_comparison/{timestamp}/logs/comparison.log
```

### **Status-Check:**
```bash
# Check running processes
ps aux | grep python | grep -E "(run_live_test|compare_fusion_strategies)"

# Check latest results
ls -lt outputs/strategy_comparison/*/data/comparison_report.json
```

---

## 📚 Weitere Informationen

- **Fusion Strategy Fix:** `docs/FUSION_STRATEGY_FIX.md`
- **Test Overview:** `docs/TEST_OVERVIEW.md`
- **Config:** `config.yaml` (Strategien: Zeile 154-260)

