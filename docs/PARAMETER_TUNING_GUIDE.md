# 🔧 Parameter Tuning Guide

**Datum:** 2025-11-07  
**Status:** 📋 Optimierungsleitfaden

---

## 📋 Übersicht

Dieses Dokument beschreibt, wie die kritischen Parameter in `config.yaml` optimiert werden, um die beste Connection F1-Score zu erreichen.

---

## 🎯 Kritische Parameter für Connection F1-Score

### 1. **Adaptive Thresholds (Line Extractor)**

**Lage:** `config.yaml` → `logic_parameters` → `line_extraction`

**Parameter:**
```yaml
line_extraction:
  adaptive_threshold_factor: 0.02  # 2% der Bilddiagonale
  adaptive_threshold_min: 25       # Minimum 25 Pixel
  adaptive_threshold_max: 150      # Maximum 150 Pixel
```

**Was sie tun:**
- Bestimmen, wie weit ein Linienendpunkt von einem Element entfernt sein darf, um als Verbindung erkannt zu werden
- Zu klein → CV findet keine Verbindungen (Connection F1 = 0.0)
- Zu groß → CV verbindet falsche Elemente (falsche Positive)

**Optimierung:**

1. **Testlauf mit verschiedenen Werten:**
   ```bash
   # Variiere adaptive_threshold_factor von 0.01 bis 0.1
   # Teste mit verschiedenen Bildern (einfach, komplex)
   python scripts/validation/run_overnight_optimization.py
   ```

2. **Parameter-Range:**
   - `adaptive_threshold_factor`: 0.01, 0.02, 0.03, 0.05, 0.07, 0.10
   - `adaptive_threshold_min`: 15, 20, 25, 30, 40
   - `adaptive_threshold_max`: 100, 125, 150, 200, 250

3. **Ziel:**
   - Höchster durchschnittlicher Connection F1-Score über alle Testbilder
   - Balance zwischen Precision (keine falschen Verbindungen) und Recall (alle echten Verbindungen)

---

## 📊 Optimierungs-Strategie

### Schritt 1: Baseline Messen

```bash
# Führe 10 Testläufe mit aktuellen Parametern durch
python scripts/validation/run_overnight_optimization.py --strategy simple_whole_image --runs 10
```

**Erwartete Outputs:**
- `outputs/overnight_optimization/results/` → JSON-Dateien mit KPIs
- Durchschnittlicher Connection F1-Score

### Schritt 2: Parameter-Grid-Search

```bash
# Variiere adaptive_threshold_factor
for factor in 0.01 0.02 0.03 0.05 0.07 0.10; do
    # Ändere config.yaml
    # Führe Testlauf durch
    python scripts/validation/run_overnight_optimization.py --strategy simple_whole_image
done
```

### Schritt 3: Beste Parameter Identifizieren

**Kriterien:**
1. **Connection F1-Score** > 0.8 (Ziel)
2. **Element F1-Score** bleibt > 0.95 (sollte nicht verschlechtern)
3. **Quality Score** > 80.0

---

## 🔍 Weitere Optimierungsparameter

### 2. **IoU Match Threshold (Fusion)**

**Lage:** `config.yaml` → `logic_parameters` → `fusion`

**Parameter:**
```yaml
fusion:
  iou_match_threshold: 0.5  # IoU-Schwellenwert für Element-Matching
```

**Optimierung:** Teste Werte von 0.3 bis 0.7

### 3. **Graph Completion Thresholds**

**Lage:** `config.yaml` → `logic_parameters` → `graph_completion`

**Parameter:**
```yaml
graph_completion:
  distance_threshold: 0.03  # Maximale Distanz für automatische Verbindung
  isolated_node_distance: 0.03  # Distanz für "echt isoliert"
```

**Optimierung:** Teste Werte von 0.02 bis 0.05

---

## 📈 Test-Strategien

### A/B-Test-Strategien:

1. **default_flash** (schnell, nur Swarm)
   - Geschwindigkeit: ~3-8 Minuten
   - Erwartete Connection F1: 0.6-0.8

2. **simple_whole_image** (Monolith, ganzes Bild)
   - Geschwindigkeit: ~5-10 Minuten
   - Erwartete Connection F1: 0.7-0.9

3. **hybrid_fusion** (Swarm + Monolith + Fusion)
   - Geschwindigkeit: ~10-20 Minuten
   - Erwartete Connection F1: 0.8-0.95

---

## 🎯 Ziel-Metriken

### Mindest-Anforderungen:
- **Element F1-Score:** > 0.95 ✅
- **Connection F1-Score:** > 0.8 🎯 (aktuell: ~0.0-0.6)
- **Quality Score:** > 80.0 ✅

### Optimal:
- **Element F1-Score:** > 0.98
- **Connection F1-Score:** > 0.90
- **Quality Score:** > 90.0

---

## 📝 Notizen

- **Adaptive Thresholds** sind die kritischsten Parameter für Connection F1
- **Hybrid Validation** (CV + Semantic) sollte immer aktiviert sein (`use_hybrid_validation: true`)
- **Parameter-Tuning** sollte auf mehreren Testbildern durchgeführt werden (einfach + komplex)

---

## 🔄 Nächste Schritte

1. ✅ Hybrid Validation implementiert
2. ✅ Strategien korrigiert (default_flash, hybrid_fusion)
3. ⏳ Parameter-Tuning durchführen (adaptiv_threshold_factor optimieren)
4. ⏳ A/B-Tests durchführen (default_flash vs. simple_whole_image vs. hybrid_fusion)

