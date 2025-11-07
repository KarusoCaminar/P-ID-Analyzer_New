# 🚀 Maximale Rate Limit Empfehlungen

**Datum:** 2025-11-07  
**Status:** ⚠️ Test läuft noch - Ergebnisse werden aktualisiert

---

## 📊 Aktuelle Test-Ergebnisse

### **Google Gemini 2.5 Flash (us-central1):**
- **10 Workers:** 463.7 RPM (0 Rate Limits) ✅
- **15 Workers:** 510.8 RPM (0 Rate Limits) ✅
- **Status:** NOCH NICHT AM LIMIT - können höher gehen!

### **Google Gemini 2.5 Pro (us-central1):**
- **10 Workers:** 14.0 RPM (0 Rate Limits) ✅
- **15 Workers:** 28.5 RPM (0 Rate Limits) ✅
- **Status:** NOCH NICHT AM LIMIT - können höher gehen!

### **Google Gemini 2.5 Flash (europe-west3):**
- **10 Workers:** 179.7 RPM (0 Rate Limits) ✅
- **15 Workers:** 135.9 RPM (0 Rate Limits) ✅
- **Status:** Langsamer als us-central1, aber funktioniert

---

## 🎯 Aktuelle Config Einstellungen

```yaml
llm_rate_limit_requests_per_minute: 200  # Initial RPM
llm_max_concurrent_requests: 15          # Max parallele Requests
llm_executor_workers: 15                 # Worker für Swarm/Monolith
```

---

## 💡 Empfehlungen basierend auf aktuellen Tests

### **Für Flash-Modelle (Geschwindigkeit):**
- **Initial RPM:** 200 → **500 RPM** (2.5x höher)
- **Max Workers:** 15 → **20-25 Workers** (testen)
- **Max Concurrent Requests:** 15 → **20-25** (testen)

### **Für Pro-Modelle (Qualität):**
- **Initial RPM:** 200 → **30 RPM** (Flash ist viel schneller)
- **Max Workers:** 15 → **15-20 Workers** (testen)
- **Max Concurrent Requests:** 15 → **15-20** (testen)

---

## ⚠️ Wichtige Erkenntnisse

1. **Flash ist 33x schneller als Pro:**
   - Flash: ~510 RPM
   - Pro: ~28 RPM
   - **Empfehlung:** Verwende Flash wo möglich!

2. **us-central1 ist schneller als europe-west3:**
   - Flash in us-central1: ~510 RPM
   - Flash in europe-west3: ~180 RPM
   - **Aber:** europe-west3 hat bessere Latenz für EU

3. **Wir sind NOCH NICHT am Limit:**
   - Alle Tests zeigen 0 Rate Limits
   - Können definitiv höher gehen!
   - Neuer Test läuft mit 15, 20, 25, 30, 40, 50 Workers

---

## 🔄 Laufender Test

**Status:** Test läuft im Hintergrund  
**Workers getestet:** 15, 20, 25, 30, 40, 50 (graduell erhöhend)  
**Target Requests:** 100 pro Test  
**Stoppt automatisch bei:** >50% Rate Limits

---

## 📝 Nächste Schritte

1. **Warte auf Test-Ergebnisse** (15, 20, 25, 30, 40, 50 Workers)
2. **Identifiziere maximale RPM** pro Modell/Region
3. **Update Config** mit optimalen Werten
4. **Teste in Produktion** mit neuen Einstellungen

---

## 🎯 Vorläufige Empfehlung (bis Test abgeschlossen)

```yaml
# Für Flash-Modelle (Geschwindigkeit optimiert)
llm_rate_limit_requests_per_minute: 500  # Erhöht von 200
llm_max_concurrent_requests: 20          # Erhöht von 15
llm_executor_workers: 20                 # Erhöht von 15

# Für Pro-Modelle (Qualität optimiert)
# Pro ist viel langsamer - behalte konservative Werte
llm_rate_limit_requests_per_minute: 30   # Pro-spezifisch
llm_max_concurrent_requests: 15          # Behalte 15
llm_executor_workers: 15                 # Behalte 15
```

**WICHTIG:** Diese Werte werden nach Abschluss des Tests aktualisiert!

