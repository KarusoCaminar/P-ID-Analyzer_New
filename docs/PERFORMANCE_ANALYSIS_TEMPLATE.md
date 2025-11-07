# 📊 Performance Analysis Template

**Datum:** 2025-11-07  
**Test:** Full Pipeline Test - Verfahrensfließbild_Uni.png  
**Strategy:** hybrid_fusion

---

## 🎯 Test-Konfiguration

- **Test Image:** Verfahrensfließbild_Uni.png
- **Strategy:** hybrid_fusion (Swarm + Monolith + Fusion)
- **DSQ Optimization:** ENABLED
- **Self-Correction:** ENABLED
- **Expected Duration:** 10-20 Minuten

---

## 📈 Performance-Metriken

### **API Calls:**
- Total API Calls: [WIRD GEMESSEN]
- API Call Rate: [WIRD GEMESSEN] calls/minute
- Swarm API Calls: [WIRD GEMESSEN]
- Monolith API Calls: [WIRD GEMESSEN]

### **Rate Limits:**
- Total 429 Errors: [WIRD GEMESSEN]
- Rate Limit Rate: [WIRD GEMESSEN]%
- DSQ Throttles: [WIRD GEMESSEN]

### **DSQ Optimizer:**
- Current RPM: [WIRD GEMESSEN]
- Success Rate: [WIRD GEMESSEN]%
- Rate Limit Events: [WIRD GEMESSEN]

### **Timing:**
- Total Duration: [WIRD GEMESSEN] Minuten
- Phase 0 (Complexity): [WIRD GEMESSEN] Sekunden
- Phase 1 (Pre-analysis): [WIRD GEMESSEN] Sekunden
- Phase 2a (Swarm): [WIRD GEMESSEN] Sekunden
- Phase 2b (Monolith): [WIRD GEMESSEN] Sekunden
- Phase 2c (Fusion): [WIRD GEMESSEN] Sekunden
- Phase 3 (Self-Correction): [WIRD GEMESSEN] Sekunden
- Phase 4 (Post-processing): [WIRD GEMESSEN] Sekunden

---

## 🔍 Analyse

### **Bottlenecks:**
- [ ] Phase 0 zu langsam?
- [ ] Swarm Analysis zu langsam?
- [ ] Monolith Analysis zu langsam?
- [ ] Fusion zu langsam?
- [ ] Self-Correction zu langsam?

### **Rate Limit Issues:**
- [ ] Zu viele 429-Fehler?
- [ ] DSQ Optimizer passt Rate nicht an?
- [ ] Request Smoothing funktioniert nicht?

### **Performance Issues:**
- [ ] API Calls zu langsam?
- [ ] Timeouts?
- [ ] Circuit Breaker öffnet?

---

## 💡 Optimierungsvorschläge

### **1. Rate Limiting:**
- [ ] Start-Rate erhöhen (60 → 100 RPM)?
- [ ] MAX_WORKERS erhöhen?
- [ ] Request Smoothing anpassen?

### **2. Strategy:**
- [ ] Self-Correction deaktivieren (für Speed)?
- [ ] Monolith Quadranten optimieren?
- [ ] Swarm Tile-Größe anpassen?

### **3. DSQ Optimizer:**
- [ ] Initial RPM erhöhen?
- [ ] Max RPM erhöhen?
- [ ] Backoff-Strategie anpassen?

---

## ✅ Ergebnisse

### **KPIs:**
- Element F1: [WIRD GEMESSEN]
- Connection F1: [WIRD GEMESSEN]
- Quality Score: [WIRD GEMESSEN]

### **Erfolg:**
- [ ] Test erfolgreich abgeschlossen
- [ ] Alle Phasen durchgelaufen
- [ ] Keine kritischen Fehler

---

## 📝 Notizen

[Wird während des Tests ausgefüllt]

