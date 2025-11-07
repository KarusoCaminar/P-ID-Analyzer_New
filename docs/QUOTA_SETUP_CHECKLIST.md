# ✅ Quota Setup Checkliste

**Datum:** 2025-11-07  
**Status:** 📋 Action Items

---

## 🎯 Schnell-Checkliste

### **1. Billing aktivieren** ⚠️ ERFORDERLICH

- [ ] Google Cloud Console → **Billing**
- [ ] **Enable Billing** aktivieren
- [ ] **Payment Method** hinzufügen (Kreditkarte)
- [ ] **Budget Alerts** einrichten (z.B. $100/Monat Warnung)

---

### **2. Quota-Erhöhung beantragen**

#### **Request 1: Requests per Minute**
- [ ] Google Cloud Console → **IAM & Admin** → **Quotas**
- [ ] Service: **Vertex AI API**
- [ ] Region: **us-central1** (oder deine Region)
- [ ] Quota: **Requests per minute**
- [ ] Current: **60** → Requested: **300**
- [ ] Justification: Siehe `QUOTA_INCREASE_REQUEST_TEMPLATE.md`
- [ ] **Submit Request**

#### **Request 2: Tokens per Minute**
- [ ] Quota: **Tokens per minute**
- [ ] Current: **32,000** → Requested: **100,000**
- [ ] Justification: Siehe `QUOTA_INCREASE_REQUEST_TEMPLATE.md`
- [ ] **Submit Request**

#### **Request 3: Concurrent Requests**
- [ ] Quota: **Concurrent requests**
- [ ] Current: **10** → Requested: **50**
- [ ] Justification: Siehe `QUOTA_INCREASE_REQUEST_TEMPLATE.md`
- [ ] **Submit Request**

---

### **3. Code-Konfiguration aktualisieren**

- [ ] `config.yaml` prüfen:
  - [ ] `llm_default_timeout: 600` (10 Minuten)
  - [ ] `llm_max_retries: 5`
  - [ ] `circuit_breaker_failure_threshold: 100`
  - [ ] `circuit_breaker_recovery_timeout: 60`
  - [ ] `llm_rate_limit_requests_per_minute: 300` (anpassen nach Quota)
  - [ ] `llm_rate_limit_tokens_per_minute: 100000` (anpassen nach Quota)

---

### **4. Testen**

- [ ] Einfachen Test laufen lassen
- [ ] Parameter-Tuning Test laufen lassen (36 Tests)
- [ ] Logs prüfen:
  - [ ] Keine "429 Rate Limit" Fehler
  - [ ] Keine Timeout-Fehler
  - [ ] Circuit Breaker bleibt geschlossen

---

## 📊 Erwartete Ergebnisse

### **Vorher (mit Standard-Quota):**
- ❌ Rate Limit Fehler (429)
- ❌ Timeout Fehler
- ❌ Circuit Breaker öffnet
- ❌ Langsame Verarbeitung

### **Nachher (mit erhöhter Quota):**
- ✅ Keine Rate Limit Fehler
- ✅ Keine Timeout Fehler
- ✅ Circuit Breaker bleibt geschlossen
- ✅ Schnellere Verarbeitung

---

## 🔗 Links

- **Quota-Management:** https://console.cloud.google.com/iam-admin/quotas
- **Billing:** https://console.cloud.google.com/billing
- **Vertex AI Quotas:** https://cloud.google.com/vertex-ai/docs/quotas
- **Gemini API Quotas:** https://cloud.google.com/vertex-ai/generative-ai/docs/quotas

---

## ⏱️ Timeline

1. **Billing aktivieren:** 5 Minuten
2. **Quota-Requests stellen:** 10 Minuten
3. **Genehmigung warten:** 24-48 Stunden
4. **Config anpassen:** 5 Minuten
5. **Testen:** 30 Minuten

**Gesamt:** ~2-3 Tage (hauptsächlich Wartezeit auf Genehmigung)

---

## 💡 Tipps

- ✅ **Billing aktivieren** beschleunigt Genehmigung
- ✅ **Region:** `us-central1` hat meist beste Verfügbarkeit
- ✅ **Justification:** Detaillierte Begründung erhöht Erfolgschance
- ✅ **Monitoring:** Nach Genehmigung Usage überwachen

