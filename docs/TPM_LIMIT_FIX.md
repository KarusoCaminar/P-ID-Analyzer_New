# 🎯 TPM Limit Fix - Das wahre Problem

**Datum:** 2025-11-08  
**Status:** ✅ Implementiert

---

## 🐛 Problem identifiziert

### **Der falsche Flaschenhals:**

Wir haben uns die ganze Zeit auf **RPM (Requests per Minute)** konzentriert, aber das eigentliche Problem ist **TPM (Tokens per Minute)**!

### **Warum TPM das Problem ist:**

**Neue Redundanz-Strategie (FUSION_STRATEGY_FIX.md):**
- **Swarm (Flash):** 500 RPM mit kleinen Kacheln (~500 Tokens pro Request) = **250.000 TPM**
- **Monolith (Pro):** 24 RPM mit GANZEM BILD (~5.000 Tokens pro Request) = **120.000 TPM**
- **Zusammen:** ~370.000 TPM

**Altes Limit:** `llm_rate_limit_tokens_per_minute: 100.000`
- **Monolith alleine sprengt bereits das Limit!** (120k TPM > 100k TPM)
- Zusammen mit Swarm: **370k TPM vs. 100k TPM Limit** = Circuit Breaker zieht Notbremse

---

## ✅ Lösung

### **1. Config-Fix (Implementiert):**

**VORHER:**
```yaml
llm_rate_limit_tokens_per_minute: 100000  # Anpassen basierend auf deiner Google Cloud Quota
```

**NACHHER:**
```yaml
llm_rate_limit_tokens_per_minute: 1500000  # CRITICAL: INCREASED from 100k to 1.5M to support Monolith (Pro) with whole image (24 RPM * 5k tokens = 120k TPM) + Swarm (Flash) without hitting limits. Monolith alone already exceeds 100k TPM!
```

### **2. Begründung:**

- **Monolith (Pro) mit whole_image:** 24 RPM * 5.000 Tokens = **120.000 TPM**
- **Swarm (Flash):** 500 RPM * 500 Tokens = **250.000 TPM**
- **Zusammen:** ~370.000 TPM
- **Neues Limit (1.5M):** Sollte ausreichen für beide Analyzer + Puffer

### **3. Google Cloud Quota:**

**WICHTIG:** Das Config-Limit ist nur ein "Tempomat" im Code. Das **echte Limit** kommt von Google Cloud Quota:

- **Standard-Quota:** 60 RPM, 32k TPM
- **Empfohlene Quota:** 300 RPM, 100k TPM
- **Benötigte Quota für unsere Strategie:** ~500 RPM, ~500k TPM

**Du musst in Google Cloud Console eine Quota-Erhöhung beantragen:**
- **Service:** Vertex AI API
- **Metric:** `Tokens per minute per project per region`
- **Region:** `us-central1` (oder `europe-west3` wenn du in Frankfurt bist)
- **Model:** `gemini-2.5-pro` (für Monolith)
- **Requested Quota:** ~500.000 TPM (oder höher)

---

## 📊 Token-Verbrauch-Analyse

### **Monolith (Pro) mit whole_image:**
- **Anfragen:** 24 RPM
- **Tokens pro Request:** ~5.000 (ganzes Bild + Prompt)
- **TPM:** 24 * 5.000 = **120.000 TPM**

### **Swarm (Flash) mit Tiles:**
- **Anfragen:** 500 RPM
- **Tokens pro Request:** ~500 (kleine Kachel + Prompt)
- **TPM:** 500 * 500 = **250.000 TPM**

### **Gesamt:**
- **TPM:** 120.000 + 250.000 = **370.000 TPM**
- **Altes Limit:** 100.000 TPM ❌ (Monolith alleine sprengt es!)
- **Neues Limit:** 1.500.000 TPM ✅ (ausreichend mit Puffer)

---

## 🔍 Warum der Circuit Breaker ausgelöst hat

1. **Monolith startet:** 120k TPM (überschreitet 100k Limit)
2. **Circuit Breaker erkennt:** Rate Limit Fehler (429)
3. **Circuit Breaker zieht Notbremse:** `Circuit breaker is open. Skipping retry to minimize API calls.`
4. **Test läuft weiter:** Aber ohne Self-Correction (Phase 3 wurde übersprungen)
5. **Ergebnis:** Schlechte Qualität (Connection F1: 0.043)

---

## 🚀 Nächste Schritte

### **1. Config-Limit erhöht (✅ Implementiert):**
- `llm_rate_limit_tokens_per_minute: 1.500.000`

### **2. Google Cloud Quota erhöhen (⚠️ Benötigt):**
- Gehe zu Google Cloud Console
- **Service:** Vertex AI API
- **Quotas:** `Tokens per minute per project per region`
- **Region:** `us-central1` (oder `europe-west3`)
- **Request:** Erhöhung auf ~500.000 TPM (oder höher)

### **3. Test nochmal durchführen:**
```bash
python scripts/validation/run_live_test.py --image complex --strategy hybrid_fusion
```

**Erwartete Verbesserungen:**
- Keine Rate Limit Fehler mehr (429)
- Circuit Breaker bleibt geschlossen
- Phase 3 (Self-Correction) läuft durch
- Connection F1 sollte sich verbessern (von 0.043 auf > 0.5)

---

## 📝 Hinweise

### **DSQ Optimizer:**
- **Aktuell:** Trackt nur RPM, nicht TPM
- **Status:** OK für jetzt (Google Cloud Quota ist das echte Limit)
- **Zukunft:** TPM tracking zum DSQ Optimizer hinzufügen für bessere Anpassung

### **Alternative Strategie (Plan B):**
Wenn Quota-Erhöhung nicht möglich ist, verwende `simple_whole_image`:
```bash
python scripts/validation/run_live_test.py --image complex --strategy simple_whole_image
```

**Vorteile:**
- Nur Monolith (Pro), kein Swarm
- ~120k TPM (unter 100k Limit, wenn Quota nicht erhöht)
- Langsam, aber keine Rate Limits

---

## 🔍 Verifikation

### **Check Config:**
```bash
grep "llm_rate_limit_tokens_per_minute" config.yaml
```

**Erwartet:** `llm_rate_limit_tokens_per_minute: 1500000`

### **Check Logs:**
```bash
grep "429\|Resource Exhausted\|Circuit breaker" outputs/live_test/*/logs/*.log
```

**Erwartet:** Keine 429-Fehler mehr (nach Quota-Erhöhung)

---

## 📚 Weitere Informationen

- **Fusion Strategy Fix:** `docs/FUSION_STRATEGY_FIX.md`
- **Quota Guide:** `docs/VERTEX_AI_QUOTA_AND_OPTIMIZATION_GUIDE.md`
- **DSQ Optimization:** `docs/DSQ_OPTIMIZATION_GUIDE.md`
- **Rate Limit Tests:** `docs/MAX_RATE_LIMIT_RECOMMENDATIONS.md`

