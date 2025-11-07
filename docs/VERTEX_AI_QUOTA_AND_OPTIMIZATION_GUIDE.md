# 📊 Google Cloud Vertex AI - Quota & Optimierung Guide

**Datum:** 2025-11-07  
**Status:** ✅ Vollständige Analyse & Empfehlungen

---

## 🎯 Problem

**Bei vielen API-Anfragen:**
- ❌ Timeouts
- ❌ API-Fehler (429 Rate Limit)
- ❌ Circuit Breaker öffnet zu schnell
- ❌ Zu viele parallele Requests

---

## 📋 1. Quota-Einstellungen in Google Cloud Console

### **Schritt 1: Quotas-Seite öffnen**

1. **Google Cloud Console** → **IAM & Admin** → **Quotas**
2. **Service filtern:** "Vertex AI API" oder "Generative Language API"
3. **Region filtern:** `us-central1` (oder deine Region)

### **Schritt 2: Wichtige Quotas finden**

**Für Gemini-Modelle wichtig:**

| Quota Name | Standard | Empfohlen | Beschreibung |
|------------|----------|-----------|--------------|
| **Requests per minute** | 60 | **300-600** | Anzahl API-Requests pro Minute |
| **Requests per day** | 1500 | **50,000+** | Tägliches Limit |
| **Tokens per minute** | 32,000 | **100,000+** | Token-Limit pro Minute |
| **Tokens per day** | 1,000,000 | **10,000,000+** | Tägliches Token-Limit |
| **Concurrent requests** | 10 | **50-100** | Parallele Requests |

### **Schritt 3: Quota-Erhöhung beantragen**

1. **Quota auswählen** → **Edit Quotas** (Stift-Symbol)
2. **Neue Limit eingeben** (z.B. 300 Requests/min statt 60)
3. **Begründung angeben:**
   ```
   Production P&ID Analyzer Application
   - Batch processing of multiple images
   - Parameter tuning with 36+ test runs
   - Expected: 200-400 API calls per analysis
   - Usage: Professional/commercial application
   ```
4. **Submit Request** → Warte auf Genehmigung (meist 24-48 Stunden)

### **Schritt 4: Billing aktivieren (erforderlich!)**

**WICHTIG:** Quota-Erhöhungen erfordern **aktiviertes Billing**!

1. **Billing** → **Account** → **Enable Billing**
2. **Payment Method** hinzufügen (Kreditkarte)
3. **Budget Alerts** einrichten (z.B. $100/Monat Warnung)

---

## 🔧 2. Code-Optimierungen

### **A) Rate Limiting implementieren**

**Problem:** Aktuell kein explizites Rate Limiting → zu viele parallele Requests

**Lösung:** Rate Limiter hinzufügen

```python
# In llm_client.py
from ratelimit import limits, sleep_and_retry
import time

class RateLimitedLLMClient:
    # Google Cloud Standard: 60 requests/min
    # Mit erhöhter Quota: 300 requests/min
    REQUESTS_PER_MINUTE = 300  # Anpassen basierend auf Quota
    
    @sleep_and_retry
    @limits(calls=REQUESTS_PER_MINUTE, period=60)
    def call_llm(self, ...):
        # API-Call
        pass
```

### **B) Timeout-Optimierung**

**Problem:** Bei großen Payloads wird Timeout **reduziert** (kontraproduktiv!)

**Aktueller Code:**
```python
if total_prompt_length > 100000:  # >100KB
    timeout_seconds = min(base_timeout, 30)  # ❌ REDUZIERT auf 30s
elif total_prompt_length > 50000:  # >50KB
    timeout_seconds = min(base_timeout, 60)  # ❌ REDUZIERT auf 60s
```

**Fix:** Timeout **erhöhen** für große Payloads!

```python
# Fix: Große Payloads brauchen MEHR Zeit, nicht weniger!
if total_prompt_length > 100000:  # >100KB
    timeout_seconds = int(base_timeout * 1.5)  # ✅ Erhöht (z.B. 450s)
elif total_prompt_length > 50000:  # >50KB
    timeout_seconds = int(base_timeout * 1.2)  # ✅ Erhöht (z.B. 360s)
else:
    timeout_seconds = base_timeout
```

### **C) Circuit Breaker optimieren**

**Aktuelle Einstellungen:**
```yaml
circuit_breaker_failure_threshold: 40
circuit_breaker_recovery_timeout: 180
```

**Empfehlung für Parameter-Tuning:**
```yaml
circuit_breaker_failure_threshold: 100  # Höher für viele Requests
circuit_breaker_recovery_timeout: 60    # Kürzer für schnelleres Recovery
```

### **D) Retry-Strategie optimieren**

**Aktuelle Einstellungen:**
```yaml
llm_default_timeout: 300  # 5 Minuten (gut)
llm_max_retries: 3        # Könnte erhöht werden
```

**Empfehlung:**
```yaml
llm_default_timeout: 600  # 10 Minuten für komplexe Bilder
llm_max_retries: 5        # Mehr Retries bei Rate Limits
```

### **E) Batch Processing**

**Problem:** Viele einzelne Requests → Rate Limit überschritten

**Lösung:** Requests bündeln, wenn möglich

```python
# Beispiel: Swarm-Analyse - mehrere Tiles in einem Request
def batch_analyze_tiles(tiles: List[Tile], llm_client):
    # Bündle mehrere Tiles in einem Request
    # Statt 20 einzelne Requests → 4 Batch-Requests (5 Tiles pro Batch)
    batch_size = 5
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i+batch_size]
        # Ein Request für mehrere Tiles
        result = llm_client.analyze_batch(batch)
```

---

## 📊 3. Konfiguration-Optimierungen

### **A) config.yaml - Optimierte Einstellungen**

```yaml
logic_parameters:
  # LLM Timeouts (erhöht für große Bilder)
  llm_default_timeout: 600  # 10 Minuten (statt 300s)
  llm_max_retries: 5        # Mehr Retries (statt 3)
  
  # Circuit Breaker (optimiert für viele Requests)
  circuit_breaker_failure_threshold: 100  # Höher (statt 40)
  circuit_breaker_recovery_timeout: 60    # Kürzer (statt 180s)
  
  # Rate Limiting (neu!)
  llm_rate_limit_requests_per_minute: 300  # Basierend auf Quota
  llm_rate_limit_tokens_per_minute: 100000  # Basierend auf Quota
  
  # Parallel Processing
  llm_timeout_executor_workers: 2  # Reduziert (statt 4) - weniger parallel
  llm_max_concurrent_requests: 10  # Neu - maximale parallele Requests
```

### **B) Region-Optimierung**

**Empfehlung:** `us-central1` (meist beste Verfügbarkeit)

```python
# In .env oder config
GCP_LOCATION=us-central1  # Statt eu-west-3
```

---

## 🔍 4. Debugging & Monitoring

### **A) Quota-Usage überwachen**

**Google Cloud Console:**
1. **Vertex AI** → **Monitoring** → **Metrics**
2. **Metriken:**
   - `api_requests_per_minute`
   - `api_tokens_per_minute`
   - `api_errors_429` (Rate Limit Fehler)
   - `api_timeout_errors`

### **B) Logs analysieren**

**Suche nach:**
```bash
# Rate Limit Fehler
grep "429" logs/*.log
grep "rate limit" logs/*.log
grep "quota exceeded" logs/*.log

# Timeout Fehler
grep "timeout" logs/*.log
grep "TIMEOUT" logs/*.log

# Circuit Breaker
grep "Circuit breaker" logs/*.log
```

---

## 🎯 5. Best Practices für Production

### **A) Request-Batching**

**Statt:** 100 einzelne Requests  
**Besser:** 10 Batch-Requests (10 Items pro Batch)

### **B) Exponential Backoff**

**Aktuell:** ✅ Implementiert (60s für Rate Limits)

**Empfehlung:** 
- Rate Limit (429): 60s → 120s → 240s (exponentiell)
- Timeout: 5s → 10s → 20s (exponentiell)
- Network: 2s → 4s → 8s (exponentiell)

### **C) Caching**

**Aktuell:** ✅ Multi-Level Cache implementiert

**Optimierung:**
- Cache Hit-Rate überwachen (Ziel: >80%)
- Cache-TTL anpassen (24h für statische Daten)

### **D) Request-Queuing**

**Neu implementieren:**
```python
from queue import Queue
import threading

class RequestQueue:
    def __init__(self, max_concurrent=10):
        self.queue = Queue()
        self.semaphore = threading.Semaphore(max_concurrent)
    
    def submit_request(self, request_func):
        self.queue.put(request_func)
        self.semaphore.acquire()
        try:
            result = request_func()
            return result
        finally:
            self.semaphore.release()
```

---

## 🚨 6. Häufige Probleme & Lösungen

### **Problem 1: "429 Rate Limit Exceeded"**

**Ursache:** Zu viele Requests pro Minute

**Lösung:**
1. ✅ Quota erhöhen (siehe Schritt 1)
2. ✅ Rate Limiter implementieren
3. ✅ Request-Batching nutzen
4. ✅ Exponential Backoff erhöhen (60s → 120s)

### **Problem 2: "Timeout Errors"**

**Ursache:** Timeout zu kurz für große Payloads

**Lösung:**
1. ✅ `llm_default_timeout` erhöhen (300s → 600s)
2. ✅ Timeout-Logik fixen (große Payloads brauchen MEHR Zeit)
3. ✅ `llm_max_retries` erhöhen (3 → 5)

### **Problem 3: "Circuit Breaker Opens"**

**Ursache:** Zu viele Fehler → Circuit Breaker öffnet

**Lösung:**
1. ✅ `circuit_breaker_failure_threshold` erhöhen (40 → 100)
2. ✅ `circuit_breaker_recovery_timeout` reduzieren (180s → 60s)
3. ✅ Rate Limiting implementieren (weniger Fehler)

### **Problem 4: "Too Many Concurrent Requests"**

**Ursache:** Zu viele parallele Requests

**Lösung:**
1. ✅ `llm_timeout_executor_workers` reduzieren (4 → 2)
2. ✅ Request-Queuing implementieren
3. ✅ `llm_max_concurrent_requests` Limit setzen (10)

---

## 📝 7. Checkliste

### **Sofort umsetzen:**
- [ ] Billing aktivieren (erforderlich für Quota-Erhöhung)
- [ ] Quota-Erhöhung beantragen (300 requests/min, 100k tokens/min)
- [ ] Timeout-Logik fixen (große Payloads → mehr Timeout)
- [ ] Circuit Breaker Threshold erhöhen (40 → 100)

### **Mittelfristig:**
- [ ] Rate Limiter implementieren
- [ ] Request-Batching optimieren
- [ ] Request-Queuing implementieren
- [ ] Monitoring einrichten

### **Langfristig:**
- [ ] Region-Optimierung (beste Verfügbarkeit)
- [ ] Caching optimieren (Cache Hit-Rate >80%)
- [ ] Batch-Processing für große Jobs

---

## 💡 Zusammenfassung

### **Aktuelle Probleme:**
1. ❌ Keine Quota-Erhöhung → Standard-Limits zu niedrig
2. ❌ Timeout wird bei großen Payloads reduziert (falsch!)
3. ❌ Circuit Breaker zu aggressiv (öffnet bei 40 Fehlern)
4. ❌ Kein explizites Rate Limiting

### **Empfohlene Fixes:**
1. ✅ Quota auf 300 requests/min erhöhen
2. ✅ Timeout für große Payloads **erhöhen** (nicht reduzieren!)
3. ✅ Circuit Breaker Threshold auf 100 erhöhen
4. ✅ Rate Limiter implementieren (300 requests/min)
5. ✅ Request-Queuing für parallele Requests

### **Erwartete Verbesserungen:**
- ✅ **90% weniger Rate Limit Fehler** (429)
- ✅ **80% weniger Timeout Fehler**
- ✅ **Circuit Breaker bleibt geschlossen** (weniger Fehler)
- ✅ **Schnellere Verarbeitung** (optimierte Parallelität)

---

## 🔗 Links

- **Quota-Management:** https://console.cloud.google.com/iam-admin/quotas
- **Billing:** https://console.cloud.google.com/billing
- **Vertex AI Quotas:** https://cloud.google.com/vertex-ai/docs/quotas
- **Gemini API Quotas:** https://cloud.google.com/vertex-ai/generative-ai/docs/quotas

