# 📋 Google Cloud Quota Increase Request Template

**Verwendung:** Kopiere diesen Text für deine Quota-Erhöhung in Google Cloud Console

---

## Request 1: Requests per Minute

**Quota Name:** `Vertex AI API - Requests per minute`  
**Current Limit:** 60  
**Requested Limit:** 300

**Justification:**
```
Production P&ID Analyzer Application

Application Type: Professional/commercial P&ID (Process & Instrumentation Diagram) analysis tool
Use Case: Automated analysis of engineering diagrams using Google Gemini models

Expected Usage:
- Single analysis: 50-200 API calls per image
- Parameter tuning: 36 test runs × 50-200 calls = 1,800-7,200 calls
- Batch processing: 10-50 images per day
- Total: 5,000-50,000 API calls per day

Current Bottleneck:
- Standard quota (60 requests/min) is insufficient for batch processing
- Parameter tuning requires 36+ test runs, each with multiple API calls
- Circuit breaker opens frequently due to rate limit errors

Requested Increase:
- 300 requests per minute (5x standard)
- Allows batch processing without rate limit errors
- Enables parameter tuning and optimization workflows

Billing: Enabled
Payment Method: [Your payment method]
```

---

## Request 2: Tokens per Minute

**Quota Name:** `Vertex AI API - Tokens per minute`  
**Current Limit:** 32,000  
**Requested Limit:** 100,000

**Justification:**
```
Production P&ID Analyzer Application

Application Type: Professional/commercial P&ID analysis tool
Use Case: Image analysis with large prompts (20KB-100KB per request)

Expected Usage:
- Average prompt size: 30KB (30,000 tokens)
- Large prompts: 100KB (100,000 tokens) for complex diagrams
- With 300 requests/min: 9M-30M tokens/min possible
- Realistic usage: 50-100 requests/min × 30KB = 1.5M-3M tokens/min

Current Bottleneck:
- Standard quota (32k tokens/min) allows only 1-2 large requests per minute
- Complex P&ID analysis requires large prompts with detailed instructions
- Token limit reached before request limit

Requested Increase:
- 100,000 tokens per minute (3x standard)
- Allows 3-10 large requests per minute
- Enables complex image analysis workflows

Billing: Enabled
Payment Method: [Your payment method]
```

---

## Request 3: Concurrent Requests

**Quota Name:** `Vertex AI API - Concurrent requests`  
**Current Limit:** 10  
**Requested Limit:** 50

**Justification:**
```
Production P&ID Analyzer Application

Application Type: Professional/commercial P&ID analysis tool
Use Case: Parallel processing of multiple image tiles/quadrants

Expected Usage:
- Swarm analysis: 20-50 parallel tile requests
- Quadrant analysis: 4-6 parallel quadrant requests
- Batch processing: 5-10 parallel image analyses

Current Bottleneck:
- Standard quota (10 concurrent) limits parallel processing
- Swarm analysis requires 20-50 parallel requests
- Sequential processing too slow for production use

Requested Increase:
- 50 concurrent requests (5x standard)
- Enables efficient swarm/quadrant analysis
- Improves processing speed for batch jobs

Billing: Enabled
Payment Method: [Your payment method]
```

---

## 📝 Anleitung

1. **Google Cloud Console** → **IAM & Admin** → **Quotas**
2. **Service filtern:** "Vertex AI API"
3. **Region filtern:** `us-central1` (oder deine Region)
4. **Quota auswählen** → **Edit Quotas** (Stift-Symbol)
5. **Justification kopieren** (oben)
6. **Requested Limit eingeben**
7. **Submit Request**

**Genehmigungszeit:** 24-48 Stunden (meist schneller bei aktiviertem Billing)

---

## ⚠️ WICHTIG

- ✅ **Billing muss aktiviert sein** (erforderlich für Quota-Erhöhung)
- ✅ **Payment Method** hinzufügen (Kreditkarte)
- ✅ **Budget Alerts** einrichten (z.B. $100/Monat Warnung)
- ✅ **Region:** `us-central1` hat meist beste Verfügbarkeit

