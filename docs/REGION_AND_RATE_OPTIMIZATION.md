# 🌍 Region & Rate Optimization Guide

**Datum:** 2025-11-07  
**Status:** ✅ Implementiert

---

## 🎯 Optimierungen

### **1. Region geändert: `us-central1` → `europe-west3` (Frankfurt)**

**Grund:**
- Bessere Latenz für europäische Benutzer
- Frankfurt ist näher als Iowa (us-central1)
- Reduziert Netzwerk-Latenz um ~50-100ms

**Konfiguration:**
- Alle Gemini 2.5 Modelle (Pro, Flash) verwenden jetzt `europe-west3`
- Embedding-Modelle verwenden auch `europe-west3`

**Verfügbarkeit prüfen:**
```bash
python scripts/validation/test_api_rate_limit.py
```

---

### **2. API Rate Limits erhöht**

#### **Initial RPM: 60 → 200**
- **Vorher:** 60 requests/min (sehr konservativ)
- **Jetzt:** 200 requests/min (3.3x höher)
- **Grund:** DSQ Optimizer passt automatisch an, wenn Rate Limits auftreten

#### **MAX_WORKERS: 10 → 15**
- **Vorher:** 10 parallele Requests
- **Jetzt:** 15 parallele Requests (50% mehr)
- **Grund:** War schon erfolgreich in vorherigen Tests

#### **Timeout Executor Workers: 4 → 8**
- **Vorher:** 4 parallele Timeout-Worker
- **Jetzt:** 8 parallele Timeout-Worker (2x mehr)
- **Grund:** Bessere Parallelität für große Bilder

---

### **3. Worker-Anzahlen erhöht**

#### **llm_executor_workers: 4 → 15**
- **Vorher:** 4 Worker für Swarm-Analyse
- **Jetzt:** 15 Worker für Swarm-Analyse (3.75x mehr)
- **Grund:** Mehr parallele Tiles = schnellere Analyse

#### **Swarm Analyzer max_workers: 6 → 15**
- **Vorher:** Cap bei 6 parallelen Tiles
- **Jetzt:** Cap bei 15 parallelen Tiles (2.5x mehr)
- **Grund:** Bessere Auslastung der API

#### **Pipeline Coordinator max_workers: 8 → 15**
- **Vorher:** Cap bei 8 parallelen Connections
- **Jetzt:** Cap bei 15 parallelen Connections (1.875x mehr)
- **Grund:** Schnellere Polyline-Extraktion

---

## 🧪 Rate Limit Test Script

**Skript:** `scripts/validation/test_api_rate_limit.py`

**Funktionen:**
- Testet verschiedene Regionen (us-central1, europe-west3, europe-west4)
- Testet verschiedene Worker-Anzahlen (5, 10, 15, 20)
- Findet maximale API Call Rate
- Findet Rate Limit Threshold

**Ausführung:**
```bash
python scripts/validation/test_api_rate_limit.py
```

**Output:**
- JSON-Datei mit Test-Ergebnissen in `outputs/rate_limit_test/`
- Zeigt beste Konfiguration (Region, Worker, RPM)
- Zeigt Rate Limit Rate und Success Rate

---

## 📊 Erwartete Verbesserungen

### **Geschwindigkeit:**
- **3-5x schneller** durch höhere Worker-Anzahlen
- **50-100ms weniger Latenz** durch europe-west3
- **2-3x höhere API Call Rate** durch DSQ Optimierung

### **Stabilität:**
- **DSQ Optimizer** passt automatisch an, wenn Rate Limits auftreten
- **Request Smoothing** verhindert Burst-Traffic
- **Exponential Backoff** für 429-Fehler

---

## ⚠️ Wichtige Hinweise

### **1. Quota erhöhen:**
- Aktuelle Quota prüfen in Google Cloud Console
- Quota-Erhöhung beantragen falls nötig:
  - Requests/min: 200-300 (empfohlen)
  - Tokens/min: 100k (empfohlen)

### **2. Region Verfügbarkeit:**
- Gemini 2.5 Pro/Flash müssen in `europe-west3` verfügbar sein
- Falls nicht verfügbar: zurück zu `us-central1` wechseln

### **3. Monitoring:**
- Rate Limit Rate überwachen (< 5% ist gut)
- DSQ Optimizer Status prüfen
- API Call Rate überwachen

---

## 🔍 Troubleshooting

### **Problem: Viele 429-Fehler**
**Lösung:**
- Initial RPM reduzieren (200 → 150)
- MAX_WORKERS reduzieren (15 → 10)
- DSQ Optimizer wird automatisch Rate reduzieren

### **Problem: Modelle nicht verfügbar in europe-west3**
**Lösung:**
- Zurück zu `us-central1` wechseln
- Oder `europe-west4` (Netherlands) testen

### **Problem: Zu langsam**
**Lösung:**
- Worker-Anzahl weiter erhöhen (15 → 20)
- Initial RPM erhöhen (200 → 250)
- Rate Limit Test ausführen um Maximum zu finden

---

## 📝 Zusammenfassung

**Vorher:**
- Region: `us-central1` (Iowa)
- Initial RPM: 60
- MAX_WORKERS: 10
- llm_executor_workers: 4
- Swarm max_workers: 6

**Jetzt:**
- Region: `europe-west3` (Frankfurt) ✅
- Initial RPM: 200 (3.3x) ✅
- MAX_WORKERS: 15 (1.5x) ✅
- llm_executor_workers: 15 (3.75x) ✅
- Swarm max_workers: 15 (2.5x) ✅

**Erwartete Verbesserung: 3-5x schneller** 🚀

