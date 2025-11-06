# ⏱️ Test-Laufzeit: Schätzung

**Datum:** 2025-11-06

---

## 📊 Laufzeit-Schätzung pro Test

### Einzelner Test (z.B. Test 2)

**Phasen:**
- Phase 0 (Complexity Analysis): ~5-10 Sekunden (CV-basiert, schnell)
- Phase 1 (Pre-Analysis): ~10-20 Sekunden (Legenden-Erkennung)
- Phase 2 (Core Analysis):
  - **Monolith-Only:** ~30-60 Sekunden (1 LLM-Call für gesamtes Bild)
  - **Swarm-Only:** ~60-120 Sekunden (50-80 Tiles × ~1-2 Sekunden pro Tile)
  - **Swarm + Monolith:** ~90-180 Sekunden (beide Phasen)
- Phase 2c (Fusion): ~5-10 Sekunden (lokal, schnell)
- Phase 2d (Predictive): ~5-10 Sekunden (lokal, schnell)
- Phase 2e (Polyline): ~10-20 Sekunden (CV-basiert)
- Phase 3 (Self-Correction): ~30-60 Sekunden (wenn aktiv, 1-2 Iterationen)
- Phase 4 (Post-Processing): ~10-20 Sekunden (KPIs, CGM, Visualisierungen)

**Gesamt pro Test:**
- **Test 1 (Phase 1 only):** ~15-30 Sekunden
- **Test 2 (Monolith-All):** ~60-120 Sekunden (1-2 Minuten)
- **Test 3 (Swarm-Only):** ~90-150 Sekunden (1.5-2.5 Minuten)
- **Test 4 (Complex P&ID):** ~120-240 Sekunden (2-4 Minuten)
- **Test 5a/5b/5c:** ~150-300 Sekunden (2.5-5 Minuten)

### Alle Tests (7 Tests)

**Gesamt-Laufzeit:**
- **Minimum:** ~10-15 Minuten (wenn alles schnell läuft)
- **Typisch:** ~20-30 Minuten (realistische Schätzung)
- **Maximum:** ~40-60 Minuten (wenn LLM-Calls langsam sind)

**Faktoren:**
- LLM-API-Latenz (kann variieren)
- Bildgröße (größere Bilder = mehr Tiles)
- Komplexität (mehr Elemente = mehr LLM-Calls)

---

## 🚀 Test starten

Ich starte jetzt **Test 2** (Baseline Simple P&ID) als ersten Test, damit Sie die Logs live sehen können.

**Geschätzte Laufzeit für Test 2:** ~1-2 Minuten

