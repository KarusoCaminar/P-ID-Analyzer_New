# OVERNIGHT TEST STATUS

## Test gestartet: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

### Konfiguration
- **Dauer**: 8 Stunden
- **Strategien**: hybrid_fusion, simple_whole_image, default_flash
- **Rate-Limits**: Optimiert für Stabilität (minimiert 429-Fehler)
  - `llm_executor_workers`: 20 (reduziert von 30)
  - `llm_rate_limit_requests_per_minute`: 300 (reduziert von 500)
  - `llm_max_concurrent_requests`: 20 (reduziert von 30)
  - `circuit_breaker_failure_threshold`: 150 (erhöht von 100)
  - `circuit_breaker_recovery_timeout`: 120 (erhöht von 60)

### Neue Features im Test
- ✅ **ID-Korrektur**: LLM-basierte ID-Korrektur aktiviert (`use_id_correction: true`)
- ✅ **Hybrid Validation**: CV + Semantic Validation aktiviert
- ✅ **Phase 3 Fix**: Respektiert `use_swarm_analysis` Flag

### Monitoring
- **Logs**: `outputs/overnight_optimization/logs/`
- **Status**: `outputs/overnight_optimization/STATUS.md`
- **Ergebnisse**: `outputs/overnight_optimization/results/`

### Erwartete Ergebnisse
- Vergleich der Strategien (hybrid_fusion vs simple_whole_image vs default_flash)
- ID-Korrektur sollte Connection F1 verbessern
- Rate-Limits sollten 429-Fehler minimieren

### Nächste Schritte
1. Test läuft automatisch über Nacht
2. Am Morgen: Ergebnisse analysieren
3. Beste Strategie identifizieren
4. Weitere Optimierungen basierend auf Ergebnissen

