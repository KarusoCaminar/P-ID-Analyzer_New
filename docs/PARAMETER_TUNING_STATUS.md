# 📊 Parameter Tuning Status

**Datum:** 2025-11-07  
**Status:** 🟢 Läuft

---

## 🎯 Ziel

Optimierung der `adaptive_threshold` Parameter für maximale Connection F1-Score.

---

## 📋 Parameter-Ranges

- **adaptive_threshold_factor**: [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
- **adaptive_threshold_min**: [15, 20, 25, 30, 40]
- **adaptive_threshold_max**: [100, 125, 150, 200, 250]

**Total:** 6 × 5 × 5 = **150 Parameter-Kombinationen**

---

## 🖼️ Test-Bild

- **Einfaches P&ID**: `training_data/simple_pids/Einfaches P&I.png`
- **Ground Truth**: `training_data/simple_pids/Einfaches P&I_truth.json`

---

## 🔧 Strategie

- **Strategy**: `simple_whole_image`
- **Geschwindigkeit**: ~5-10 Minuten pro Test
- **Geschätzte Gesamtzeit**: ~12-25 Stunden

---

## 📁 Output-Struktur

```
outputs/parameter_tuning/
├── logs/
│   └── parameter_tuning.log          # Live-Logs
├── data/
│   ├── parameter_tuning_results.json # Alle Ergebnisse
│   └── parameter_tuning_summary.json # Zusammenfassung + Top 5
├── artifacts/                         # Test-Artifacts
├── visualizations/                    # Visualisierungen
└── README.md                          # Struktur-Erklärung
```

---

## 📊 Live-Monitoring

Das Skript läuft im Hintergrund mit Live-Logging:

```bash
# Logs anzeigen:
Get-Content outputs\parameter_tuning\logs\parameter_tuning.log -Tail 50 -Wait

# Oder:
tail -f outputs/parameter_tuning/logs/parameter_tuning.log
```

---

## 🎯 Ziel-Metriken

- **Connection F1-Score**: > 0.8 (Ziel)
- **Element F1-Score**: > 0.95 (sollte konstant bleiben)
- **Quality Score**: > 80.0

---

## 📈 Ergebnisse

Ergebnisse werden automatisch gespeichert in:
- `outputs/parameter_tuning/data/parameter_tuning_results.json`
- `outputs/parameter_tuning/data/parameter_tuning_summary.json` (Top 5 + Beste Parameter)

---

## 🔄 Nächste Schritte

1. ⏳ Parameter-Tuning läuft (150 Kombinationen)
2. ⏳ Beste Parameter identifizieren
3. ⏳ Parameter in `config.yaml` aktualisieren
4. ⏳ Validierung auf komplexem Bild

