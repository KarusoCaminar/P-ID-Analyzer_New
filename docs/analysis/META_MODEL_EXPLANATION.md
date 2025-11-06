# Meta-Modell Erklärung

## Was ist das Meta-Modell?

Das **Meta-Modell** in der Strategie-Konfiguration (`meta_model`) ist **NICHT aktiv verwendet** in der aktuellen Pipeline.

### Was wird tatsächlich verwendet?

**Metacritic** (nicht Meta-Modell):
- **Zweck:** Cross-Validation zwischen Monolith- und Swarm-Ergebnisse
- **Funktion:** Vergleicht zwei Analyse-Ergebnisse, um Diskrepanzen zu finden:
  - Halluzinierte Elemente (in einem vorhanden, im anderen nicht)
  - Fehlende Verbindungen (in einem gefunden, im anderen nicht)
  - Globale Inkonsistenzen (strukturelle Unterschiede)
- **Modell:** Verwendet `critic_model_name` (nicht `meta_model`)
- **Aktivierung:** Über `use_metacritic` in `logic_parameters`

### Meta-Modell in Strategien

Das `meta_model` in den Strategien wird **nicht verwendet** in der aktuellen Pipeline-Implementierung.

**Empfehlung:** Für kleine P&IDs deaktivieren (Metacritic nicht benötigt, da Monolith zuerst läuft).

---

**Status:** ✅ Erklärt - Empfehlung: Deaktivieren für kleine P&IDs

