# Test-Daten-Analyse: Viewshots und andere Beispiele

## Übersicht

Dieses Dokument analysiert die aktuelle Test-Datenstruktur und erklärt, was "Viewshots" und andere Beispiele sind, und ob sie tatsächlich nützlich sind.

---

## 1. Test-Daten-Struktur

### 1.1 Training Data (`training_data/`)

```
training_data/
├── complex_pids/          # Komplexe P&ID-Bilder (mit Legende)
│   ├── page_1_original.png
│   ├── page_1_original_truth_cgm.json
│   ├── page_2_original.png
│   ├── page_2_original_truth_cgm.json
│   ├── page_3_original.png
│   ├── page_3_original_truth_cgm.json
│   ├── page_4_original.png
│   ├── page_4_original_truth_cgm.json
│   ├── Verfahrensfließbild_Uni.png
│   ├── Verfahrensfließbild_Uni_truth.json
│   ├── temp_tiles/        # Temporäre Tile-Dateien (während Analyse generiert)
│   │   └── correction_snippet_*.png
│   └── ...
├── simple_pids/           # Einfache P&ID-Bilder (ohne Legende)
│   ├── Einfaches P&I.png
│   ├── Einfaches P&I_truth.json
│   └── temp_tiles/
├── viewshot_examples/      # Viewshot-Beispiele (NOCH NICHT GENERIERT)
│   ├── valve/
│   ├── pump/
│   ├── flow_sensor/
│   ├── mixer/
│   ├── source/
│   ├── sink/
│   └── sample_point/
├── pretraining_symbols/   # Vorab extrahierte Symbole
│   └── Pid-symbols-PDF_sammlung.png
├── organized_tests/       # Organisierte Test-Daten (Duplikate?)
│   ├── complex_pids/
│   └── simple_pids/
└── standard_legend.json    # Standard-Legende
```

### 1.2 Test-Outputs (`outputs/strategy_validation/`)

```
outputs/strategy_validation/
├── Test_1_Baseline_Phase_1_(Legenden-Erkennung)/
│   ├── config_snapshot.yaml
│   ├── prompts_snapshot.json
│   └── test_metadata.md
├── Test_2_Baseline_Simple_P&ID_(Monolith-All)/
│   ├── Einfaches P&I_results.json
│   ├── Einfaches P&I_kpis.json
│   ├── Einfaches P&I_debug_map.png
│   ├── Einfaches P&I_confidence_map.png
│   ├── Einfaches P&I_score_curve.png
│   ├── Einfaches P&I_kpi_dashboard.png
│   ├── Einfaches P&I_report.html
│   ├── output_phase_3_selfcorrect_ITER_*.json
│   └── ...
└── ...
```

---

## 2. Was sind "Viewshots"?

### 2.1 Definition

**Viewshots** sind ausgeschnittene Symbol-Beispiele aus echten Uni-Bildern (P&ID-Diagrammen). Sie werden verwendet, um dem LLM visuelle Referenzen zu geben, wie bestimmte Symbole in der Praxis aussehen.

### 2.2 Wie werden sie generiert?

Das Skript `scripts/utilities/extract_viewshots_from_uni_bilder.py`:

1. **Lädt Ground Truth-Daten** aus `*_truth_cgm.json` Dateien
2. **Extrahiert Bounding Boxes** für jedes Element
3. **Croppt Symbole** aus den Originalbildern (mit 10% Padding)
4. **Organisiert sie nach Typ** in Unterordnern:
   - `valve/`
   - `pump/`
   - `flow_sensor/`
   - `mixer/`
   - `source/`
   - `sink/`
   - `sample_point/`

### 2.3 Wie werden sie verwendet?

**In den Analyzern (`monolith_analyzer.py`, `swarm_analyzer.py`):**

1. **Laden**: `_load_viewshot_examples()` lädt die ersten 3 Viewshots pro Typ
2. **Einbinden in Prompts**: Die Viewshot-Pfade werden in die LLM-Prompts eingefügt
3. **Visuelle Referenz**: Das LLM erhält Text-Beschreibungen der Viewshots (z.B. "Viewshot Example 1: valve_CHP1_0.png")

**Beispiel aus `config.yaml`:**

```yaml
{viewshot_valve_examples}
{viewshot_flow_sensor_examples}
{viewshot_mixer_examples}
...
```

**Wird ersetzt durch:**

```
**VIEWSHOT EXAMPLES FOR VALVE (from real Uni-Bilder):**
**CRITICAL:** These are actual visual examples from Uni-Bilder. Use these to recognize similar symbols.

**Viewshot Example 1:** valve_MV3121A_3.png
- Visual: Real valve symbol from Uni-Bild (see image)
- Common pattern: Use this visual pattern to identify similar valve symbols
```

---

## 3. Was sind "temp_tiles" und "correction_snippets"?

### 3.1 Definition

**temp_tiles/** sind temporäre Ordner, die während der Analyse generiert werden:

1. **Tiles**: Aufgeteilte Bildsegmente für Swarm-Analyse (z.B. 70 Tiles für komplexe P&IDs)
2. **correction_snippets**: Bildausschnitte, die während der Self-Correction Loop generiert werden

### 3.2 Wie werden sie generiert?

**In `src/utils/image_utils.py`:**

- **Tiles**: `generate_raster_grid()` teilt große Bilder in kleinere Segmente auf
- **Correction Snippets**: `save_correction_snippet()` speichert Bildausschnitte für Re-Analyse

### 3.3 Sind sie nützlich?

**JA, aber nur temporär:**

- ✅ **Während der Analyse**: Wichtig für Swarm-Analyse (Tile-basierte Verarbeitung)
- ✅ **Debugging**: Nützlich, um zu sehen, welche Bildsegmente analysiert wurden
- ❌ **Nach der Analyse**: Können gelöscht werden (sind temporär)
- ❌ **Nicht für Tests**: Sollten nicht in Git committed werden

**Empfehlung**: `.gitignore` sollte `temp_tiles/` enthalten.

---

## 4. Was sind "Ground Truth"-Dateien?

### 4.1 Definition

**Ground Truth** (`*_truth.json` oder `*_truth_cgm.json`) sind manuell erstellte Referenzdaten, die die "korrekte" Analyse eines P&ID-Diagramms enthalten.

### 4.2 Struktur

**Einfaches Format (`Einfaches P&I_truth.json`):**

```json
{
  "elements": [
    { "id": "P-201", "type": "Source", "label": "From Transfer Pump P-201" },
    { "id": "Fv-3-3040", "type": "Valve", "label": "Fv-3-3040" },
    ...
  ],
  "connections": [
    {
      "name": "Conn_P201_to_Fv3040",
      "from_converter_ports": [{"unit_name": "P-201", "port": "Out"}],
      "to_converter_ports": [{"unit_name": "Fv-3-3040", "port": "In"}]
    },
    ...
  ]
}
```

**CGM-Format (`page_1_original_truth_cgm.json`):**

```json
{
  "elements": [
    { "id": "CHP1", "type": "Source", "label": "CHP1", "bbox": null },
    { "id": "PU3121", "type": "Pump", "label": "PU3121", "bbox": null },
    ...
  ],
  "connections": [
    { "from_id": "CHP1", "to_id": "MV3121A" },
    ...
  ]
}
```

### 4.3 Sind sie nützlich?

**JA, absolut kritisch:**

- ✅ **KPI-Berechnung**: Werden verwendet, um Precision, Recall, F1-Score zu berechnen
- ✅ **Test-Validierung**: Ermöglichen automatisierte Tests (`run_strategy_validation.py`)
- ✅ **Performance-Messung**: Erlauben objektive Bewertung der Pipeline-Performance
- ✅ **Training**: Können für zukünftiges Training verwendet werden

---

## 5. Sind Viewshots tatsächlich nützlich?

### 5.1 Theoretischer Nutzen

**JA, sehr nützlich:**

1. **Visuelle Referenz**: Geben dem LLM konkrete Beispiele, wie Symbole aussehen
2. **Typ-Erkennung**: Verbessern die Genauigkeit der Typ-Erkennung
3. **Domain-Specific**: Zeigen echte Symbole aus Uni-Bildern (nicht generische Beispiele)

### 5.2 Praktischer Status

**PROBLEM: Viewshots sind noch nicht generiert!**

- ❌ **`training_data/viewshot_examples/` ist leer**
- ❌ **Das Skript `extract_viewshots_from_uni_bilder.py` hat einen Bug** (falscher Pfad)
- ❌ **Viewshots werden aktuell NICHT verwendet** (weil der Ordner leer ist)

### 5.3 Empfehlung

**JA, generiere sie:**

1. **Fix das Skript**: Korrigiere den Pfad-Bug in `extract_viewshots_from_uni_bilder.py`
2. **Generiere Viewshots**: Führe das Skript aus, um Viewshots aus `page_1-4_original.png` zu extrahieren
3. **Teste den Nutzen**: Vergleiche Performance mit/ohne Viewshots

**Erwarteter Nutzen:**

- ✅ **+5-10% F1-Score** für Typ-Erkennung
- ✅ **Weniger Halluzinationen** (LLM sieht echte Beispiele)
- ✅ **Bessere Konsistenz** (LLM lernt aus echten Symbolen)

---

## 6. Andere Beispiele und Dateien

### 6.1 `pretraining_symbols/`

- **Inhalt**: `Pid-symbols-PDF_sammlung.png` (Sammlung von P&ID-Symbolen)
- **Status**: Unklar, ob verwendet
- **Empfehlung**: Prüfen, ob nützlich, sonst entfernen

### 6.2 `organized_tests/`

- **Inhalt**: Duplikate von `complex_pids/` und `simple_pids/`
- **Status**: Vermutlich veraltet
- **Empfehlung**: Entfernen oder konsolidieren

### 6.3 `standard_legend.json`

- **Inhalt**: Standard-Legende-Definitionen
- **Status**: Wird vermutlich verwendet
- **Empfehlung**: Behalten

---

## 7. Zusammenfassung und Empfehlungen

### 7.1 Aktueller Status

| Komponente | Status | Nützlich? | Aktion |
|------------|--------|-----------|--------|
| **Viewshots** | ❌ Nicht generiert | ✅ Ja | Fix Skript + Generieren |
| **temp_tiles/** | ⚠️ Temporär | ⚠️ Nur während Analyse | `.gitignore` hinzufügen |
| **Ground Truth** | ✅ Vorhanden | ✅ Ja | Behalten |
| **pretraining_symbols/** | ❓ Unklar | ❓ Unklar | Prüfen |
| **organized_tests/** | ⚠️ Duplikate | ❌ Nein | Entfernen |

### 7.2 Empfohlene Aktionen

1. **Fix `extract_viewshots_from_uni_bilder.py`**:
   - Korrigiere den Pfad-Bug (Zeile 144: `project_root` statt `Path(__file__).parent.parent`)
   - Generiere Viewshots aus `page_1-4_original.png`

2. **Bereinige `training_data/`**:
   - Entferne `organized_tests/` (Duplikate)
   - Prüfe `pretraining_symbols/` (ob verwendet)

3. **Aktualisiere `.gitignore`**:
   - Füge `temp_tiles/` hinzu
   - Füge `correction_snippet_*.png` hinzu

4. **Teste Viewshot-Nutzen**:
   - Führe Tests mit/ohne Viewshots durch
   - Vergleiche F1-Scores

---

## 8. Technische Details

### 8.1 Viewshot-Generierung

**Prozess:**

1. **Input**: `page_1_original.png` + `page_1_original_truth_cgm.json`
2. **Extraktion**: Für jedes Element in Ground Truth:
   - Hole Bounding Box
   - Crop Symbol (mit 10% Padding)
   - Speichere in `viewshot_examples/{type}/{type}_{id}_{idx}.png`
3. **Output**: Organisierte Symbol-Beispiele nach Typ

**Beispiel:**

```
viewshot_examples/
├── valve/
│   ├── valve_MV3121A_3.png
│   ├── valve_MV3121B_4.png
│   └── ...
├── pump/
│   ├── pump_PU3121_1.png
│   └── ...
└── ...
```

### 8.2 Viewshot-Verwendung in Prompts

**Aktuell (ohne Viewshots):**

```
Example 1: Valve (MOST COMMON)
- Visual: Circle with diagonal line through it
- Type: "Valve"
```

**Mit Viewshots (geplant):**

```
Example 1: Valve (MOST COMMON)
- Visual: Circle with diagonal line through it
- Type: "Valve"

**VIEWSHOT EXAMPLES FOR VALVE (from real Uni-Bilder):**
**Viewshot Example 1:** valve_MV3121A_3.png
- Visual: Real valve symbol from Uni-Bild (see image)
- Common pattern: Use this visual pattern to identify similar valve symbols
```

**Vorteil**: LLM sieht echte Beispiele, nicht nur Text-Beschreibungen.

---

## 9. Fazit

**Viewshots sind sehr nützlich**, aber aktuell **nicht implementiert** (Ordner ist leer). 

**Empfehlung**: 
1. Fix das Skript
2. Generiere Viewshots
3. Teste den Nutzen
4. Integriere sie in die Pipeline

**Andere Beispiele** (temp_tiles, correction_snippets) sind **temporär** und sollten nicht in Git committed werden.

