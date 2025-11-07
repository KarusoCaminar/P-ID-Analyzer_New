# Output Structure Standard - Gold Standard für Test-Outputs

## Übersicht

Dieses Dokument definiert den **Gold Standard** für die Output-Ordnerstruktur aller Test-Läufe. Diese Struktur **MUSS** von allen Testskripten, GUIs und CLI-Runnern verwendet werden.

## Ordnerstruktur

```
outputs/
  {test_type}/  (z.B. live_test, iterative_tests, phase1_tests)
    YYYYMMDD_HHMMSS/  (Timestamp für jeden Testlauf)
      logs/
        test.log
        analysis.log
        errors.log
      visualizations/
        {image_name}_score_curve.png
        {image_name}_confidence_map.png
        {image_name}_debug_map.png
        {image_name}_kpi_dashboard.png
        debug_map_iteration_1.png
        debug_map_iteration_2.png
        ...
      data/
        test_result.json
        {image_name}_results.json
        {image_name}_cgm_data.json
        {image_name}_cgm_network_generated.py
        {image_name}_kpis.json
        {image_name}_legend_info.json
        output_phase_2d_predictive.json
        output_phase_2e_polyline.json
        output_phase_3_selfcorrect_ITER_1.json
        output_phase_3_selfcorrect_ITER_2.json
        ...
      artifacts/
        config_snapshot.yaml
        prompts_snapshot.json
        test_metadata.md
        {image_name}_report.html
      temp/
        temp_quadrants/
        temp_polylines/
      README.md  (Erklärt die Struktur)
```

## Verwendung

### 1. Mit OutputStructureManager (Empfohlen)

```python
from src.utils.output_structure_manager import OutputStructureManager

# Erstelle Output-Struktur
output_manager = OutputStructureManager(
    base_output_dir=project_root / "outputs",
    test_type="live_test"
)

# Verwende die strukturierten Pfade
log_file = output_manager.get_log_file("test.log")
data_file = output_manager.get_data_path("test_result.json")
viz_file = output_manager.get_visualization_path("score_curve.png")
artifact_file = output_manager.get_artifact_path("config_snapshot.yaml")
```

### 2. Ohne OutputStructureManager (Fallback)

```python
from src.utils.output_structure_manager import ensure_output_structure

# Stelle sicher, dass die Struktur existiert
ensure_output_structure(Path(output_dir))

# Verwende die Unterordner manuell
log_file = Path(output_dir) / "logs" / "test.log"
data_file = Path(output_dir) / "data" / "test_result.json"
viz_file = Path(output_dir) / "visualizations" / "score_curve.png"
artifact_file = Path(output_dir) / "artifacts" / "config_snapshot.yaml"
```

## Dateitypen-Zuordnung

| Dateityp | Unterordner | Beispiele |
|----------|-------------|-----------|
| Logs | `logs/` | test.log, analysis.log, errors.log |
| Visualisierungen | `visualizations/` | *.png (score_curve, debug_map, etc.) |
| Daten | `data/` | *.json, *.py (results, CGM, KPIs, etc.) |
| Artefakte | `artifacts/` | *.yaml, *.json, *.md, *.html (config, metadata, reports) |
| Temporäre Dateien | `temp/` | temp_quadrants/, temp_polylines/ |

## Implementierungsrichtlinien

### CRITICAL: Diese Regeln MÜSSEN befolgt werden

1. **NIEMALS Dateien direkt im Haupt-Output-Verzeichnis speichern**
   - ❌ FALSCH: `output_dir / "test_result.json"`
   - ✅ RICHTIG: `output_dir / "data" / "test_result.json"`

2. **ALLE Output-Funktionen MÜSSEN die Struktur verwenden**
   - `_save_artifacts()` → `data/` und `artifacts/`
   - `_generate_visualizations()` → `visualizations/`
   - `save_intermediate_result()` → `data/`
   - `save_config_snapshot()` → `artifacts/`

3. **Output-Struktur wird AUTOMATISCH erstellt**
   - `PipelineCoordinator.process()` erstellt die Struktur automatisch
   - `OutputStructureManager` erstellt die Struktur bei Initialisierung
   - `ensure_output_structure()` kann manuell aufgerufen werden

4. **Backward Compatibility**
   - Alte Code-Pfade werden automatisch erkannt
   - Fehlende Unterordner werden automatisch erstellt
   - Fallback-Verhalten ist vorhanden

## Code-Beispiele

### PipelineCoordinator

```python
# Automatisch in process()
from src.utils.output_structure_manager import ensure_output_structure
ensure_output_structure(Path(final_output_dir))

# In _save_artifacts()
data_dir = output_path / "data"
artifacts_dir = output_path / "artifacts"
data_dir.mkdir(parents=True, exist_ok=True)
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Speichere Dateien in Unterordnern
results_path = data_dir / f"{base_name}_results.json"
html_path = artifacts_dir / f"{base_name}_report.html"
```

### Test-Skripte

```python
# run_live_test.py
from src.utils.output_structure_manager import OutputStructureManager

output_manager = OutputStructureManager(OUTPUT_BASE_DIR.parent, "live_test")
self.output_dir = output_manager.get_output_dir()
self.log_file = output_manager.get_log_file("test.log")

# Verwende strukturierte Pfade
result_file = output_manager.get_data_path("test_result.json")
```

## Migration bestehender Skripte

### Vorher (Unorganisiert):
```python
output_dir = project_root / "outputs" / "live_test" / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

# Dateien werden direkt im Haupt-Ordner gespeichert
log_file = output_dir / "test.log"
result_file = output_dir / "test_result.json"
```

### Nachher (Strukturiert):
```python
from src.utils.output_structure_manager import OutputStructureManager

output_manager = OutputStructureManager(
    project_root / "outputs",
    "live_test"
)

# Dateien werden in Unterordnern gespeichert
log_file = output_manager.get_log_file("test.log")
result_file = output_manager.get_data_path("test_result.json")
```

## Vorteile

1. **Organisiert**: Alle Dateien sind logisch gruppiert
2. **Konsistent**: Gleiche Struktur für alle Test-Typen
3. **Wartbar**: Einfach zu finden und zu analysieren
4. **Skalierbar**: Funktioniert für kleine und große Test-Läufe
5. **Dokumentiert**: README.md erklärt die Struktur

## Enforcement

- **Automatisch**: `PipelineCoordinator` erstellt die Struktur automatisch
- **Zwangsläufig**: Alle Output-Funktionen verwenden die Unterordner
- **Dokumentiert**: Diese README erklärt die Verwendung
- **Zukunftssicher**: Neue Testskripte müssen diese Struktur verwenden

## Troubleshooting

### Problem: Dateien werden im Haupt-Ordner gespeichert
**Lösung**: Stelle sicher, dass `ensure_output_structure()` aufgerufen wird und die Output-Funktionen die Unterordner verwenden.

### Problem: Struktur wird nicht erstellt
**Lösung**: Überprüfe, ob `OutputStructureManager` oder `ensure_output_structure()` verwendet wird.

### Problem: Alte Skripte funktionieren nicht mehr
**Lösung**: Die Struktur wird automatisch erstellt, auch wenn der Code sie nicht explizit verwendet. Alte Skripte sollten weiterhin funktionieren.

