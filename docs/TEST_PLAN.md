# 🧪 P&ID Analyzer v2.0 - Test Plan

**Datum:** 2025-11-07  
**Status:** 📋 In Bearbeitung

---

## 📋 Übersicht

Dieses Dokument beschreibt den Test-Plan für den P&ID Analyzer v2.0. Es identifiziert fehlende Tests, priorisiert Test-Implementierungen und definiert Test-Strategien für alle kritischen Komponenten.

---

## ✅ Bereits vorhandene Tests

### Unit Tests (`tests/unit/`)
- ✅ `test_cache_service.py` - Cache Service Tests
- ✅ `test_error_handler.py` - Error Handler Tests
- ✅ `test_graph_utils.py` - Graph Utilities Tests
- ✅ `test_image_utils.py` - Image Utilities Tests
- ✅ `test_llm_client.py` - LLM Client Tests

### Integration Tests (`tests/`)
- ✅ `test_imports.py` - Module Import Tests
- ✅ `test_config_service.py` - Config Service Tests
- ✅ `test_integration.py` - Pipeline Initialization Tests
- ✅ `test_api_robustness.py` - API Robustness Tests
- ✅ `test_utils.py` - Utility Tests

---

## ❌ Fehlende Tests (Priorität: HOCH)

### 1. **Pipeline Coordinator Tests** (`tests/unit/test_pipeline_coordinator.py`)
**Priorität:** 🔴 KRITISCH  
**Komplexität:** Hoch  
**Geschätzte Zeit:** 4-6 Stunden

**Zu testende Funktionen:**
- `process()` - Haupt-Pipeline-Orchestrierung
- `_run_phase_0_complexity_analysis()` - Complexity Analysis
- `_run_phase_1_pre_analysis()` - Pre-Analysis (Metadata, Legend)
- `_run_phase_2a_swarm_analysis()` - Swarm Analysis
- `_run_phase_2c_monolith_analysis()` - Monolith Analysis
- `_run_phase_2d_fusion()` - Fusion Engine
- `_run_phase_2d_predictive_completion()` - Predictive Completion
- `_run_phase_2e_polyline_refinement()` - Polyline Refinement
- `_run_phase_3_self_correction()` - Self-Correction Loop
- `_run_phase_4_post_processing()` - Post-Processing
- `_validate_connection_semantics()` - Connection Semantics Validation
- `_detect_ports()` - Port Detection
- `_generate_cgm_data()` - CGM Generation

**Test-Strategie:**
- Mock LLM Client und Knowledge Manager
- Verwende Test-Bilder aus `training_data/simple_pids/`
- Validiere Output-Struktur (elements, connections, KPIs)
- Teste Edge Cases (leere Bilder, fehlende Elemente, etc.)

**Beispiel-Test:**
```python
def test_pipeline_coordinator_process_simple_pid():
    """Test pipeline coordinator with simple P&ID."""
    # Setup
    coordinator = PipelineCoordinator(...)
    test_image = "training_data/simple_pids/Einfaches P&I.png"
    
    # Execute
    result = coordinator.process(image_path=test_image)
    
    # Assert
    assert result.elements is not None
    assert len(result.elements) > 0
    assert result.connections is not None
    assert result.quality_score >= 0.0
```

---

### 2. **Swarm Analyzer Tests** (`tests/unit/test_swarm_analyzer.py`)
**Priorität:** 🔴 KRITISCH  
**Komplexität:** Mittel  
**Geschätzte Zeit:** 3-4 Stunden

**Zu testende Funktionen:**
- `analyze()` - Haupt-Analyse-Methode
- `_create_tiles()` - Tile-Erstellung
- `_analyze_tile()` - Einzelne Tile-Analyse
- `_merge_results()` - Ergebnis-Merging
- `_load_viewshot_examples()` - Viewshot Loading

**Test-Strategie:**
- Mock LLM Client
- Teste mit verschiedenen Tile-Größen
- Validiere Element-Erkennung (Typ, BBox, Confidence)
- Teste Edge Cases (leere Kacheln, Überschneidungen)

---

### 3. **Monolith Analyzer Tests** (`tests/unit/test_monolith_analyzer.py`)
**Priorität:** 🔴 KRITISCH  
**Komplexität:** Mittel  
**Geschätzte Zeit:** 3-4 Stunden

**Zu testende Funktionen:**
- `analyze()` - Haupt-Analyse-Methode
- `_analyze_whole_image()` - Whole-Image-Analyse
- `_analyze_quadrants()` - Quadrant-Analyse
- `_merge_quadrant_results()` - Quadrant-Merging

**Test-Strategie:**
- Mock LLM Client
- Teste mit verschiedenen Bildgrößen
- Validiere Connection-Erkennung (from_id, to_id, ports)
- Teste Edge Cases (fehlende Elemente, falsche IDs)

---

### 4. **Line Extractor Tests** (`tests/unit/test_line_extractor.py`)
**Priorität:** 🟠 HOCH  
**Komplexität:** Mittel  
**Geschätzte Zeit:** 2-3 Stunden

**Zu testende Funktionen:**
- `extract_pipeline_lines()` - Haupt-Extraktions-Methode
- `_mask_symbols()` - Symbol-Masking
- `_extract_pipeline_colors()` - Color Extraction
- `_extract_contours()` - Contour Extraction
- `_detect_junctions_from_contours()` - Junction Detection
- `_match_to_connections()` - Connection Matching

**Test-Strategie:**
- Verwende Test-Bilder mit bekannten Linien
- Validiere Polylinien-Koordinaten
- Teste Junction-Erkennung
- Teste Edge Cases (keine Linien, überlappende Linien)

---

### 5. **Fusion Engine Tests** (`tests/unit/test_fusion_engine.py`)
**Priorität:** 🟠 HOCH  
**Komplexität:** Mittel  
**Geschätzte Zeit:** 2-3 Stunden

**Zu testende Funktionen:**
- `fuse()` - Haupt-Fusion-Methode
- `_deduplicate_elements()` - Element-Deduplizierung
- `_merge_elements()` - Element-Merging
- `_propagate_confidence()` - Confidence-Propagation

**Test-Strategie:**
- Erstelle Mock Swarm- und Monolith-Ergebnisse
- Validiere Deduplizierung (IoU-basiert)
- Teste Confidence-Propagation
- Teste Edge Cases (keine Überschneidungen, vollständige Überschneidungen)

---

### 6. **KPI Calculator Tests** (`tests/unit/test_kpi_calculator.py`)
**Priorität:** 🟠 HOCH  
**Komplexität:** Mittel  
**Geschätzte Zeit:** 2-3 Stunden

**Zu testende Funktionen:**
- `calculate_comprehensive_kpis()` - Haupt-KPI-Berechnung
- `_calculate_element_metrics()` - Element-Metriken
- `_calculate_connection_metrics()` - Connection-Metriken
- `_calculate_quality_score()` - Quality Score

**Test-Strategie:**
- Verwende Ground Truth-Daten aus `training_data/simple_pids/`
- Validiere Precision, Recall, F1-Score
- Teste Quality Score-Berechnung
- Teste Edge Cases (leere Ergebnisse, perfekte Übereinstimmung)

---

### 7. **Knowledge Manager Tests** (`tests/unit/test_knowledge_manager.py`)
**Priorität:** 🟡 MITTEL  
**Komplexität:** Hoch  
**Geschätzte Zeit:** 3-4 Stunden

**Zu testende Funktionen:**
- `find_similar_symbols()` - Symbol-Similarity-Search
- `find_similar_solutions()` - Solution-Similarity-Search
- `add_correction()` - Correction-Adding
- `bulk_add_corrections()` - Bulk-Correction-Adding
- `_lazy_load_learning_database()` - Lazy Loading

**Test-Strategie:**
- Mock Learning Database
- Teste Similarity-Search mit bekannten Symbolen
- Validiere Lazy Loading (große DBs)
- Teste Vector-Index-Loading

---

### 8. **CGM Generator Tests** (`tests/unit/test_cgm_generator.py`)
**Priorität:** 🟡 MITTEL  
**Komplexität:** Niedrig  
**Geschätzte Zeit:** 1-2 Stunden

**Zu testende Funktionen:**
- `generate_cgm_data()` - CGM-Daten-Generierung
- `generate_cgm_network()` - CGM-Network-Generierung
- `_generate_network_instance()` - Network-Instance-Generierung

**Test-Strategie:**
- Verwende Mock Elements und Connections
- Validiere JSON-Struktur
- Validiere Python-Code-Generierung
- Teste Edge Cases (leere Elemente/Connections)

---

### 9. **Complexity Analyzer Tests** (`tests/unit/test_complexity_analyzer.py`)
**Priorität:** 🟡 MITTEL  
**Komplexität:** Niedrig  
**Geschätzte Zeit:** 1-2 Stunden

**Zu testende Funktionen:**
- `analyze_complexity_cv_advanced()` - CV-basierte Komplexitäts-Analyse
- `_skeletonize()` - Skeletonization (für Junction-Detection)
- `_detect_junctions()` - Junction-Detection

**Test-Strategie:**
- Verwende Test-Bilder mit bekannter Komplexität
- Validiere Komplexitäts-Kategorisierung (simple, moderate, complex, very_complex)
- Teste Junction-Detection

---

### 10. **Active Learner Tests** (`tests/unit/test_active_learner.py`)
**Priorität:** 🟡 MITTEL  
**Komplexität:** Mittel  
**Geschätzte Zeit:** 2-3 Stunden

**Zu testende Funktionen:**
- `learn_from_analysis_result()` - Learning aus Analyse-Ergebnissen
- `get_recommendations()` - Empfehlungen abrufen
- `save_learned_patterns()` - Gelernte Patterns speichern

**Test-Strategie:**
- Mock Knowledge Manager
- Teste Pattern-Learning
- Validiere Empfehlungen

---

## 📊 Test-Coverage-Ziele

### Aktuelle Coverage
- **Unit Tests:** ~30% (geschätzt)
- **Integration Tests:** ~20% (geschätzt)
- **Gesamt:** ~25% (geschätzt)

### Ziel-Coverage
- **Unit Tests:** >80%
- **Integration Tests:** >60%
- **Gesamt:** >75%

---

## 🚀 Implementierungs-Strategie

### Phase 1: Kritische Komponenten (Woche 1)
1. Pipeline Coordinator Tests
2. Swarm Analyzer Tests
3. Monolith Analyzer Tests

### Phase 2: Wichtige Komponenten (Woche 2)
4. Line Extractor Tests
5. Fusion Engine Tests
6. KPI Calculator Tests

### Phase 3: Unterstützende Komponenten (Woche 3)
7. Knowledge Manager Tests
8. CGM Generator Tests
9. Complexity Analyzer Tests
10. Active Learner Tests

---

## 📝 Test-Best-Practices

### 1. Mocking
- **LLM Client:** Mock alle LLM-Aufrufe für schnellere Tests
- **Knowledge Manager:** Mock für Unit Tests, verwende echte DB für Integration Tests
- **File I/O:** Mock Datei-Zugriffe wo möglich

### 2. Test-Daten
- **Test-Bilder:** Verwende `training_data/simple_pids/` und `training_data/complex_pids/`
- **Ground Truth:** Verwende `*_truth.json` Dateien für Validierung
- **Mock-Daten:** Erstelle wiederverwendbare Mock-Daten-Factory

### 3. Test-Organisation
- **Unit Tests:** Ein Test pro Funktion
- **Integration Tests:** Ein Test pro Pipeline-Phase
- **Fixture-Organisation:** Verwende `conftest.py` für gemeinsame Fixtures

### 4. Test-Ausführung
```bash
# Alle Tests ausführen
pytest tests/

# Mit Coverage
pytest tests/ --cov=src --cov-report=html

# Spezifische Tests
pytest tests/unit/test_pipeline_coordinator.py

# Verbose Output
pytest tests/ -v
```

---

## 🔍 Test-Debugging

### Häufige Probleme
1. **LLM Client Timeouts:** Mock LLM Client in Unit Tests
2. **File Path Issues:** Verwende `pathlib.Path` für plattform-unabhängige Pfade
3. **GCP Credentials:** Mock GCP-Aufrufe in Unit Tests

### Debug-Tools
- `pytest --pdb` - Startet Debugger bei Fehlern
- `pytest --verbose` - Detailliertes Output
- `pytest --cov-report=term-missing` - Zeigt fehlende Coverage

---

## 📚 Ressourcen

### Dokumentation
- [pytest Dokumentation](https://docs.pytest.org/)
- [pytest-mock Dokumentation](https://pytest-mock.readthedocs.io/)
- [coverage.py Dokumentation](https://coverage.readthedocs.io/)

### Test-Beispiele
- Siehe `tests/unit/test_llm_client.py` für LLM Client Mocking
- Siehe `tests/integration/test_integration.py` für Pipeline-Tests

---

## ✅ Erfolgs-Kriterien

- [ ] Alle kritischen Komponenten haben Unit Tests
- [ ] Test-Coverage >75%
- [ ] Alle Tests laufen erfolgreich
- [ ] CI/CD-Pipeline führt Tests automatisch aus
- [ ] Tests werden bei jedem Commit ausgeführt

---

**Status:** 📋 Test-Plan erstellt, Implementierung ausstehend

**Nächste Schritte:**
1. Implementiere Pipeline Coordinator Tests
2. Implementiere Swarm Analyzer Tests
3. Implementiere Monolith Analyzer Tests
4. Setup CI/CD-Pipeline (GitHub Actions)

