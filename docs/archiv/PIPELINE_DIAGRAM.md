# P&ID Analyzer Pipeline - Flow Diagram

## Aktuelle Pipeline-Struktur (mit Self-Correction Loop)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         P&ID ANALYSIS PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

START
 │
 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Pre-Analysis                                                   │
│ ─────────────────────────────────────────────────────────────────────   │
│ • Metadata Extraction (Title Block)                                     │
│ • Legend Extraction (Symbols & Line Rules)                              │
│ • BBox Validation (Black Rectangle Detection)                            │
│ • Exclusion Zones (Metadata/Legend Areas)                                │
│                                                                          │
│ Output: Global Knowledge Repository (Symbol Map, Line Map)              │
└─────────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Parallel Core Analysis                                         │
│ ─────────────────────────────────────────────────────────────────────   │
│                                                                          │
│  ┌──────────────────────────┐      ┌──────────────────────────┐        │
│  │  SWARM ANALYZER          │      │  MONOLITH ANALYZER        │        │
│  │  (Component Detection)   │      │  (Structure Detection)    │        │
│  │                          │      │                          │        │
│  │  • Tiled Analysis        │      │  • Quadrant Analysis      │        │
│  │  • 50-80 Tiles          │      │  • 4 Quadrants            │        │
│  │  • Parallel Processing  │      │  • Whole Image (<3000px)  │        │
│  │  • Symbol Library Match │      │  • Structural Context    │        │
│  │  • Viewshot Examples    │      │  • Viewshot Examples      │        │
│  │  • CV Text Detection    │      │  • Symbol Library Match   │        │
│  │                          │      │  • CV Text Detection     │        │
│  │  Model: swarm_model     │      │  Model: monolith_model    │        │
│  │  (Flash/Lite/Pro)       │      │  (Flash/Pro)             │        │
│  └──────────────────────────┘      └──────────────────────────┘        │
│           │                                    │                          │
│           └──────────────┬────────────────────┘                          │
│                          ▼                                                  │
│  ┌──────────────────────────────────────────────────────────┐            │
│  │  PHASE 2c: FUSION ENGINE                                  │            │
│  │  ──────────────────────────────────────────────────────  │            │
│  │  • Quality-Aware Merging                                  │            │
│  │  • IoU Matching (Threshold: 0.3)                         │            │
│  │  • Element Type/Label Correction                          │            │
│  │  • Confidence Boosting                                    │            │
│  │  • Connection Merging                                     │            │
│  │                                                           │            │
│  │  Strategy:                                                │            │
│  │  - Consensus (Type+Label+High IoU) → Combine & Boost     │            │
│  │  - Type Match → Combine & Boost                           │            │
│  │  - Monolith Better (≥30%) → Overwrite                     │            │
│  │  - Swarm Better/Equal → Keep Swarm                        │            │
│  │  - Low Quality → Reject                                   │            │
│  └──────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2d: Predictive Completion (Optional)                              │
│ ─────────────────────────────────────────────────────────────────────   │
│ • Heuristic Gap Closing                                                 │
│ • Geometric Distance-Based Connection Prediction                         │
│ • Fills Missing Connections Between Nearby Elements                     │
│                                                                          │
│ Flag: use_predictive_completion                                         │
└─────────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2e: Polyline Refinement (Optional)                                │
│ ─────────────────────────────────────────────────────────────────────   │
│ • Skeleton-Based Line Extraction OR                                      │
│ • LLM-Based Polyline Extraction                                          │
│ • Precisely Extracts Line Paths for Each Connection                      │
│                                                                          │
│ Flags: use_polyline_refinement, use_skeleton_line_extraction            │
└─────────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Self-Correction Loop (Iterative)                               │
│ ─────────────────────────────────────────────────────────────────────   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  ITERATION LOOP (max 15 iterations)                              │   │
│  │  ──────────────────────────────────────────────────────────────  │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  Validation & Critics                                    │   │   │
│  │  │  ──────────────────────────────────────────────────────  │   │   │
│  │  │  • Topology Critic (Graph Consistency)                   │   │   │
│  │  │  • Legend Consistency Critic                              │   │   │
│  │  │  • Multi-Model Critic (Cross-Validation)                  │   │   │
│  │  │  • Error Explanation (LLM-based)                         │   │   │
│  │  │  • Quality Score Calculation                             │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │           │                                                       │   │
│  │           ▼                                                       │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  Early Stop Check                                        │   │   │
│  │  │  ──────────────────────────────────────────────────────  │   │   │
│  │  │  • Quality Score ≥ 98% → STOP                            │   │   │
│  │  │  • All Elements Confidence ≥ 0.9 → STOP                  │   │   │
│  │  │  • Plateau Detection (no improvement) → STOP             │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │           │                                                       │   │
│  │           ▼ (if not stopping)                                     │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  Targeted Re-Analysis OR Whole Image Re-Analysis          │   │   │
│  │  │  ──────────────────────────────────────────────────────  │   │   │
│  │  │                                                           │   │   │
│  │  │  Option A: Targeted Re-Analysis (default)                │   │   │
│  │  │  • Identify Uncertain Zones (confidence < 0.7)           │   │   │
│  │  │  • Generate Targeted Tiles (512px, high precision)       │   │   │
│  │  │  • Re-analyze ONLY uncertain zones                       │   │   │
│  │  │  • Parallel: Swarm + Monolith                             │   │   │
│  │  │  • Fusion Engine merges results                           │   │   │
│  │  │                                                           │   │   │
│  │  │  Option B: Whole Image Re-Analysis (fallback)            │   │   │
│  │  │  • Re-analyze entire image                                │   │   │
│  │  │  • Parallel: Swarm + Monolith                             │   │   │
│  │  │  • Fusion Engine merges results                           │   │   │
│  │  │                                                           │   │   │
│  │  │  Features:                                                │   │   │
│  │  │  • Error Feedback (from previous iteration)               │   │   │
│  │  │  • Legend Context (from Phase 1)                         │   │   │
│  │  │  • Symbol Library (learning from errors)                 │   │   │
│  │  │  • Viewshot Examples                                      │   │   │
│  │  │  • CV Text Detection                                      │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │           │                                                       │   │
│  │           ▼                                                       │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  Merge Results                                            │   │   │
│  │  │  ──────────────────────────────────────────────────────  │   │   │
│  │  │  • Remove Hallucinated Elements                           │   │   │
│  │  │  • Add New Elements (from re-analysis)                    │   │   │
│  │  │  • Update Existing Elements (improved confidence)         │   │   │
│  │  │  • Merge Connections                                      │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │           │                                                       │   │
│  │           ▼                                                       │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  BBox Refinement (Iterative)                             │   │   │
│  │  │  ──────────────────────────────────────────────────────  │   │   │
│  │  │  • Cascade BBox Regression                                │   │   │
│  │  │  • Priority: Low-Confidence Elements                      │   │   │
│  │  │  • CV-Based Anchor Method (centers symbols)              │   │   │
│  │  │  • Iterative IoU Improvement                              │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │           │                                                       │   │
│  │           ▼                                                       │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  Loop Back to Validation                                 │   │   │
│  │  │  (until max iterations or early stop)                     │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │           │                                                       │   │
│  │           └───────────────────────────────────────────────────────┘   │
│  │                     (repeat until stop condition)                    │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ Flag: use_self_correction_loop                                          │
└─────────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Post-Processing                                                │
│ ─────────────────────────────────────────────────────────────────────   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: Legend Symbol Matching (Optional)                      │   │
│  │  ──────────────────────────────────────────────────────────────  │   │
│  │  • Visual Similarity (Embeddings)                               │   │
│  │  • Type/Label Validation                                         │   │
│  │  • Match Diagram Symbols with Legend Symbols                     │   │
│  │                                                                  │   │
│  │  Flag: use_legend_matching                                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: Context-Aware Type Inference (Optional)                │   │
│  │  ──────────────────────────────────────────────────────────────  │   │
│  │  • P&ID Naming Conventions                                       │   │
│  │  • Label-Based Type Inference                                    │   │
│  │  • Pattern Matching (MV, PU, FT, etc.)                          │   │
│  │                                                                  │   │
│  │  Flag: use_context_type_inference                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: LLM-based ID Correction (Optional)                     │   │
│  │  ──────────────────────────────────────────────────────────────  │   │
│  │  • Semantic ID Correction                                        │   │
│  │  • Alias Resolution ("Valve" vs "valve")                        │   │
│  │  • Type Consistency                                              │   │
│  │                                                                  │   │
│  │  Flag: use_llm_id_correction                                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 4: Type Validation (Optional)                              │   │
│  │  ──────────────────────────────────────────────────────────────  │   │
│  │  • Validate against Known Types List                             │   │
│  │  • Synonym Correction                                            │   │
│  │  • Never Remove if confidence >= 0.5                             │   │
│  │  • Mark as "Unknown" if needed                                   │   │
│  │                                                                  │   │
│  │  Flag: use_type_validation                                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 5: Confidence Filtering (Two-Stage)                       │   │
│  │  ──────────────────────────────────────────────────────────────  │   │
│  │  • Stage 1: confidence >= 0.5 → Keep                             │   │
│  │  • Stage 2: confidence 0.3-0.5 → Keep with Penalty (0.8x)      │   │
│  │  • Stage 3: confidence < 0.3 → Remove (only if very low)       │   │
│  │                                                                  │   │
│  │  Flag: use_confidence_filtering                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 6: Chain-of-Thought Reasoning (Optional)                  │   │
│  │  ──────────────────────────────────────────────────────────────  │   │
│  │  • Validate Connections with LLM                                │   │
│  │  • Complete Missing Connections                                 │   │
│  │  • P&ID Domain Knowledge                                        │   │
│  │                                                                  │   │
│  │  Flag: use_cot_reasoning                                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Step 7: CV BBox Refinement (Optional)                          │   │
│  │  ──────────────────────────────────────────────────────────────  │   │
│  │  • Anchor-Based Symbol Centering                                │   │
│  │  • Contour Detection                                             │   │
│  │  • Precise Symbol Cropping                                       │   │
│  │                                                                  │   │
│  │  Flag: use_cv_bbox_refinement                                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Final Output & KPIs                                            │
│ ─────────────────────────────────────────────────────────────────────   │
│ • KPI Calculation (Precision, Recall, F1-Score, Structural KPIs)        │
│ • CGM Generation (Abstract Graph)                                        │
│ • Visualization (Confidence Map, KPI Dashboard, Score Curve)            │
│ • Reports (HTML, JSON)                                                   │
│ • Artifacts (Polylines, Debug Maps, etc.)                                │
└─────────────────────────────────────────────────────────────────────────┘
 │
 ▼
END


═══════════════════════════════════════════════════════════════════════════

## Wichtige Flags & Settings

### Analysis Flags
- `use_swarm_analysis`: true/false - Swarm-Analyse ein/aus
- `use_monolith_analysis`: true/false - Monolith-Analyse ein/aus
- `use_fusion`: true/false - Fusion ein/aus

### Self-Correction Loop
- `use_self_correction_loop`: true/false - Loop ein/aus
- `max_self_correction_iterations`: 15 - Max Iterationen
- `target_quality_score`: 98.0 - Early Stop bei Score
- `max_no_improvement_iterations`: 3 - Plateau-Erkennung
- `use_targeted_reanalysis`: true - Targeted vs Whole Image

### Post-Processing Flags
- `use_type_validation`: true/false - Type-Validierung
- `use_confidence_filtering`: true/false - Confidence-Filterung
- `use_predictive_completion`: true/false - Predictive Completion
- `use_polyline_refinement`: true/false - Polyline-Extraktion
- `use_cv_bbox_refinement`: true/false - CV BBox Refinement
- `use_legend_matching`: true/false - Legend Matching
- `use_llm_id_correction`: true/false - LLM ID Correction
- `use_context_type_inference`: true/false - Type Inference
- `use_cot_reasoning`: true/false - CoT Reasoning

### Thresholds
- `confidence_threshold`: 0.5 - Confidence-Schwelle
- `iou_match_threshold`: 0.3 - IoU-Schwelle für Fusion
- `low_confidence_threshold`: 0.7 - Schwelle für Uncertain Zones

═══════════════════════════════════════════════════════════════════════════

## Geschätzter Zeitaufwand (Simple PID)

### Phase 1: Pre-Analysis
- **Zeit**: ~5-10 Sekunden
- Metadata Extraction, Legend Extraction

### Phase 2: Parallel Core Analysis
- **Zeit**: ~2-5 Minuten (abhängig von Strategie)
- Swarm: ~2-4 Minuten (28 Tiles, parallel)
- Monolith: ~30-60 Sekunden (whole image)
- Fusion: ~1-2 Sekunden

### Phase 2d: Predictive Completion
- **Zeit**: ~1-2 Sekunden
- Heuristic Gap Closing

### Phase 2e: Polyline Refinement
- **Zeit**: ~1-2 Minuten
- Polyline Extraction für alle Connections

### Phase 3: Self-Correction Loop
- **Zeit**: ~3-10 Minuten (abhängig von Iterationen)
- Pro Iteration: ~1-2 Minuten
- Typisch: 2-5 Iterationen für Simple PID

### Phase 4: Post-Processing
- **Zeit**: ~10-30 Sekunden
- Alle Post-Processing Steps

### Phase 5: Final Output
- **Zeit**: ~5-10 Sekunden
- KPI Calculation, Visualization

### **GESAMT (Simple PID)**
- **Minimal** (ohne Loop, Flash-only): ~4-6 Minuten
- **Typisch** (mit Loop, Flash-only): ~8-12 Minuten
- **Optimal** (mit Loop, Flash+Pro): ~10-15 Minuten
- **Ultra-Fast** (Flash-Lite, ohne Loop): ~3-5 Minuten

═══════════════════════════════════════════════════════════════════════════

## Strategie-Vergleich (Simple PID)

| Strategie | Swarm | Monolith | Geschwindigkeit | Erwartete Zeit |
|-----------|-------|----------|-----------------|----------------|
| `all_flash` | Flash | Flash | Hoch | ~8-12 Min |
| `optimal_swarm_monolith` | Flash | Pro | Mittel | ~10-15 Min |
| `optimal_swarm_monolith_lite` | Flash-Lite | Flash | Sehr Hoch | ~6-9 Min |

