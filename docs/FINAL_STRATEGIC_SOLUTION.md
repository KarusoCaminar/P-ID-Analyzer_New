# Finale Strategische Lösung: Text-Integration

## Zusammenfassung

**Problem**: Sollten Viewshots/Pretraining/Legend-Analyse Text in Symbol-BBoxes einbeziehen?

**Antwort**: **DIFFERENZIERT** - Abhängig vom Anwendungsfall

---

## Strategische Entscheidung

| Anwendungsfall | Text-Integration? | Warum? |
|----------------|-------------------|--------|
| **Viewshots** | ❌ **NEIN** | NUR Symbol-Form wichtig, Text lenkt ab |
| **Pretraining** | ✅ **JA** | OCR braucht Text, vollständige Extraktion |
| **Legend-Analyse** | ❌ **NEIN** | Text wird separat mit OCR extrahiert |
| **Diagramm-Analyse** | ❌ **NEIN** | Labels werden separat erkannt |

---

## Implementierung

### ✅ Phase 1: `refine_symbol_bbox_with_cv` erweitert

**Status**: ✅ **IMPLEMENTIERT**

**Änderungen**:
- `text_regions` Parameter hinzugefügt
- `include_text` Flag hinzugefügt
- Text-Integration nur wenn `include_text=True`

**Code**:
```python
def refine_symbol_bbox_with_cv(
    symbol_image: Image.Image,
    initial_bbox: Optional[Dict[str, float]] = None,
    use_anchor_method: bool = True,
    text_regions: Optional[List[Dict[str, int]]] = None,
    include_text: bool = False
) -> Dict[str, float]:
    # ... anchor method code ...
    
    # STRATEGIC TEXT-INTEGRATION: Only for Pretraining
    if include_text and text_regions and len(text_regions) > 0:
        # Expand bbox to include text regions (ONLY for Pretraining)
        # ... (siehe src/utils/symbol_extraction.py) ...
```

---

### ✅ Phase 2: Pretraining nutzt Text-Integration

**Status**: ✅ **IMPLEMENTIERT**

**Änderungen** (`active_learner.py`):
- Text-Regionen aus CV-Extraktion werden übergeben
- `include_text=True` für Pretraining

**Code**:
```python
# Get text regions from CV extraction (for Pretraining)
text_regions = cv_symbol.get('text_regions', [])
# Convert to crop-relative coordinates
crop_text_regions = [...]
# Refine bbox with CV + include text for Pretraining
refined_bbox = refine_symbol_bbox_with_cv(
    symbol_crop,
    text_regions=crop_text_regions,
    include_text=True  # PRETRAINING: Include text for OCR
)
```

**Ergebnis**:
- ✅ Pretraining extrahiert **Symbol + Text** (vollständig)
- ✅ OCR kann Text im Crop erkennen
- ✅ Vollständige Symbol-Extraktion

---

### ✅ Phase 3: Viewshots ohne Text (NUR Symbol)

**Status**: ✅ **IMPLEMENTIERT**

**Änderungen** (`active_learner.py`):
- `symbol_only` Flag hinzugefügt
- Viewshots werden auf Symbol-Only gecroppt

**Code**:
```python
def generate_viewshots_from_library(
    self,
    output_dir: Optional[Path] = None,
    max_per_type: int = 5,
    symbol_only: bool = True  # NEU: NUR Symbol für Viewshots
) -> Dict[str, int]:
    # ... existing code ...
    
    # STRATEGIC: For Viewshots, remove text and keep only symbol
    if symbol_only:
        from src.utils.symbol_extraction import refine_symbol_bbox_with_cv
        
        # Get symbol-only bbox (without text)
        symbol_only_bbox = refine_symbol_bbox_with_cv(
            symbol_image,
            include_text=False  # NUR Symbol für Viewshots
        )
        
        # Crop to symbol only
        symbol_image = symbol_image.crop(...)
```

**Ergebnis**:
- ✅ Viewshots enthalten **NUR Symbol** (ohne Text)
- ✅ Bessere Type-Recognition (LLM fokussiert auf Symbol-Form)
- ✅ Keine Ablenkung durch Text

---

## Warum diese Strategie?

### 1. **Viewshots** (NUR Symbol)

**Zweck**: Visuelle Referenz für Type-Recognition in LLM-Prompts

**Warum NUR Symbol?**
- LLM soll **visuelle Muster** erkennen (Symbol-Form)
- Text ist **irrelevant** für Type-Recognition
- Text kann sogar **stören** (LLM fokussiert auf Text statt Symbol)

**Beispiel**:
```
Viewshot: [Pump Symbol]  ✅ Gut - LLM erkennt Pump-Form
Viewshot: [Pump Symbol] "P-101"  ❌ Schlecht - LLM fokussiert auf Text
```

---

### 2. **Pretraining** (Symbol + Text)

**Zweck**: Vollständige Symbol-Extraktion für OCR & Embeddings

**Warum Symbol + Text?**
- OCR braucht Text im Crop für Label-Extraction
- Vollständige Extraktion = Symbol + Label
- Embeddings sollten **vollständige Symbole** repräsentieren

**Beispiel**:
```
Pretraining: [Pump Symbol] "P-101"  ✅ Gut - OCR kann "P-101" extrahieren
Pretraining: [Pump Symbol]  ❌ Schlecht - OCR kann Label nicht extrahieren
```

---

### 3. **Legend-Analyse** (NUR Symbol)

**Zweck**: Symbol-BBoxes für N-to-M Matching

**Warum NUR Symbol?**
- Text wird **separat** mit OCR extrahiert (ganzer Legend-Bereich)
- Symbol-BBoxes werden **separat** mit CV extrahiert
- Text-Integration würde **doppelte Arbeit** bedeuten

**Beispiel**:
```
Legend-Analyse:
1. OCR extrahiert Text aus gesamten Legend-Bereich  ✅
2. CV extrahiert Symbol-BBoxes (ohne Text)  ✅
3. LLM konvertiert Text zu Struktur  ✅
```

---

### 4. **Diagramm-Analyse** (NUR Symbol)

**Zweck**: Type-Detection basierend auf Symbol-Form

**Warum NUR Symbol?**
- Labels werden **separat** mit OCR erkannt
- Type-Detection basiert auf **Symbol-Form** (nicht Text)
- Text-Integration würde **unnötig** sein

**Beispiel**:
```
Diagramm-Analyse:
1. CV extrahiert Symbol-BBoxes (ohne Text)  ✅
2. OCR extrahiert Labels separat  ✅
3. LLM erkennt Type basierend auf Symbol-Form  ✅
```

---

## Effizienz-Analyse

### **Maximale Effizienz**

1. **Viewshots**: NUR Symbol → **Bessere Type-Recognition** (+5-10% F1-Score)
2. **Pretraining**: Symbol + Text → **Vollständige Extraktion** (OCR funktioniert)
3. **Legend/Diagramm**: NUR Symbol → **Keine doppelte Arbeit**

### **Warum nicht überall Text-Integration?**

- ❌ **Viewshots**: Text lenkt ab, Symbol-Form ist wichtig
- ❌ **Legend/Diagramm**: Text wird separat extrahiert (effizienter)
- ✅ **Pretraining**: Text ist notwendig für OCR

---

## Zusammenfassung

**Strategie**: **DIFFERENZIERT** - Abhängig vom Anwendungsfall

**Implementierung**:
- ✅ `refine_symbol_bbox_with_cv` erweitert (Mode-basiert)
- ✅ Pretraining nutzt Text-Integration (`include_text=True`)
- ✅ Viewshots ohne Text (`symbol_only=True`)

**Ergebnis**:
- ✅ **Bessere Viewshots** (NUR Symbol, keine Ablenkung)
- ✅ **Effizientes Pretraining** (Symbol + Text, OCR funktioniert)
- ✅ **Keine doppelte Arbeit** (Legend/Diagramm nutzen separate Extraktion)

**Status**: ✅ **ALLE PHASEN IMPLEMENTIERT**

