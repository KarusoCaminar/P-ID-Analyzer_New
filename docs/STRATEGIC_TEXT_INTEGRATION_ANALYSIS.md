# Strategische Text-Integration: Differenzierte Analyse

## Anwendungsfälle & Anforderungen

### 1. **Viewshots** (für LLM-Prompts)

**Zweck**: Visuelle Referenz für Type-Recognition in Diagramm-Analyse

**Aktuelle Verwendung**:
- Viewshots werden in LLM-Prompts eingebettet
- Prompt: "Use these visual patterns to identify similar symbols"
- **Fokus**: Symbol-Form, nicht Text

**Anforderung**:
- ✅ **NUR Symbol** (ohne Text)
- ❌ **KEIN Text** (Text lenkt ab, Symbol-Form ist wichtig)
- ✅ **Saubere Symbol-Extraktion** (Anchor-Methode ohne Text)

**Warum?**
- LLM soll **visuelle Muster** erkennen (Symbol-Form)
- Text ist **irrelevant** für Type-Recognition
- Text kann sogar **stören** (LLM fokussiert auf Text statt Symbol)

---

### 2. **Pretraining** (für Symbol-Library)

**Zweck**: Lernen von Symbolen für Embeddings & Ähnlichkeitssuche

**Aktuelle Verwendung**:
- Symbole werden in Symbol-Library gespeichert
- Embeddings werden generiert für Ähnlichkeitssuche
- **Fokus**: Vollständige Symbol-Extraktion (Symbol + Label)

**Anforderung**:
- ✅ **Symbol + Text** (für vollständige Extraktion)
- ✅ **Text-Integration** (OCR braucht Text im Crop)
- ✅ **Vollständige BBox** (Symbol + Label)

**Warum?**
- OCR braucht Text im Crop für Label-Extraction
- Vollständige Extraktion = Symbol + Label
- Embeddings sollten **vollständige Symbole** repräsentieren

---

### 3. **Legend-Analyse** (OCR + Symbol-Detection)

**Zweck**: Strukturierte Legend-Daten (symbol_map, line_map)

**Aktuelle Verwendung**:
- OCR extrahiert Text aus Legend-Bereich
- LLM konvertiert Text zu Struktur (symbol_map)
- CV extrahiert Symbol-BBoxes aus Legend-Bereich
- **Fokus**: Text-Extraktion (OCR) + Symbol-Detection (CV)

**Anforderung**:
- ❌ **KEINE Text-Integration** (Text wird separat mit OCR extrahiert)
- ✅ **NUR Symbol-BBoxes** (für N-to-M Matching)
- ✅ **Separate Text-Extraktion** (OCR auf gesamten Legend-Bereich)

**Warum?**
- Text wird **separat** mit OCR extrahiert (ganzer Legend-Bereich)
- Symbol-BBoxes werden **separat** mit CV extrahiert
- Text-Integration würde **doppelte Arbeit** bedeuten

---

### 4. **Diagramm-Analyse** (Type-Detection)

**Zweck**: Type-Detection basierend auf Symbol-Form

**Aktuelle Verwendung**:
- Labels werden **separat** erkannt (OCR auf Diagramm)
- Type-Detection basiert auf **Symbol-Form** (LLM + Viewshots)
- **Fokus**: Symbol-Form, nicht Text

**Anforderung**:
- ❌ **KEINE Text-Integration** (Labels werden separat erkannt)
- ✅ **NUR Symbol-BBoxes** (für Type-Detection)
- ✅ **Separate Label-Extraktion** (OCR auf Diagramm)

**Warum?**
- Labels werden **separat** mit OCR erkannt
- Type-Detection basiert auf **Symbol-Form** (nicht Text)
- Text-Integration würde **unnötig** sein

---

## Strategische Lösung: Differenzierte Behandlung

### **Option 1: Mode-basierte Text-Integration**

**Konzept**: Text-Integration nur für Pretraining, nicht für Viewshots/Legend/Diagramm

**Implementierung**:
```python
def refine_symbol_bbox_with_cv(
    symbol_image: Image.Image,
    initial_bbox: Optional[Dict[str, float]] = None,
    use_anchor_method: bool = True,
    include_text: bool = False  # NEU: Mode-Flag
) -> Dict[str, float]:
    """
    Refine symbol bounding box using Computer Vision with anchor-based centering.
    
    Args:
        include_text: If True, include nearby text regions in bbox (for Pretraining)
                     If False, only symbol (for Viewshots/Legend/Diagramm)
    """
    # ... existing anchor method code ...
    
    # AFTER anchor-based centering:
    if include_text and text_regions and len(text_regions) > 0:
        # Expand bbox to include text regions (ONLY for Pretraining)
        # ... (siehe ANCHOR_METHOD_TEXT_INTEGRATION_ANALYSIS.md) ...
    
    # ... rest of code ...
```

**Verwendung**:
- **Pretraining**: `include_text=True` (Symbol + Text)
- **Viewshots**: `include_text=False` (NUR Symbol)
- **Legend**: `include_text=False` (Text wird separat extrahiert)
- **Diagramm**: `include_text=False` (Labels werden separat erkannt)

---

### **Option 2: Separate Funktionen**

**Konzept**: Separate Funktionen für verschiedene Anwendungsfälle

**Implementierung**:
```python
# Für Viewshots: NUR Symbol
def refine_symbol_bbox_for_viewshot(symbol_image, ...):
    return refine_symbol_bbox_with_cv(symbol_image, include_text=False)

# Für Pretraining: Symbol + Text
def refine_symbol_bbox_for_pretraining(symbol_image, text_regions, ...):
    return refine_symbol_bbox_with_cv(symbol_image, include_text=True, text_regions=text_regions)
```

**Vorteile**:
- ✅ Klare Trennung der Anwendungsfälle
- ✅ Keine Verwirrung über Mode-Flags
- ✅ Explizite Funktionen für jeden Zweck

---

## Empfehlung: **Option 1 (Mode-basierte Text-Integration)**

### Warum Option 1?

1. **Flexibilität**: Eine Funktion für alle Anwendungsfälle
2. **Klarheit**: Mode-Flag macht Absicht explizit
3. **Wartbarkeit**: Weniger Code-Duplikation
4. **Rückwärtskompatibel**: Default `include_text=False` (wie bisher)

### Implementierung

**Schritt 1**: Erweitere `refine_symbol_bbox_with_cv` um `include_text` Flag

**Schritt 2**: Nutze `include_text=True` nur für Pretraining

**Schritt 3**: Nutze `include_text=False` für Viewshots/Legend/Diagramm

---

## Zusammenfassung: Was macht Sinn?

| Anwendungsfall | Text-Integration? | Warum? |
|----------------|-------------------|--------|
| **Viewshots** | ❌ **NEIN** | NUR Symbol-Form wichtig, Text lenkt ab |
| **Pretraining** | ✅ **JA** | OCR braucht Text, vollständige Extraktion |
| **Legend-Analyse** | ❌ **NEIN** | Text wird separat mit OCR extrahiert |
| **Diagramm-Analyse** | ❌ **NEIN** | Labels werden separat erkannt |

---

## Fazit

**Für Viewshots**: **NUR Symbol** (ohne Text) - Symbol-Form ist wichtig

**Für Pretraining**: **Symbol + Text** - OCR braucht Text, vollständige Extraktion

**Für Legend/Diagramm**: **NUR Symbol** - Text wird separat extrahiert

**Implementierung**: Mode-basierte Text-Integration (`include_text` Flag)

