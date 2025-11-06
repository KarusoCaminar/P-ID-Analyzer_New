# Strategische Implementierung: Text-Integration

## Implementierungsplan

### Phase 1: Erweitere `refine_symbol_bbox_with_cv` ✅

**Status**: ✅ Implementiert

**Änderungen**:
- `text_regions` Parameter hinzugefügt
- `include_text` Flag hinzugefügt
- Text-Integration nur wenn `include_text=True`

**Verwendung**:
- **Pretraining**: `include_text=True` (Symbol + Text)
- **Viewshots**: `include_text=False` (NUR Symbol)
- **Legend/Diagramm**: `include_text=False` (Text wird separat extrahiert)

---

### Phase 2: Nutze Text-Integration für Pretraining ✅

**Status**: ✅ Implementiert

**Änderungen** (`active_learner.py`):
- Text-Regionen aus CV-Extraktion werden übergeben
- `include_text=True` für Pretraining

**Ergebnis**:
- Pretraining extrahiert **Symbol + Text** (vollständig)
- OCR kann Text im Crop erkennen
- Vollständige Symbol-Extraktion

---

### Phase 3: Viewshots ohne Text (NUR Symbol)

**Status**: ⚠️ **MUSS IMPLEMENTIERT WERDEN**

**Problem**: Viewshots werden aktuell aus Symbol-Library generiert, die **Symbol + Text** enthält

**Lösung**: Viewshots sollten **NUR Symbol** enthalten (ohne Text)

**Optionen**:

#### Option A: Separate Viewshot-Generierung (Symbol ohne Text)

**Konzept**: Beim Generieren von Viewshots, entferne Text-Regionen

**Implementierung**:
```python
def generate_viewshots_from_library(
    self,
    output_dir: Optional[Path] = None,
    max_per_type: int = 5,
    symbol_only: bool = True  # NEU: NUR Symbol für Viewshots
) -> Dict[str, int]:
    """
    Generate viewshots from symbol library.
    
    Args:
        symbol_only: If True, crop to symbol only (remove text) - for Viewshots
                    If False, keep symbol + text - for Pretraining
    """
    # ... existing code ...
    
    # Load symbol image
    image_path = Path(symbol_data.get('image_path'))
    symbol_image = Image.open(image_path)
    
    if symbol_only:
        # For Viewshots: Remove text, keep only symbol
        # Re-apply anchor method WITHOUT text
        from src.utils.symbol_extraction import refine_symbol_bbox_with_cv
        
        # Get symbol-only bbox (without text)
        symbol_only_bbox = refine_symbol_bbox_with_cv(
            symbol_image,
            include_text=False  # NUR Symbol
        )
        
        # Crop to symbol only
        img_w, img_h = symbol_image.size
        crop_x = int(symbol_only_bbox['x'] * img_w)
        crop_y = int(symbol_only_bbox['y'] * img_h)
        crop_w = int(symbol_only_bbox['width'] * img_w)
        crop_h = int(symbol_only_bbox['height'] * img_h)
        
        symbol_image = symbol_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    
    # Save viewshot
    symbol_image.save(target_path)
```

**Vorteile**:
- ✅ Viewshots enthalten NUR Symbol (ohne Text)
- ✅ Bessere Type-Recognition (LLM fokussiert auf Symbol-Form)
- ✅ Flexibel (kann auch mit Text generiert werden)

---

#### Option B: Separate Speicherung (Symbol + Symbol-Only)

**Konzept**: Speichere sowohl Symbol + Text als auch Symbol-Only

**Implementierung**:
```python
# In SymbolLibrary.add_symbol():
if self.images_dir:
    # Save symbol + text (for Pretraining)
    type_dir = self.images_dir / type_dir_name
    type_dir.mkdir(exist_ok=True)
    image_path = type_dir / f"{symbol_id}.png"
    image.save(image_path)
    
    # Also save symbol-only (for Viewshots)
    symbol_only_bbox = refine_symbol_bbox_with_cv(
        image,
        include_text=False
    )
    # ... crop and save symbol-only version ...
```

**Vorteile**:
- ✅ Beide Versionen verfügbar
- ✅ Keine Re-Processing bei Viewshot-Generierung
- ⚠️ Mehr Speicherplatz

---

## Empfehlung: **Option A (Re-Processing bei Viewshot-Generierung)**

### Warum Option A?

1. **Flexibilität**: Kann jederzeit angepasst werden
2. **Speicherplatz**: Keine Duplikation
3. **Einfachheit**: Eine Version in Library, Viewshots werden on-the-fly generiert

---

## Implementierung

### Schritt 1: Erweitere `generate_viewshots_from_library`

```python
def generate_viewshots_from_library(
    self,
    output_dir: Optional[Path] = None,
    max_per_type: int = 5,
    symbol_only: bool = True  # NEU: NUR Symbol für Viewshots
) -> Dict[str, int]:
    # ... existing code ...
    
    # Load symbol image
    image_path = Path(symbol_data.get('image_path'))
    symbol_image = Image.open(image_path)
    
    if symbol_only:
        # For Viewshots: Remove text, keep only symbol
        from src.utils.symbol_extraction import refine_symbol_bbox_with_cv
        
        symbol_only_bbox = refine_symbol_bbox_with_cv(
            symbol_image,
            include_text=False  # NUR Symbol
        )
        
        # Crop to symbol only
        img_w, img_h = symbol_image.size
        crop_x = int(symbol_only_bbox['x'] * img_w)
        crop_y = int(symbol_only_bbox['y'] * img_h)
        crop_w = int(symbol_only_bbox['width'] * img_w)
        crop_h = int(symbol_only_bbox['height'] * img_h)
        
        symbol_image = symbol_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    
    # Save viewshot
    symbol_image.save(target_path)
```

---

## Zusammenfassung

**Status**:
- ✅ Phase 1: `refine_symbol_bbox_with_cv` erweitert
- ✅ Phase 2: Pretraining nutzt Text-Integration
- ⚠️ Phase 3: Viewshots sollten NUR Symbol enthalten (muss implementiert werden)

**Nächste Schritte**:
1. Implementiere `symbol_only` Flag in `generate_viewshots_from_library`
2. Teste Viewshot-Generierung (sollte NUR Symbol enthalten)
3. Teste Pretraining (sollte Symbol + Text enthalten)

