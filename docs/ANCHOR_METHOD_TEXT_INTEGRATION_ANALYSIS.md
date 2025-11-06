# Anchor-Methode & Text-Integration: Analyse

## Problem-Identifikation

### Aktuelle Situation

1. **Anchor-Methode** (`refine_symbol_bbox_with_cv`):
   - Findet Contours des Symbols
   - Findet Center of Mass
   - Zentriert BBox auf Center of Mass
   - Verhindert, dass Symbole abgeschnitten werden
   - ✅ **Funktioniert gut für Symbole allein**

2. **Text-Integration**:
   - `_find_nearby_text` findet Text-Regionen in der Nähe von Symbolen
   - Text-Regionen werden in `symbol_region['text_regions']` gespeichert
   - ❌ **ABER: Text-Regionen werden NICHT in die BBox-Refinement einbezogen!**

### Problem

**Wenn ein Symbol ein Text-Label hat** (z.B. "P-101" neben einer Pumpe):
- Die Anchor-Methode zentriert nur auf dem Symbol selbst
- Das Text-Label wird **nicht** in die BBox einbezogen
- **Ergebnis**: Text-Label wird abgeschnitten oder fehlt im Crop

**Beispiel**:
```
[Symbol] "P-101"
  ↑
  └─ Anchor-Methode zentriert nur auf [Symbol]
     Text "P-101" wird nicht einbezogen
```

---

## Lösung: Text-Integration in Anchor-Methode

### Option 1: Erweitere `refine_symbol_bbox_with_cv` um Text-Integration

**Konzept**: Wenn Text-Regionen vorhanden sind, erweitere die BBox, um sie einzuschließen.

**Implementierung**:
```python
def refine_symbol_bbox_with_cv(
    symbol_image: Image.Image,
    initial_bbox: Optional[Dict[str, float]] = None,
    use_anchor_method: bool = True,
    text_regions: Optional[List[Dict[str, int]]] = None  # NEU
) -> Dict[str, float]:
    """
    Refine symbol bounding box using Computer Vision with anchor-based centering.
    
    Now also includes nearby text regions in the bbox if provided.
    """
    # ... existing anchor method code ...
    
    # AFTER anchor-based centering:
    if text_regions and len(text_regions) > 0:
        # Expand bbox to include text regions
        min_x = x
        min_y = y
        max_x = x + w
        max_y = y + h
        
        for text_region in text_regions:
            tx = text_region['x']
            ty = text_region['y']
            tw = text_region['width']
            th = text_region['height']
            
            # Expand bbox to include text
            min_x = min(min_x, tx)
            min_y = min(min_y, ty)
            max_x = max(max_x, tx + tw)
            max_y = max(max_y, ty + th)
        
        # Update bbox
        x = max(0, min_x)
        y = max(0, min_y)
        w = min(img_width - x, max_x - x)
        h = min(img_height - y, max_y - y)
    
    # ... rest of code ...
```

---

### Option 2: Erweitere `_extract_symbols_from_collection` um Text-Integration

**Konzept**: Nutze `text_regions` aus CV-Extraktion und übergebe sie an `refine_symbol_bbox_with_cv`.

**Implementierung** (`active_learner.py`):
```python
# In _extract_symbols_from_collection:
for idx, cv_symbol in enumerate(cv_symbols):
    # ... existing code ...
    
    # Get text regions from CV extraction
    text_regions = cv_symbol.get('text_regions', [])
    
    # Refine bbox with CV (remove white space) + include text
    refined_bbox = refine_symbol_bbox_with_cv(
        symbol_crop,
        text_regions=text_regions  # NEU: Übergebe Text-Regionen
    )
    
    # ... rest of code ...
```

---

## Empfehlung: Option 1 + Option 2 (Kombiniert)

### Warum beide?

1. **Option 1**: Macht `refine_symbol_bbox_with_cv` flexibler (kann Text einbeziehen, muss aber nicht)
2. **Option 2**: Nutzt die bereits erkannten Text-Regionen aus CV-Extraktion

### Vorteile

- ✅ Text-Labels werden nicht abgeschnitten
- ✅ Vollständige Symbol-Extraktion (Symbol + Label)
- ✅ Bessere OCR-Ergebnisse (Label ist im Crop enthalten)
- ✅ Rückwärtskompatibel (wenn keine Text-Regionen, funktioniert wie vorher)

---

## Wann ist Text-Integration wichtig?

### ✅ WICHTIG für Pretraining:

- **Legenden-Symbole**: Haben oft Text-Labels (z.B. "Pumpe", "Ventil")
- **PDF-Sammlungen**: Symbole haben oft Beschriftungen
- **Bessere OCR**: Wenn Label im Crop enthalten ist, ist OCR genauer

### ⚠️ NICHT WICHTIG für Diagramm-Analyse:

- **Diagramm-Elemente**: Haben Labels, aber diese werden separat erkannt
- **BBox-Refinement**: Sollte nur das Symbol selbst enthalten (für Type-Detection)

---

## Implementierung

### Schritt 1: Erweitere `refine_symbol_bbox_with_cv`

```python
def refine_symbol_bbox_with_cv(
    symbol_image: Image.Image,
    initial_bbox: Optional[Dict[str, float]] = None,
    use_anchor_method: bool = True,
    text_regions: Optional[List[Dict[str, int]]] = None  # NEU
) -> Dict[str, float]:
    # ... existing anchor method code ...
    
    # AFTER anchor-based centering:
    if text_regions and len(text_regions) > 0:
        # Expand bbox to include text regions
        # ... (siehe oben) ...
    
    # ... rest of code ...
```

### Schritt 2: Nutze Text-Regionen in `_extract_symbols_from_collection`

```python
# In _extract_symbols_from_collection:
text_regions = cv_symbol.get('text_regions', [])
refined_bbox = refine_symbol_bbox_with_cv(
    symbol_crop,
    text_regions=text_regions  # NEU
)
```

---

## Zusammenfassung

**Problem**: Text-Labels werden nicht in die BBox einbezogen → werden abgeschnitten

**Lösung**: Erweitere Anchor-Methode um Text-Integration

**Vorteile**:
- ✅ Vollständige Symbol-Extraktion (Symbol + Label)
- ✅ Bessere OCR-Ergebnisse
- ✅ Rückwärtskompatibel

**Wichtig für**: Pretraining (Legenden, PDF-Sammlungen)

**Nicht wichtig für**: Diagramm-Analyse (Labels werden separat erkannt)

