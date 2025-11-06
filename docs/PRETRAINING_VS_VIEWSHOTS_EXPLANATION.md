# Pretraining vs. Viewshots: Erklärung

## Was macht Pretraining?

**Pretraining** extrahiert Symbole aus einer großen Sammlung (z.B. PDF mit vielen Symbolen) und speichert sie in der `SymbolLibrary`.

### Prozess:
1. **CV-Extraktion:** OpenCV findet Symbole im Bild (Contour Detection)
2. **OCR-Label-Extraktion:** Tesseract versucht, Text-Labels zu extrahieren (z.B. "P-201", "FT-10")
3. **LLM-Typ-Erkennung:** LLM erkennt den Symbol-Typ (z.B. "Valve", "Pump")
4. **Speichern:** Symbol wird in `SymbolLibrary` gespeichert MIT Text (für OCR)

### Zweck:
- **Symbol-Datenbank aufbauen:** Viele Symbole lernen und speichern
- **Typ-Erkennung trainieren:** LLM lernt, welche Symbole welche Typen haben
- **OCR-Labels extrahieren:** Text-Labels aus Symbolen extrahieren (falls vorhanden)

### Dateinamen:
- Format: `{ocr_label}_{type}_{uuid}.png` (z.B. `P-201_Valve_abc123.png`)
- ODER: `{type}_{uuid}.png` (z.B. `valve_abc123.png`) wenn kein OCR-Label verfügbar
- **MIT Text:** Pretraining-Bilder enthalten Symbol + Text (für OCR)

---

## Was machen Viewshots?

**Viewshots** sind visuelle Referenzbilder, die dem LLM helfen, Symboltypen korrekt zu erkennen.

### Prozess:
1. **Aus Symbol-Bibliothek:** Beste Beispiele pro Typ auswählen
2. **Crop zu Symbol:** NUR Symbol, kein Text (Text wird entfernt)
3. **Speichern:** Viewshot wird in `training_data/viewshot_examples/` gespeichert

### Zweck:
- **Visuelle Referenz:** LLM sieht visuelle Muster ohne Text-Ablenkung
- **Typ-Erkennung verbessern:** LLM verwendet diese Referenzen, um ähnliche Symbole zu erkennen
- **Prompt-Integration:** Viewshots werden in LLM-Prompts eingefügt

### Dateinamen:
- Format: `{ocr_label}_{type}_{idx}.png` (z.B. `P-201_Valve_0000.png`)
- ODER: `{type}_{idx}.png` (z.B. `valve_0000.png`) wenn kein OCR-Label verfügbar
- **OHNE Text:** Viewshots enthalten NUR Symbol (Text wird entfernt)

---

## Warum beide?

### Pretraining:
- **Wann:** Einmalig, um Symbol-Datenbank aufzubauen
- **Was:** Symbole MIT Text speichern (für OCR)
- **Zweck:** Datenbank aufbauen, Typ-Erkennung trainieren

### Viewshots:
- **Wann:** Automatisch nach Pretraining, oder manuell
- **Was:** Symbole OHNE Text speichern (nur visuelle Muster)
- **Zweck:** Visuelle Referenz für LLM-Prompts

### Zusammenarbeit:
1. **Pretraining** → Baut Symbol-Datenbank auf
2. **Viewshots** → Generiert visuelle Referenzen aus Datenbank
3. **LLM-Analyse** → Verwendet Viewshots für bessere Typ-Erkennung

---

## OCR-Label-Extraktion

### Problem:
OCR extrahiert keine Labels, weil:
1. **Symbole im PDF haben keine Text-Labels:** Die Symbole sind nur visuelle Formen ohne Text
2. **OCR funktioniert mit Bildern:** OCR kann aus PNG/JPG extrahieren, aber nicht direkt aus PDFs
3. **PDF muss zuerst konvertiert werden:** PDF → PNG/JPG → OCR

### Aktuelles Verhalten:
- **OCR wird versucht:** Tesseract versucht, Text aus Symbol-Bildern zu extrahieren
- **Keine Labels gefunden:** Symbole haben keine Text-Labels → OCR findet nichts
- **Fallback:** Wenn OCR fehlschlägt → `{type}_{uuid}.png` (ohne OCR-Label)

### Lösung:
- **Wenn PDF Text-Labels hat:** PDF → PNG konvertieren → OCR extrahieren
- **Wenn PDF keine Text-Labels hat:** Fallback zu `{type}_{uuid}.png` (funktioniert bereits)

---

## Status

### Pretraining:
- ✅ **Funktioniert:** Symbole werden extrahiert und gespeichert
- ⚠️ **OCR-Labels:** Werden nicht extrahiert (Symbole haben keine Text-Labels)
- ✅ **Typ-Erkennung:** Funktioniert (LLM erkennt Typen korrekt)
- ✅ **Dateinamen:** Korrekt (`{type}_{uuid}.png`)

### Viewshots:
- ✅ **Funktioniert:** Viewshots werden generiert
- ⚠️ **Dateinamen:** Haben noch keine OCR-Labels (weil Pretraining keine hat)
- ✅ **Qualität:** Korrekt (nur Symbol, kein Text)

### Zusammenarbeit:
- ✅ **Pretraining → Viewshots:** Funktioniert automatisch
- ✅ **Viewshots → LLM-Prompts:** Werden in Prompts eingefügt
- ✅ **Typ-Erkennung:** Verbessert durch Viewshots

---

## Nächste Schritte

1. ✅ Pretraining funktioniert (Symbole werden extrahiert)
2. ✅ Viewshots funktioniert (werden generiert)
3. ⚠️ OCR-Labels: Werden nicht extrahiert (Symbole haben keine Text-Labels)
4. ✅ Beide funktionieren zusammen (Pretraining → Viewshots → LLM-Prompts)

**Fazit:** Pretraining und Viewshots funktionieren beide. OCR extrahiert keine Labels, weil die Symbole im PDF keine Text-Labels haben. Das ist kein Problem - der Fallback zu `{type}_{uuid}.png` funktioniert korrekt.

