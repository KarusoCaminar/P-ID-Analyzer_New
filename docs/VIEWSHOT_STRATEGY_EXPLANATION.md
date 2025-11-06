# Viewshot-Strategie: Erklärung

## Was sind Viewshots?

**Viewshots** sind visuelle Referenzbilder, die dem LLM helfen, Symboltypen korrekt zu erkennen.

### Strategie:

1. **Pretraining (Symbol + Text):**
   - Symbole werden MIT Text gespeichert (für OCR-Label-Extraktion)
   - Dateiname: `{ocr_label}_{type}_{uuid}.png` (z.B. `P-201_Valve_abc123.png`)
   - Zweck: LLM lernt, welche Symbole welche Namen haben

2. **Viewshots (NUR Symbol, kein Text):**
   - Viewshots werden OHNE Text generiert (nur Symbol-Form)
   - Dateiname: `{ocr_label}_{type}_{idx}.png` (z.B. `P-201_Valve_0000.png`)
   - Zweck: LLM sieht visuelle Muster ohne Text-Ablenkung

3. **LLM-Prompt-Integration:**
   - Viewshots werden in LLM-Prompts eingefügt
   - Beispiel: "Hier sind Beispiele für Valve-Symbole: [Viewshot 1] [Viewshot 2]"
   - LLM verwendet diese visuellen Referenzen, um ähnliche Symbole zu erkennen

## Warum Dateinamen wichtig sind:

**Problem:** Wenn wir dem LLM nur ein Bild geben ohne Kontext, weiß es nicht, was das Bild bedeutet.

**Lösung:** Dateinamen enthalten OCR-Label + Typ, z.B.:
- `P-201_Valve_abc123.png` → LLM weiß: "Das ist ein Valve mit Label P-201"
- `FT-10_Volume_Flow_Sensor_def456.png` → LLM weiß: "Das ist ein Volume Flow Sensor mit Label FT-10"

**Vorteil:**
- LLM kann Dateinamen lesen und verstehen, was das Bild bedeutet
- Dateinamen dienen als "Metadaten" für das Bild
- Bessere Typ-Erkennung durch visuelle Referenz + Kontext

## Workflow:

1. **Pretraining:**
   - PDF-Sammlung → CV extrahiert Symbole
   - OCR extrahiert Labels (z.B. "P-201", "FT-10")
   - LLM erkennt Typ (z.B. "Valve", "Volume Flow Sensor")
   - Speichern: `{ocr_label}_{type}_{uuid}.png` MIT Text

2. **Viewshot-Generierung:**
   - Aus Symbol-Bibliothek: Beste Beispiele pro Typ
   - Crop: NUR Symbol, kein Text
   - Speichern: `{ocr_label}_{type}_{idx}.png` OHNE Text

3. **LLM-Analyse:**
   - Viewshots werden in Prompts eingefügt
   - LLM sieht visuelle Muster + Dateinamen (Kontext)
   - Bessere Typ-Erkennung durch Referenz

## Evaluierung:

Das Skript `scripts/training/evaluate_extracted_symbols.py` prüft:

1. **Naming:**
   - Haben Symbole OCR-Labels (echte Namen)?
   - Oder nur generische Namen (`Pid-symbols-PDF_sammlung_sym_0`)?

2. **Completeness:**
   - Sind Symbole vollständig (nicht abgeschnitten)?
   - Oder berühren sie Bildränder (cut off)?

3. **Text Content:**
   - Enthalten Pretraining-Bilder Text (für OCR)?
   - Oder nur Symbol-Form?

4. **Viewshot Quality:**
   - Sind Viewshots korrekt (nur Symbol, kein Text)?
   - Haben sie aussagekräftige Dateinamen?

## Nächste Schritte:

1. ✅ Dateinamen-Logik korrigiert (OCR-Label + Typ + UUID)
2. ✅ Evaluierungs-Skript erstellt
3. ⏳ Evaluierung ausführen, um Probleme zu identifizieren
4. ⏳ Pretraining erneut ausführen mit korrigierten Dateinamen
5. ⏳ Viewshots generieren mit aussagekräftigen Namen

