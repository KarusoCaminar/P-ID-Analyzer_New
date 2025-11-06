# Pretraining vs. Viewshots: Analyse und Konsolidierungsstrategie

## Problem-Identifikation

Es gibt aktuell **zwei separate Systeme** für Symbol-Extraktion, die ähnliche Aufgaben erfüllen, aber unterschiedliche Zwecke haben:

### 1. Pretraining-System (Active Learning)

**Zweck**: Aktives Lernen und Symbol-Library für Ähnlichkeitssuche

**Komponenten**:
- `src/analyzer/core/pretraining_orchestrator.py`
- `src/analyzer/learning/active_learner.py`
- `src/analyzer/learning/symbol_library.py`

**Input**:
- `pretraining_symbols/page_1_original_symbols.png`
- `pretraining_symbols/page_2_original_symbols.png`
- `pretraining_symbols/page_3_original_symbols.png`
- `pretraining_symbols/page_4_original_symbols.png`

**Output**:
- `learning_db.json` (Symbol-Library mit Embeddings)
- `pretraining_symbols/pretraining_report.json`

**Verwendung**:
- Symbol-Ähnlichkeitssuche während der Analyse
- Aktives Lernen aus neuen Symbolen
- Deduplizierung von Symbolen

**Speicherung**:
- In `SymbolLibrary` (Datenbank)
- Mit Embeddings für Ähnlichkeitssuche
- Mit Metadaten (source, label, element_type)

---

### 2. Viewshot-System (Visuelle Referenz)

**Zweck**: Visuelle Beispiele für LLM-Prompts

**Komponenten**:
- `scripts/utilities/extract_viewshots_from_pretraining_pdf.py`
- `scripts/utilities/extract_viewshots_from_uni_bilder.py`

**Input**:
- `training_data/pretraining_symbols/Pid-symbols-PDF_sammlung.png` (große Sammlung)
- `training_data/complex_pids/page_*_original.png` (mit Ground Truth)

**Output**:
- `training_data/viewshot_examples/{type}/{type}_{id}.png`
- Organisiert nach Typ (valve/, pump/, flow_sensor/, etc.)

**Verwendung**:
- Visuelle Referenz in LLM-Prompts (`config.yaml`)
- "Few-shot examples" für bessere Typ-Erkennung
- Direkt in Prompts eingebettet (als Bild-Pfade)

**Speicherung**:
- Als PNG-Dateien in Ordnern
- Nach Typ organisiert
- Ohne Embeddings (nur visuelle Referenz)

---

## Vergleich

| Aspekt | Pretraining | Viewshots |
|--------|-------------|-----------|
| **Zweck** | Aktives Lernen, Ähnlichkeitssuche | Visuelle Referenz für LLM |
| **Speicherung** | Datenbank (JSON) mit Embeddings | PNG-Dateien in Ordnern |
| **Verwendung** | Runtime-Ähnlichkeitssuche | Prompt-Integration |
| **Organisation** | Nach Symbol-ID | Nach Typ |
| **Metadaten** | Vollständig (source, label, type, embedding) | Minimal (nur Typ) |
| **Deduplizierung** | Ja (Ähnlichkeitssuche) | Nein |
| **Skalierung** | Unbegrenzt (Datenbank) | Begrenzt (3-5 pro Typ) |

---

## Problem: Duplikation und Verwirrung

### Aktuelle Probleme:

1. **Zwei separate Extraktions-Skripte**:
   - `extract_viewshots_from_pretraining_pdf.py` (neu)
   - `active_learner.py` (bestehend)
   - Beide extrahieren Symbole, aber für unterschiedliche Zwecke

2. **Zwei separate Input-Quellen**:
   - `pretraining_symbols/page_*_original_symbols.png` (Pretraining)
   - `training_data/pretraining_symbols/Pid-symbols-PDF_sammlung.png` (Viewshots)

3. **Zwei separate Output-Formate**:
   - `learning_db.json` (Pretraining)
   - `viewshot_examples/{type}/` (Viewshots)

4. **Keine klare Trennung**:
   - Unklar, welches System für welchen Zweck verwendet werden soll
   - Potenzielle Duplikation von Symbolen

---

## Lösungsstrategie: Konsolidierung

### Option 1: Einheitliches System mit zwei Output-Formaten

**Konzept**: Ein Extraktions-Skript, das beide Formate generiert

**Vorteile**:
- ✅ Keine Duplikation
- ✅ Einheitliche Quelle
- ✅ Konsistente Metadaten

**Nachteile**:
- ⚠️ Komplexeres Skript
- ⚠️ Zwei Output-Pfade müssen synchronisiert werden

**Implementierung**:
```python
def extract_symbols_unified(
    input_image: Path,
    output_pretraining: Path,  # Für Symbol-Library
    output_viewshots: Path     # Für LLM-Prompts
) -> Dict[str, Any]:
    # 1. Extrahiere Symbole (einmalig)
    symbols = detect_and_extract_symbols(input_image)
    
    # 2. Identifiziere Typen (mit LLM)
    for symbol in symbols:
        symbol_type = identify_type_with_llm(symbol)
        symbol['type'] = symbol_type
    
    # 3. Speichere in Pretraining-Format (mit Embeddings)
    save_to_symbol_library(symbols, output_pretraining)
    
    # 4. Speichere in Viewshot-Format (als PNG)
    save_to_viewshots(symbols, output_viewshots)
    
    return stats
```

---

### Option 2: Klare Trennung mit gemeinsamer Basis

**Konzept**: Gemeinsame Extraktions-Logik, aber separate Output-Handler

**Vorteile**:
- ✅ Klare Trennung der Zwecke
- ✅ Einfache Wartung
- ✅ Flexible Erweiterung

**Nachteile**:
- ⚠️ Zwei separate Skripte
- ⚠️ Potenzielle Duplikation (aber kontrolliert)

**Implementierung**:
```python
# Gemeinsame Basis
def extract_symbols_base(image_path: Path) -> List[Dict[str, Any]]:
    """Gemeinsame Extraktions-Logik"""
    symbols = detect_symbols_cv(image_path)
    symbols = identify_types_llm(symbols)
    return symbols

# Pretraining-Handler
def save_to_pretraining(symbols: List[Dict], output_path: Path):
    """Speichere in Symbol-Library-Format"""
    for symbol in symbols:
        embedding = generate_embedding(symbol['image'])
        symbol_library.add_symbol(
            symbol_id=symbol['id'],
            image=symbol['image'],
            element_type=symbol['type'],
            embedding=embedding,
            metadata=symbol['metadata']
        )

# Viewshot-Handler
def save_to_viewshots(symbols: List[Dict], output_path: Path):
    """Speichere in Viewshot-Format"""
    for symbol in symbols:
        type_dir = output_path / symbol['type'].lower().replace(' ', '_')
        type_dir.mkdir(exist_ok=True)
        symbol['image'].save(type_dir / f"{symbol['type']}_{symbol['id']}.png")
```

---

### Option 3: Viewshots aus Pretraining generieren

**Konzept**: Viewshots werden aus der bestehenden Symbol-Library generiert

**Vorteile**:
- ✅ Keine Duplikation
- ✅ Einheitliche Quelle (Symbol-Library)
- ✅ Automatische Synchronisation

**Nachteile**:
- ⚠️ Abhängigkeit von Pretraining
- ⚠️ Viewshots müssen aus Datenbank generiert werden

**Implementierung**:
```python
def generate_viewshots_from_library(
    symbol_library: SymbolLibrary,
    output_path: Path,
    max_per_type: int = 5
) -> Dict[str, int]:
    """Generiere Viewshots aus Symbol-Library"""
    stats = {}
    
    # Gruppiere nach Typ
    symbols_by_type = symbol_library.get_symbols_by_type()
    
    for symbol_type, symbols in symbols_by_type.items():
        # Nimm die ersten N Symbole pro Typ
        viewshot_symbols = symbols[:max_per_type]
        
        type_dir = output_path / symbol_type.lower().replace(' ', '_')
        type_dir.mkdir(exist_ok=True)
        
        for idx, symbol in enumerate(viewshot_symbols):
            image = symbol_library.get_symbol_image(symbol['id'])
            image.save(type_dir / f"{symbol_type}_{idx:04d}.png")
            stats[symbol_type] = stats.get(symbol_type, 0) + 1
    
    return stats
```

---

## Empfehlung: Option 3 (Viewshots aus Pretraining)

### Warum Option 3?

1. **Keine Duplikation**: Viewshots werden aus der bestehenden Symbol-Library generiert
2. **Einheitliche Quelle**: Alle Symbole kommen aus dem Pretraining-System
3. **Automatische Synchronisation**: Wenn neue Symbole gelernt werden, können Viewshots neu generiert werden
4. **Einfache Wartung**: Nur ein System muss gepflegt werden

### Implementierung:

1. **Erweitere `active_learner.py`**:
   - Füge Methode `generate_viewshots_from_library()` hinzu
   - Generiere Viewshots nach Pretraining

2. **Erweitere `extract_viewshots_from_pretraining_pdf.py`**:
   - Option A: Nutze Pretraining-System, dann generiere Viewshots
   - Option B: Ersetze durch Wrapper, der Pretraining + Viewshot-Generierung aufruft

3. **Konsolidiere Input-Quellen**:
   - Nutze `Pid-symbols-PDF_sammlung.png` als Input für Pretraining
   - Generiere Viewshots automatisch nach Pretraining

---

## Nächste Schritte

1. **Analyse**: Prüfe, ob `Pid-symbols-PDF_sammlung.png` bereits im Pretraining-System verarbeitet wurde
2. **Konsolidierung**: Implementiere Option 3 (Viewshots aus Pretraining generieren)
3. **Dokumentation**: Kläre die Trennung zwischen Pretraining (Lernen) und Viewshots (Referenz)
4. **Cleanup**: Entferne doppelte Extraktions-Logik

---

## Zusammenfassung

**Aktueller Status**:
- ✅ Pretraining-System: Funktioniert, speichert in Symbol-Library
- ✅ Viewshot-System: Funktioniert, speichert als PNG-Dateien
- ❌ Problem: Zwei separate Systeme, potenzielle Duplikation

**Empfohlene Lösung**:
- ✅ Viewshots aus Pretraining generieren (Option 3)
- ✅ Einheitliche Quelle (Symbol-Library)
- ✅ Automatische Synchronisation

**Vorteile**:
- Keine Duplikation
- Einfache Wartung
- Konsistente Metadaten
- Automatische Aktualisierung

