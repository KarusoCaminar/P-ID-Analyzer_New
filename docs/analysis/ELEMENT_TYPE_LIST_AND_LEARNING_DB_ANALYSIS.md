# Analyse: element_type_list.json und learning_db.json

## 1. element_type_list.json

### Status: ✅ ESSENZIELL - DARF NICHT ENTFERNT WERDEN

### Zweck:
- **Komponenten-Schema / Stammdaten-Katalog**: Definiert, was ein Element ist
- **Beantwortet**: "Was ist ein 'Druckhalteautomat'?" (nicht "Gibt es 'P-101'?")
- **Enthält**: Blaupausen für jeden Komponententyp, inklusive Ports und Vererbung

### Verwendung:
1. **Mit Legende**: 
   - Phase 1 extrahiert `symbol_map` aus Legende
   - Phase 2 (Swarm/Monolith) verwendet `symbol_map` als striktes Filterkriterium
   - `element_type_list.json` liefert das Schema für die Typen (Ports, Vererbung)

2. **Ohne Legende (Fallback)**:
   - Phase 2 (Swarm/Monolith) verwendet `element_type_list.json` als Basis
   - `get_known_types()` liefert Liste aller verfügbaren Typen
   - LLM kann "best guess"-Klassifizierung vornehmen

### Implementierung:
- ✅ `KnowledgeManager._load_config_library()` lädt `element_type_list.json`
- ✅ `KnowledgeManager.get_known_types()` liefert Liste aller Typen
- ✅ `KnowledgeManager.find_element_type_by_name()` findet Typ-Schema
- ✅ Wird in Prompts als `{known_types_json}` verwendet

## 2. learning_db.json

### Status: ✅ FUNKTIONIERT - BEREIT FÜR AKTIVES LERNEN

### Zweck:
- **Zukünftiges Wissen**: Speichert Korrekturen, Aliase, Patterns
- **Aktuell**: Leer (bereit für neues Wissen)
- **Wird verwendet**: Wenn `use_active_learning: true` (aktuell deaktiviert)

### Verwendung:
1. **Aktuell (deaktiviert)**:
   - `use_active_learning: false` in `config.yaml`
   - `learning_db.json` wird geladen, aber nicht geschrieben
   - Bereit für zukünftige Aktivierung

2. **Zukünftig (aktiviert)**:
   - Phase 3 (Self-Correction) speichert Korrekturen
   - Aliase werden gespeichert (z.B. "Pumpe P-101" = "P101")
   - Patterns werden gelernt

### Implementierung:
- ✅ `KnowledgeManager._load_learning_database()` lädt `learning_db.json`
- ✅ `KnowledgeManager._init_empty_database()` erstellt leere Struktur
- ✅ `ActiveLearner` kann in `learning_db.json` schreiben (wenn aktiviert)
- ✅ `SymbolLibrary` kann Symbole speichern (wenn aktiviert)

## 3. symbol_map als striktes Filterkriterium

### Status: ⚠️ TEILWEISE IMPLEMENTIERT - MUSS VERBESSERT WERDEN

### Aktueller Stand:
- ✅ `symbol_map` wird in Phase 1 extrahiert
- ✅ `legend_context` wird an Swarm/Monolith weitergegeben
- ✅ `legend_context` wird in Prompts als Few-Shot-Beispiele verwendet
- ⚠️ **FEHLT**: `symbol_map` als striktes Filterkriterium (nur diese Typen erlauben)

### Empfehlung:
1. **Wenn Legende vorhanden**:
   - Prompts müssen `symbol_map` als striktes Filterkriterium verwenden
   - Anweisung: "Analysiere das Bild, aber erkenne AUSSCHLIESSLICH die Symbole, die in diesem JSON-Objekt definiert sind: {symbol_map}"
   - `element_type_list.json` wird weiterhin für Schema-Informationen verwendet

2. **Wenn keine Legende vorhanden**:
   - Prompts bleiben wie bisher
   - Verwenden `element_type_list.json` als Basis
   - LLM kann "best guess"-Klassifizierung vornehmen

### Nächste Schritte:
1. Prompts anpassen, um `symbol_map` als striktes Filterkriterium zu verwenden
2. Logik implementieren: Wenn `symbol_map` vorhanden, nur diese Typen erlauben
3. Fallback auf `element_type_list.json` wenn keine Legende vorhanden

## 4. Zusammenfassung

### element_type_list.json:
- ✅ **ESSENZIELL**: Darf nicht entfernt werden
- ✅ **FUNKTIONIERT**: Wird korrekt geladen und verwendet
- ✅ **FALLBACK**: Wird verwendet, wenn keine Legende vorhanden

### learning_db.json:
- ✅ **FUNKTIONIERT**: Wird korrekt geladen
- ✅ **BEREIT**: Bereit für aktives Lernen (aktuell deaktiviert)
- ✅ **SAUBER**: Leer, bereit für neues Wissen

### symbol_map:
- ✅ **EXTRAHIERT**: Wird in Phase 1 korrekt extrahiert
- ✅ **WEITERGEGEBEN**: Wird an Swarm/Monolith weitergegeben
- ⚠️ **VERBESSERUNG NÖTIG**: Muss als striktes Filterkriterium verwendet werden

