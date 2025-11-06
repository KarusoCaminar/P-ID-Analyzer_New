# GUI Test-Protokoll

## Übersicht
Dieses Protokoll dokumentiert die Tests aller GUI-Funktionen und Features.

## Vorbereitung
- [ ] GUI gestartet (`python run_gui.py`)
- [ ] GCP_PROJECT_ID in `.env` gesetzt
- [ ] Testbilder verfügbar

## Test 1: GUI-Initialisierung
- [ ] GUI öffnet sich ohne Fehler
- [ ] Alle Tabs sichtbar (Analyse, Training, Visualisierungen)
- [ ] Status zeigt "Bereit" oder "Backend initialisiert"
- [ ] Log-Fenster zeigt keine kritischen Fehler

## Test 2: Analyse Tab - Datei-Auswahl
- [ ] "Dateien hinzufügen" Button funktioniert
- [ ] Datei-Dialog öffnet sich
- [ ] Mehrere Dateien können ausgewählt werden
- [ ] Dateien erscheinen in der Liste
- [ ] "Dateien entfernen" entfernt selektierte Dateien
- [ ] "Alle löschen" entfernt alle Dateien

## Test 3: Analyse Tab - Optionen
- [ ] Monolith Fusion Checkbox funktioniert
- [ ] Predictive Completion Checkbox funktioniert
- [ ] Polyline Refinement Checkbox funktioniert
- [ ] Self-Correction Checkbox funktioniert
- [ ] Alle Checkboxen haben Standard-Werte (True)

## Test 4: Analyse Tab - Model-Auswahl
- [ ] "Model-Auswahl pro Phase" Frame ist sichtbar
- [ ] Alle 7 Phasen-Modelle haben Dropdown-Menüs:
  - [ ] Meta Model
  - [ ] Hotspot Model
  - [ ] Detail Model
  - [ ] Coarse Model
  - [ ] Correction Model
  - [ ] Code Gen Model
  - [ ] Critic Model
- [ ] Dropdown-Menüs sind mit verfügbaren Modellen gefüllt
- [ ] Standard-Werte werden aus `default_flash` Strategie geladen
- [ ] Model-Auswahl kann geändert werden
- [ ] Model-Auswahl wird gespeichert

## Test 5: Analyse Tab - Analyse starten
- [ ] "Analyse starten" Button ist aktiv
- [ ] Button deaktiviert sich während Analyse läuft
- [ ] Progress Bar zeigt Fortschritt (0-100%)
- [ ] Status-Label zeigt aktuelle Phase
- [ ] Log-Fenster zeigt Analyse-Fortschritt
- [ ] Ergebnisse erscheinen nach Abschluss
- [ ] Quality Score wird angezeigt

## Test 6: Analyse Tab - Ergebnisse
- [ ] Ergebnisse werden im Textfeld angezeigt
- [ ] Quality Score ist sichtbar
- [ ] Element-Anzahl ist sichtbar
- [ ] Verbindungs-Anzahl ist sichtbar
- [ ] "Ergebnisse anzeigen" Button funktioniert (falls implementiert)

## Test 7: Training Tab
- [ ] Training Tab ist sichtbar
- [ ] Duration Input funktioniert (1-168 Stunden)
- [ ] Max Cycles Input funktioniert (0-1000)
- [ ] "Training Camp starten" Button funktioniert
- [ ] Training Status wird angezeigt
- [ ] Log zeigt Training-Fortschritt

## Test 8: Visualisierungen Tab
- [ ] Visualisierungen Tab ist sichtbar
- [ ] Visualisierungen werden nach Analyse angezeigt (falls implementiert)

## Test 9: Datenbank-Speicherung
- [ ] Learning Database wird nach Analyse gespeichert
- [ ] Symbol Library wird gespeichert
- [ ] Keine JSON-Serialisierungsfehler (BBox-Objekte werden konvertiert)
- [ ] Periodische Bereinigung funktioniert (alle 10 Analysen)

## Test 10: Symbol-Training
- [ ] Symbole werden aus Analysis-Ergebnissen extrahiert
- [ ] Symbole werden zur SymbolLibrary hinzugefügt
- [ ] Symbol-Embeddings werden generiert
- [ ] Symbol-Hints werden in SwarmAnalyzer verwendet
- [ ] Similarity-Search funktioniert

## Test 11: Datenbank-Bereinigung
- [ ] Bereinigung wird alle 10 Analysen automatisch durchgeführt
- [ ] Outdated entries werden entfernt (>90 Tage)
- [ ] Duplikate werden entfernt
- [ ] Low-quality entries werden entfernt (<0.5 confidence)
- [ ] recent_analyses wird auf max 100 begrenzt
- [ ] Bereinigungs-Report wird geloggt

## Bekannte Probleme
- [ ] Liste bekannter Probleme hier

## Verbesserungsvorschläge
- [ ] Liste Verbesserungsvorschläge hier

