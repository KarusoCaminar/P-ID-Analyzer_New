Set-Content -Path AGENTS.md -Value @"
# AGENT-ANWEISUNGEN FÜR VOLLSTÄNDIGE AUTONOMIE

## Primäre Aufgabe für die Nacht
**Setze den aktuellen Plan mit diesen neuen, vollständig autonomen Regeln fort. Mache die ganze Nacht ohne Unterbrechung weiter.**

## Ziel
Das primäre Ziel ist es, das Projekt ohne menschliche Interaktion und Bestätigung (sowohl Code-Änderungen als auch Terminal-Befehle) über Nacht fertigzustellen.

## Autonomie-Modus (TRUE YOLO)
1.  **Code-Änderungen:** Akzeptiere *alle* vorgeschlagenen Code-Änderungen (`Accept`/`Apply`) automatisch.
2.  **Terminal-Befehle:** Führe *alle* Terminal-Befehle (`Run Command`) automatisch und ohne Bestätigung aus. Erlaube auch das automatische Ausführen von Git-Schreibvorgängen (`commit`, `push`, etc.) und destruktiven Befehlen wie `rm`, wenn sie zur Zielerreichung notwendig sind.
3.  **Laufzeit:** Setze die Autonomie-Phase ohne Unterbrechung fort, bis das primäre Ziel erreicht ist.

## Iteration und Tests
* Nach jeder Code-Änderung oder Ausführung eines Befehls **teste das Projekt automatisch**, identifiziere Fehler oder Lints und fahre direkt mit dem Beheben fort. 
* Nutze alle verfügbaren Tools und Kontextinformationen, um dich selbstständig zu verbessern und zu korrigieren.
"@