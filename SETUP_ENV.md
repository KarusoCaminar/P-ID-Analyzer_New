# 🔧 .env Datei Einrichtung

## ⚠️ Wichtig: .env Datei manuell erstellen!

Die `.env` Datei ist aus Sicherheitsgründen nicht automatisch erstellbar.

## 📋 Anleitung

### Schritt 1: .env Datei erstellen

Erstelle eine `.env` Datei im Projekt-Root (`C:\Users\Moritz\Desktop\pid_analyzer_v2\.env`)

### Schritt 2: Folgenden Inhalt eintragen

```
GCP_PROJECT_ID=koretex-zugang
GCP_PROJECT_NUMBER=748084370989
GCP_LOCATION=us-central1
```

### Schritt 3: Speichern

Speichere die Datei als `.env` (ohne weitere Endung!)

## ✅ Überprüfung

Nach dem Erstellen der .env Datei, führe aus:

```bash
python test_system_ready.py
```

Du solltest sehen:
```
[OK] GCP_PROJECT_ID: koretex-zugang
[OK] Backend initialization
```

## 🔒 Sicherheit

⚠️ **Wichtig:** Die `.env` Datei ist in `.gitignore` und wird NICHT ins Git hochgeladen!

Die `.env.example` Datei dient als Vorlage und kann ins Git hochgeladen werden (ohne sensible Daten).

## 🚀 Danach

Nach der Erstellung kannst du starten:

```bash
# System-Check
python test_system_ready.py

# Erste Analyse
python run_cli.py path/to/image.png

# Oder GUI
python run_gui.py
```


