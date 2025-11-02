# Setup-Anleitung für P&ID Analyzer v2.0

## Schnellstart

### 1. Projekt-Clone/Verzeichnis

```bash
cd C:\Users\Moritz\Desktop\pid_analyzer_v2
```

### 2. Virtuelle Umgebung erstellen und aktivieren

```bash
# Erstelle virtuelle Umgebung
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### 3. Dependencies installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment-Variablen konfigurieren

Erstelle eine `.env` Datei im Projekt-Root:

```env
GCP_PROJECT_ID=dein-project-id
GCP_LOCATION=us-central1
```

### 5. Google Cloud Authentifizierung

```bash
# Google Cloud SDK muss installiert sein
gcloud auth application-default login
```

### 6. Projekt-Struktur prüfen

```bash
# Prüfe ob alle Dateien vorhanden sind
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
python -c "import json; json.load(open('element_type_list.json'))"
python -c "import json; json.load(open('learning_db.json'))"
```

## Verwendung

### Test der Basis-Komponenten

```python
# test_basic.py
import os
from dotenv import load_dotenv
from src.services.config_service import ConfigService
from src.analyzer.learning.knowledge_manager import KnowledgeManager
from src.analyzer.ai.llm_client import LLMClient

load_dotenv()

# Config laden
config_service = ConfigService("config.yaml")
config = config_service.get_raw_config()

# Services initialisieren
llm_client = LLMClient(
    project_id=os.getenv("GCP_PROJECT_ID"),
    default_location=os.getenv("GCP_LOCATION", "us-central1"),
    config=config
)

knowledge_manager = KnowledgeManager(
    element_type_list_path="element_type_list.json",
    learning_db_path="learning_db.json",
    llm_handler=llm_client,
    config=config
)

print("✓ Config Service geladen")
print("✓ LLM Client initialisiert")
print("✓ Knowledge Manager geladen")
print(f"✓ {len(knowledge_manager.get_known_types())} bekannte Typen geladen")
```

## Troubleshooting

### Problem: Module nicht gefunden

**Lösung:** Stelle sicher, dass das Projekt-Root im PYTHONPATH ist:

```bash
# Windows PowerShell
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Linux/Mac
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

### Problem: Vertex AI Authentifizierung fehlt

**Lösung:** 
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Problem: Config-Validation Fehler

**Lösung:** Prüfe `config.yaml` auf Syntax-Fehler:
```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

## Nächste Schritte

1. **Pipeline Coordinator implementieren** (siehe `IMPLEMENTATION_STATUS.md`)
2. **Analysis Components migrieren**
3. **Tests schreiben**
4. **GUI/CLI modernisieren**


