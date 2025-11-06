"""
Strategy Validation Test Runner mit Live-Log-Anzeige

Führt Strategy Validation Tests aus und zeigt Logs live im Terminal an.
"""

import sys
import os
from pathlib import Path

# Projekt-Root zum Pfad hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import des Haupt-Skripts
from scripts.validation.run_strategy_validation import main

if __name__ == "__main__":
    # Prüfe GCP-Credentials
    if not os.getenv("GCP_PROJECT_ID"):
        print("=" * 60)
        print("⚠️  GCP_PROJECT_ID nicht gesetzt!")
        print("=" * 60)
        print("\nBitte setzen Sie die Umgebungsvariablen:")
        print("  Windows PowerShell:")
        print("    $env:GCP_PROJECT_ID='dein_project_id'")
        print("    $env:GCP_LOCATION='us-central1'")
        print("\n  Windows CMD:")
        print("    set GCP_PROJECT_ID=dein_project_id")
        print("    set GCP_LOCATION=us-central1")
        print("\n  Oder erstellen Sie eine .env Datei im Projekt-Root:")
        print("    GCP_PROJECT_ID=dein_project_id")
        print("    GCP_LOCATION=us-central1")
        print("\n" + "=" * 60)
        sys.exit(1)
    
    # Führe Tests aus (Logs werden automatisch angezeigt)
    print("=" * 60)
    print("🚀 Strategy Validation Tests - Live Logs")
    print("=" * 60)
    print("\nLogs werden live angezeigt...\n")
    
    main()

