#!/usr/bin/env python3
"""
Quick Start Script für P&ID Analyzer - CLI Mode

Vereinfachter Start für die Kommandozeile.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load .env file automatically
try:
    from src.utils.env_loader import load_env_automatically
    load_env_automatically()
except (ImportError, Exception):
    # Fallback: Try direct dotenv import
    try:
        from dotenv import load_dotenv
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass

# Run CLI
if __name__ == "__main__":
    from src.analyzer.cli import main
    main()


