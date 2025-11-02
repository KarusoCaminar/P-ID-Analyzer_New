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

# Run CLI
if __name__ == "__main__":
    from src.analyzer.cli import main
    main()


