#!/usr/bin/env python3
"""
Quick Start Script für P&ID Analyzer - GUI Mode

Vereinfachter Start für die GUI.
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
    if not load_env_automatically():
        print("WARNING: .env file not found!")
        print("Please create .env file with:")
        print("  GCP_PROJECT_ID=your_project_id")
        print("  GCP_LOCATION=us-central1")
        print()
except (ImportError, Exception):
    # Fallback: Try direct dotenv import
    try:
        from dotenv import load_dotenv
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        else:
            print("WARNING: .env file not found!")
    except ImportError:
        pass

# Run GUI
if __name__ == "__main__":
    try:
        from src.gui.optimized_gui import OptimizedGUI
        
        app = OptimizedGUI()
        app.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


