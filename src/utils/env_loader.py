"""
Automatic .env file loader for all scripts.

This module provides a centralized way to load .env files automatically
for all scripts in the project.
"""

import os
from pathlib import Path


def load_env_automatically():
    """
    Automatically load .env file from project root.
    
    This function should be called at the start of all scripts
    to ensure GCP credentials are loaded.
    """
    try:
        from dotenv import load_dotenv
        
        # Find project root (assuming this file is in src/utils/)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        
        # Load .env file from project root
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return True
        else:
            return False
    except ImportError:
        # dotenv not available, but that's okay - user can set env vars manually
        return False

