"""
Configuration loader and project path setup.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Ensure directories exist
for path in [RAW_DIR, PROCESSED_DIR, LOG_DIR]:
    os.makedirs(path, exist_ok=True)

# Environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_KEYWORD = os.getenv("DEFAULT_KEYWORD", "Instagram update")

def get_config_summary():
    """Print configuration details."""
    print("\n⚙️  Configuration Summary")
    print(f"Root Dir: {ROOT_DIR}")
    print(f"Raw Dir: {RAW_DIR}")
    print(f"Processed Dir: {PROCESSED_DIR}")
    print(f"Log Dir: {LOG_DIR}")
    print(f"Default Keyword: {DEFAULT_KEYWORD}")
    print(f"API Key Loaded: {'✅ Yes' if YOUTUBE_API_KEY else '❌ No'}\n")
