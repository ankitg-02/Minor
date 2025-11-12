"""
Configuration loader and directory setup for project.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base project directory
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Ensure all folders exist
for folder in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOG_DIR]:
    os.makedirs(folder, exist_ok=True)

# Environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_KEYWORD = os.getenv("DEFAULT_KEYWORD", "Instagram update")
