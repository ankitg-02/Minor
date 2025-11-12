"""
YouTube API configuration and authentication.
"""

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from googleapiclient.discovery import build
    from utils.config import YOUTUBE_API_KEY
    from utils.exceptions import ConfigurationError
except ImportError as e:
    raise ImportError(f"Required package missing. Please run: pip install google-api-python-client\nError: {e}")

def get_youtube_service():
    """Return an authenticated YouTube API service."""
    if not YOUTUBE_API_KEY:
        raise ConfigurationError("YOUTUBE_API_KEY missing in .env")
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
