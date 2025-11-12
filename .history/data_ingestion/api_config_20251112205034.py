
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def get_youtube_service():
    """Return an authenticated YouTube API service."""
    try:
        from googleapiclient.discovery import build
        from utils.config import YOUTUBE_API_KEY
        from utils.exceptions import ConfigurationError
        
        if not YOUTUBE_API_KEY:
            raise ConfigurationError("YouTube API key not found in .env file")
        return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        
    except ImportError as e:
        raise ImportError(
            "Required packages missing. Please run:\n"
            "pip install google-api-python-client python-dotenv"
        ) from e
