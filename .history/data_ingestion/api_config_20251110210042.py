"""
YouTube API configuration and authentication.
"""

from googleapiclient.discovery import build
from utils.config import YOUTUBE_API_KEY
from utils.exceptions import ConfigurationError

def get_youtube_service():
    """Return an authenticated YouTube API service."""
    if not YOUTUBE_API_KEY:
        raise ConfigurationError("YOUTUBE_API_KEY missing in .env")
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
