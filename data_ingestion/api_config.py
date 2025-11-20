"""
youtube_api_config.py
--------------------------------
Secure setup and connection handler for YouTube Data API v3.
Loads API key from .env file and returns a reusable service object.
"""

from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_youtube_service():
    """
    Creates and returns an authenticated YouTube API service instance.

    Returns:
        youtube (googleapiclient.discovery.Resource): YouTube API service object
    """
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError("❌ YouTube API key not found. Set YOUTUBE_API_KEY in your .env file.")

    youtube = build("youtube", "v3", developerKey=api_key)
    return youtube


# Optional: Test connection
if __name__ == "__main__":
    youtube = get_youtube_service()

    request = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        maxResults=2,
        regionCode="US"
    )
    response = request.execute()

    print("✅ YouTube API Connection Successful!")
    for item in response["items"]:
        print(f"🎬 {item['snippet']['title']} — Views: {item['statistics']['viewCount']}")
