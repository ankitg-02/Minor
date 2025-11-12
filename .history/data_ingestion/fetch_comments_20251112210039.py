"""
Fetch YouTube video comments and save them in /data/raw.
"""

from pathlib import Path
import sys
# Ensure project root is on sys.path so `utils` imports work when running the file directly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.config import RAW_DIR
from utils.file_helper import safe_filename
from utils.exceptions import YouTubeAPIError, DataIngestionError, handle_exception
from utils.timer import timer
from .youtube_api_config import get_youtube_service

import pandas as pd
import json
import os
import time
from datetime import datetime

log = get_logger(__name__)

def search_videos(keyword: str, max_results: int = 5):
    """Search for YouTube videos by keyword."""
    youtube = get_youtube_service()
    req = youtube.search().list(q=keyword, part="id", type="video", maxResults=max_results)
    res = req.execute()
    videos = [item["id"]["videoId"] for item in res.get("items", [])]
    log.info(f"🔍 Found {len(videos)} videos for '{keyword}'")
    return videos

def fetch_comments(video_id: str, max_results: int = 100):
    """Fetch comments for a given YouTube video."""
    youtube = get_youtube_service()
    comments, next_token = [], None
    while True:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results),
            pageToken=next_token,
            order="time"
        )
        res = req.execute()
        for item in res.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "author": snippet.get("authorDisplayName"),
                "text": snippet.get("textDisplay"),
                "like_count": snippet.get("likeCount"),
                "published_at": snippet.get("publishedAt"),
                "fetched_at": datetime.utcnow().isoformat() + "Z"
            })
        next_token = res.get("nextPageToken")
        if not next_token:
            break
        time.sleep(1)
    log.info(f"💬 {len(comments)} comments fetched for video {video_id}")
    return comments

@timer
def run_ingestion(keyword="Instagram update", videos=3, comments_per_video=100):
    """Complete ingestion pipeline saving data to data/raw."""
    try:
        log.info(f"🚀 Starting ingestion for '{keyword}'")
        video_ids = search_videos(keyword, videos)
        if not video_ids:
            raise YouTubeAPIError(f"No videos found for keyword: {keyword}")

        all_comments = []
        for vid in video_ids:
            all_comments.extend(fetch_comments(vid, comments_per_video))

        if not all_comments:
            raise DataIngestionError("No comments fetched from API.")

        # Ensure data/raw exists
        os.makedirs(RAW_DIR, exist_ok=True)

        # Save CSV + JSON
        filename = safe_filename("comments", keyword)
        csv_path = os.path.join(RAW_DIR, filename)
        pd.DataFrame(all_comments).to_csv(csv_path, index=False)

        json_path = csv_path.replace(".csv", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_comments, f, ensure_ascii=False, indent=2)

        log.info(f"✅ Saved {len(all_comments)} comments to {csv_path}")
        return csv_path

    except Exception as e:
        handle_exception(e, "data_ingestion")
        raise
