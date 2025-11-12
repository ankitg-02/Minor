"""
Fetches YouTube video comments using the YouTube Data API.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.logger import get_logger
    from utils.config import RAW_DIR
    from .file_helper import safe_filename
    from .exception import YouTubeAPIError, DataIngestionError, handle_exception
    from .timer import timer
    from .api_config import get_youtube_service
except ImportError as e:
    raise ImportError(f"Required module missing. Error: {e}\nMake sure utils package is in PYTHONPATH")

log = get_logger(__name__)

def search_videos(keyword: str, max_results: int = 5):
    """Search videos by keyword."""
    youtube = get_youtube_service()
    req = youtube.search().list(q=keyword, part="id", type="video", maxResults=max_results)
    res = req.execute()
    vids = [i["id"]["videoId"] for i in res.get("items", [])]
    log.info(f"🔍 Found {len(vids)} videos for '{keyword}'")
    return vids

def fetch_comments(video_id: str, max_results: int = 100):
    """Fetch comments for a given video ID."""
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
        for it in res.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "author": s.get("authorDisplayName"),
                "text": s.get("textDisplay"),
                "like_count": s.get("likeCount"),
                "published_at": s.get("publishedAt"),
                "fetched_at": datetime.utcnow().isoformat() + "Z"
            })
        next_token = res.get("nextPageToken")
        if not next_token: break
        time.sleep(1)
    log.info(f"💬 {len(comments)} comments fetched for {video_id}")
    return comments

@timer
def run_ingestion(keyword="Instagram update", videos=3, comments_per_video=100):
    """Full data ingestion workflow."""
    try:
        video_ids = search_videos(keyword, max_results=videos)
        if not video_ids:
            raise YouTubeAPIError("No videos found.")

        all_comments = []
        for vid in video_ids:
            all_comments.extend(fetch_comments(vid, comments_per_video))

        if not all_comments:
            raise DataIngestionError("No comments fetched.")

        os.makedirs(RAW_DIR, exist_ok=True)
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
