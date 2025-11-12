"""
fetch_youtube_comments.py
Fetches comments and video details from YouTube Data API.
"""
from data_ingestion.youtube_api_config import get_youtube_service
import pandas as pd
from datetime import datetime
import time
import json
import os

def fetch_comments(video_id, max_results=100):
    youtube = get_youtube_service()
    comments, next_page_token = [], None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            order="time",
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "author": snippet.get("authorDisplayName"),
                "text": snippet.get("textDisplay"),
                "like_count": snippet.get("likeCount"),
                "published_at": snippet.get("publishedAt"),
                "fetched_at": datetime.now().isoformat()
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(1)

    return comments

def save_to_csv(comments, filename):
    df = pd.DataFrame(comments)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} comments to {filename}")

def run_ingestion(video_id):
    comments = fetch_comments(video_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to_csv(comments, f"data/raw/comments_{timestamp}.csv")
    return comments
