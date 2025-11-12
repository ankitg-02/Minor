"""
fetch_youtube_comments.py
----------------------------------
Fetches comments and metadata from YouTube using Data API.
"""


from data_ingestion.api_config import get_youtube_service
import pandas as pd
from datetime import datetime
import time
import os
import json


def search_videos_by_keyword(keyword, max_results=5):
    """
    Searches for videos related to a keyword.

    Args:
        keyword (str): Keyword or topic to search.
        max_results (int): Number of videos to fetch.

    Returns:
        list: List of video IDs.
    """
    youtube = get_youtube_service()

    request = youtube.search().list(
        q=keyword,
        part="id",
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    video_ids = [item["id"]["videoId"] for item in response["items"]]
    print(f"🔍 Found {len(video_ids)} videos for keyword '{keyword}'")
    return video_ids


def fetch_comments(video_id, max_results=100):
    """
    Fetches comments from a specific YouTube video.

    Args:
        video_id (str): YouTube video ID.
        max_results (int): Number of comments to retrieve.

    Returns:
        list: List of comment dictionaries.
    """
    youtube = get_youtube_service()
    comments = []
    next_page_token = None

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

        time.sleep(1)  # Prevent hitting API rate limits

    print(f"💬 {len(comments)} comments fetched for video ID: {video_id}")
    return comments


def save_comments(comments, keyword):
    """
    Saves comments to CSV and JSON files.

    Args:
        comments (list): List of comment dictionaries.
        keyword (str): Keyword used for file naming.
    """
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = f"data/raw/comments_{keyword}_{timestamp}.csv"
    json_path = f"data/raw/comments_{keyword}_{timestamp}.json"

    pd.DataFrame(comments).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comments, f, indent=4, ensure_ascii=False)

    print(f"✅ Data saved: {csv_path}")
    print(f"✅ Data saved: {json_path}")


def run_ingestion_pipeline(keyword="Instagram update"):
    """
    Main ingestion function that runs the full pipeline:
    1. Searches videos by keyword
    2. Fetches comments
    3. Saves to CSV/JSON

    Args:
        keyword (str): Search term for YouTube videos.
    """
    print(f"\n🚀 Starting YouTube Data Ingestion for: '{keyword}'\n")

    all_comments = []
    video_ids = search_videos_by_keyword(keyword)

    for vid in video_ids:
        comments = fetch_comments(vid, max_results=100)
        all_comments.extend(comments)

    if all_comments:
        save_comments(all_comments, keyword)
    else:
        print("⚠️ No comments fetched. Try a different keyword.")

    print("\n✅ Data Ingestion Pipeline Completed Successfully.\n")


# Optional: Run directly
if __name__ == "__main__":
    run_ingestion_pipeline(keyword="Instagram update")
