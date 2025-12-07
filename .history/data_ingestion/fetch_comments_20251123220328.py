"""
fetch_youtube_comments.py
----------------------------------
Fetches comments and metadata from YouTube using Data API.
"""

from data.api_config import get_youtube_service
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json


def search_videos_by_keyword(keyword, max_results=50, days_back=60):
    """
    Searches for videos related to a keyword published in the last specified days.

    Args:
        keyword (str): Keyword or topic to search.
        max_results (int): Number of videos to fetch.
        days_back (int): Number of days back to search (default 20 for t-20 to t-1).

    Returns:
        list: List of video IDs.
    """
    youtube = get_youtube_service()

    # Calculate publishedAfter as t-20 days
    published_after = (datetime.now() - timedelta(days=days_back)).isoformat() + 'Z'

    request = youtube.search().list(
        q=keyword,
        part="id,snippet",
        type="video",
        maxResults=max_results,
        publishedAfter=published_after,
        order="relevance"
    )
    response = request.execute()

    video_ids = [item["id"]["videoId"] for item in response["items"]]
    print(f"🔍 Found {len(video_ids)} videos for keyword '{keyword}' published in the last {days_back} days")
    return video_ids


def fetch_comments(video_id, max_results=None):
    """
    Fetches comments from a specific YouTube video.

    Args:
        video_id (str): YouTube video ID.
        max_results (int or None): Number of comments to retrieve. None fetches all comments.

    Returns:
        list: List of comment dictionaries.
    """
    youtube = get_youtube_service()
    comments = []
    next_page_token = None

    fetched_count = 0

    while True:
        request_max = 100
        if max_results is not None:
            request_max = min(100, max_results - fetched_count)
            if request_max <= 0:
                break

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=request_max,
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
            fetched_count += 1

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


def fetch_video_view_counts(video_ids):
    """
    Fetches view counts for a list of YouTube video IDs.

    Args:
        video_ids (list): List of YouTube video IDs.

    Returns:
        dict: Mapping from video ID to view count (int).
    """
    youtube = get_youtube_service()
    view_counts = {}
    # YouTube API allows up to 50 IDs per request
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i + 50]
        request = youtube.videos().list(
            part="statistics",
            id=",".join(batch_ids)
        )
        response = request.execute()
        for item in response.get("items", []):
            vid = item["id"]
            statistics = item.get("statistics", {})
            view_count_str = statistics.get("viewCount")
            view_counts[vid] = int(view_count_str) if view_count_str is not None else 0
    return view_counts


def run_ingestion_pipeline(keyword="Instagram update", max_video_results=50, max_comment_results=None, days_back=60):
    """
    Main ingestion function that runs the full pipeline:
    1. Searches videos by keyword
    2. Fetches comments
    3. Fetches video view counts
    4. Saves combined data to CSV/JSON

    Args:
        keyword (str): Search term for YouTube videos.
        max_video_results (int): Number of videos to fetch.
        max_comment_results (int or None): Number of comments to fetch per video. None fetches all comments.
        days_back (int): Number of days back to search.


    """
    print(f"\n🚀 Starting YouTube Data Ingestion for: '{keyword}'\n")

    all_comments = []
    video_ids = search_videos_by_keyword(keyword, max_results=max_video_results, days_back=days_back)

    total_videos = len(video_ids)
    print(f"📊 Total videos fetched: {total_videos}")

    # Fetch view counts for videos
    video_view_counts = fetch_video_view_counts(video_ids)

    for vid in video_ids:
        comments = fetch_comments(vid, max_results=max_comment_results)
        # Add view count to each comment dictionary
        for comment in comments:
            comment["view_count"] = video_view_counts.get(vid, 0)
        all_comments.extend(comments)

    if all_comments:
        save_comments(all_comments, keyword)
        total_comments = len(all_comments)
    else:
        print("⚠️ No comments fetched. Try a different keyword.")
        total_comments = 0

    print("\n✅ Data Ingestion Pipeline Completed Successfully.\n")


# Optional: Run directly
if __name__ == "__main__":
    run_ingestion_pipeline(keyword="Instagram update")
