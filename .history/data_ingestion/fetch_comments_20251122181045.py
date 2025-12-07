"""
fetch_youtube_comments.py
----------------------------------
Fetches comments and metadata from YouTube using Data API.
"""

from .api_config import get_youtube_service
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


# Optional: Run directly
if __name__ == "__main__":
    run_ingestion_pipeline(keyword="Instagram update")
