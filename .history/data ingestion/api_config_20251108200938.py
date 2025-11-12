api_key="AIzaSyCBecJZLhpbdRRXuusshQwfzq4_1tGvYw8"
from googleapiclient.discovery import build

# api_key = "YOUR_API_KEY"
youtube = build("youtube", "v3", developerKey=api_key)

request = youtube.videos().list(
    part="snippet,statistics",
    chart="mostPopular",
    maxResults=2,
    regionCode="US"
)
response = request.execute()
print(response)

